#!/usr/bin/env python3
"""
Compute motion-tracking metrics for SONIC deploy by reading the StateLogger CSVs.

Inputs:
  --ref-npz                  Reference NPZ (from build_ref_npz_from_pkl.py)
  --state-logger-dir         Directory containing q.csv, dq.csv, base_quat.csv (...)
  --urdf                     G1 fixed-base URDF (pelvis root)

Optional:
  --output                   metrics.json output path
  --joint-names              CLI override (29, IsaacLab order)
  --body-names               CLI override (subset)
  --anchor-body              default 'pelvis'
  --dz-thresh / --ori-thresh success thresholds (deploy has no global pelvis Z;
                             we ignore height-fail by default — see flag)
  --skip-success-height      drop the dz check (recommended for SONIC sim deploy)

Output JSON fields mirror eval_motion_metrics.py:
  joint_rmse_rad, frames_used, ref_fps, anchor_body
  E_mpjpe_m, E_vel_mm_s_ref/robot, E_acc_mm_s2_ref/robot, E_*_diff_*
  success_rate, frac_fail_*, mean_abs_dz, mean_ori_err_rad

q.csv on disk is in MuJoCo (hardware) order WITH default_angles offsets re-added.
We invert that: q_il[isaaclab_to_mujoco[i]] = q_csv[i] - default_angles[i].
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# Reuse helpers + constants
from build_ref_npz_from_pkl import (  # noqa: E402
    DEFAULT_BODY_NAMES,
    G1_JOINT_NAMES_MUJOCO,
    _fk_one,
    _g1_joint_names_isaaclab,
    _quat_wxyz_to_R,
)
from gmr_pkl_to_sonic_motion import ISAACLAB_TO_MUJOCO  # noqa: E402

# G1 hardware default_angles (must match policy_parameters.hpp).
G1_DEFAULT_ANGLES_MUJOCO = np.array([
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
    0.0, 0.0, 0.0,
    0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
], dtype=np.float64)


def _read_csv_with_index(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (t_seconds, values, value_headers).

    StateLogger CSV columns: index, time_ms, time_realtime_ms, time_monotonic_ms,
                             ros_timestamp, <values...>
    We use time_ms (column 1, normalised so first sample = 0 ms).
    """
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    if len(header) < 6:
        raise ValueError(f"Unexpected CSV header in {path}: {header}")
    data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    t_ms = data[:, 1]
    values = data[:, 5:]
    value_headers = header[5:]
    return t_ms * 1e-3, values, value_headers


def _csv_q_to_isaaclab(q_csv: np.ndarray) -> np.ndarray:
    """Hardware order with default_angles offsets -> IsaacLab order, offsets removed."""
    if q_csv.shape[1] != 29:
        raise ValueError(f"q.csv must have 29 joint columns; got {q_csv.shape[1]}")
    q_il = np.zeros_like(q_csv)
    raw = q_csv - G1_DEFAULT_ANGLES_MUJOCO[None, :]
    for i in range(29):
        q_il[:, ISAACLAB_TO_MUJOCO[i]] = raw[:, i]
    return q_il


def _resample_lin(t_src: np.ndarray, y_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    if y_src.ndim == 1:
        y_src = y_src.reshape(-1, 1)
    if t_src.size < 2:
        return np.repeat(y_src[:1], len(t_tgt), axis=0) if t_src.size else np.zeros((len(t_tgt), y_src.shape[1]))
    idx = np.searchsorted(t_src, t_tgt, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 2)
    t0, t1 = t_src[idx], t_src[idx + 1]
    a = np.clip((t_tgt - t0) / (t1 - t0 + 1e-12), 0.0, 1.0).reshape(-1, 1)
    return (1 - a) * y_src[idx] + a * y_src[idx + 1]


def _resample_quat_wxyz(t_src: np.ndarray, q_src: np.ndarray, t_tgt: np.ndarray) -> np.ndarray:
    """Linear in 4D + renormalise (cheap; close-enough at 50 Hz)."""
    out = _resample_lin(t_src, q_src, t_tgt)
    n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
    return out / n


def _bodies_in_anchor_frame(body_pos_w: np.ndarray, anchor_pos: np.ndarray, anchor_R: np.ndarray):
    rel = body_pos_w - anchor_pos[:, None, :]
    Rinv = np.transpose(anchor_R, (0, 2, 1))
    return np.einsum("tij,tbj->tbi", Rinv, rel)


def _vel_acc_mm_s(rel: np.ndarray, dt: float) -> tuple[float, float]:
    if rel.shape[0] < 3 or dt <= 0:
        return float("nan"), float("nan")
    v = np.diff(rel, axis=0) / dt
    a = np.diff(v, axis=0) / dt
    return (float(np.mean(np.linalg.norm(v * 1000.0, axis=-1))),
            float(np.mean(np.linalg.norm(a * 1000.0, axis=-1))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-npz", required=True)
    ap.add_argument("--state-logger-dir", required=True,
                    help="Dir containing q.csv, dq.csv, base_quat.csv")
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--output", default="")
    ap.add_argument("--joint-names", default="")
    ap.add_argument("--body-names", default="")
    ap.add_argument("--anchor-body", default="pelvis")
    ap.add_argument("--dz-thresh", type=float, default=0.25)
    ap.add_argument("--ori-thresh", type=float, default=1.0)
    ap.add_argument("--skip-success-height", action="store_true",
                    help="Ignore the dz failure component (recommended for SONIC sim).")
    args = ap.parse_args()

    log_dir = Path(args.state_logger_dir)
    q_csv = log_dir / "q.csv"
    bq_csv = log_dir / "base_quat.csv"
    if not q_csv.is_file() or not bq_csv.is_file():
        print(f"Error: missing q.csv or base_quat.csv in {log_dir}", file=sys.stderr)
        return 1

    # Reference.
    ref = np.load(args.ref_npz, allow_pickle=True)
    fps = float(np.asarray(ref["fps"]).ravel()[0])
    ref_joint = np.asarray(ref["joint_pos"], dtype=np.float64)
    ref_body_pos = np.asarray(ref["body_pos_w"], dtype=np.float64)
    ref_body_quat = np.asarray(ref["body_quat_w"], dtype=np.float64)
    ref_joint_names = [str(x) for x in np.asarray(ref["joint_names"]).ravel()]
    ref_body_names = [str(x) for x in np.asarray(ref["body_names"]).ravel()]

    # Joint / body name selection (defaults reuse build_ref_npz defaults).
    if args.joint_names.strip():
        joint_names = [s.strip() for s in args.joint_names.split(",") if s.strip()]
    else:
        joint_names = _g1_joint_names_isaaclab()
    if len(joint_names) != 29:
        print(f"Error: need 29 joint_names, got {len(joint_names)}", file=sys.stderr); return 1

    if args.body_names.strip():
        body_names = [s.strip() for s in args.body_names.split(",") if s.strip()]
    else:
        body_names = list(DEFAULT_BODY_NAMES)
    anchor_name = args.anchor_body
    if anchor_name not in body_names:
        print(f"Error: anchor '{anchor_name}' not in body_names", file=sys.stderr); return 1
    anchor_idx = body_names.index(anchor_name)

    # Validate reference contains requested bodies.
    try:
        ref_body_idx = [ref_body_names.index(b) for b in body_names]
    except ValueError as exc:
        print(f"Error: body name not found in ref NPZ: {exc}", file=sys.stderr); return 1
    ref_anchor_idx = ref_body_names.index(anchor_name)

    # Validate joint name match.
    try:
        joint_remap = [ref_joint_names.index(n) for n in joint_names]
    except ValueError as exc:
        print(f"Error: joint name not in ref NPZ: {exc}", file=sys.stderr); return 1
    ref_j = ref_joint[:, joint_remap]

    # Load deploy logs.
    t_q, q_raw, _ = _read_csv_with_index(q_csv)
    if q_raw.shape[1] != 29:
        print(f"Error: q.csv must have 29 cols, got {q_raw.shape[1]}", file=sys.stderr); return 1
    rob_j_all = _csv_q_to_isaaclab(q_raw)

    t_bq, bq_raw, _ = _read_csv_with_index(bq_csv)
    if bq_raw.shape[1] != 4:
        print(f"Error: base_quat.csv must have 4 cols (w,x,y,z); got {bq_raw.shape[1]}",
              file=sys.stderr); return 1

    # Time grid: ref dt at fps; clip to deploy duration.
    T_ref = ref_j.shape[0]
    ref_dt = 1.0 / fps
    t_ref_full = np.arange(T_ref) * ref_dt
    t_clip = min(float(t_ref_full[-1]), float(t_q[-1]), float(t_bq[-1]))
    T = max(1, int(np.searchsorted(t_ref_full, t_clip, side="right")))
    T = min(T, T_ref)
    t_ref = t_ref_full[:T]

    rob_j = _resample_lin(t_q, rob_j_all, t_ref)
    rob_bq = _resample_quat_wxyz(t_bq, bq_raw, t_ref)
    ref_j_T = ref_j[:T]

    # Joint RMSE.
    joint_rmse = float(np.sqrt(np.mean((ref_j_T - rob_j) ** 2)))

    # Robot body world poses via FK with base from base_quat.csv (no global pelvis pos).
    # We anchor at origin (p_root = 0); since metrics are relative to anchor, this is fine.
    import pinocchio as pin
    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()

    Tn = T
    rob_body_pos = np.zeros((Tn, len(body_names), 3), dtype=np.float64)
    rob_body_R = np.zeros((Tn, len(body_names), 3, 3), dtype=np.float64)
    p_zero = np.zeros(3, dtype=np.float64)
    for t in range(Tn):
        R_root = _quat_wxyz_to_R(rob_bq[t])
        positions, quats = _fk_one(model, data, joint_names, rob_j[t],
                                   body_names, R_root, p_zero)
        rob_body_pos[t] = positions
        for bi in range(len(body_names)):
            rob_body_R[t, bi] = _quat_wxyz_to_R(quats[bi])

    # Reference body slices.
    ref_b = ref_body_pos[:T][:, ref_body_idx, :]
    ref_bq_arr = ref_body_quat[:T][:, ref_body_idx, :]
    ref_anchor_pos = ref_body_pos[:T, ref_anchor_idx]
    ref_anchor_quat = ref_body_quat[:T, ref_anchor_idx]

    # Build per-frame anchor SE(3).
    rob_anchor_pos = rob_body_pos[:, anchor_idx]
    rob_anchor_R = rob_body_R[:, anchor_idx]
    ref_anchor_R = np.zeros_like(rob_anchor_R)
    for t in range(Tn):
        ref_anchor_R[t] = _quat_wxyz_to_R(ref_anchor_quat[t])

    # MPJPE in anchor frame (purely relative — robust to absent global root pos).
    rel_ref = _bodies_in_anchor_frame(ref_b, ref_anchor_pos, ref_anchor_R)
    rel_rob = _bodies_in_anchor_frame(rob_body_pos, rob_anchor_pos, rob_anchor_R)
    e_mpjpe = float(np.mean(np.linalg.norm(rel_ref - rel_rob, axis=-1)))

    ev_r, ea_r = _vel_acc_mm_s(rel_ref, ref_dt)
    ev_o, ea_o = _vel_acc_mm_s(rel_rob, ref_dt)

    # Success: orientation error of anchor relative to its initial; height fail
    # disabled (or controlled) since SONIC deploy has no world Z reference.
    ang = np.zeros(Tn, dtype=np.float64)
    for t in range(Tn):
        dR = ref_anchor_R[t].T @ rob_anchor_R[t]
        tr = float(np.clip(np.trace(dR), -1.0, 3.0))
        ang[t] = math.acos(np.clip(0.5 * (tr - 1.0), -1.0, 1.0))
    fail_ori = ang > args.ori_thresh
    if args.skip_success_height:
        fail = fail_ori
        frac_fail_height = float("nan")
        mean_abs_dz = float("nan")
    else:
        # Use anchor Z from FK (relative to root which we set to 0); compare ref's anchor z.
        # NOTE: ref_anchor pos is in world; rob_anchor is in URDF base frame because p_root=0.
        # So |dz| is meaningless without odometry — keeping the option for completeness.
        dz = np.abs(ref_anchor_pos[:, 2] - rob_anchor_pos[:, 2])
        fail_z = dz > args.dz_thresh
        fail = fail_ori | fail_z
        frac_fail_height = float(np.mean(fail_z))
        mean_abs_dz = float(np.mean(dz))

    out: dict[str, Any] = {
        "joint_rmse_rad": joint_rmse,
        "frames_used": int(Tn),
        "ref_fps": fps,
        "anchor_body": anchor_name,
        "E_mpjpe_m": e_mpjpe,
        "E_vel_mm_s_ref": ev_r,
        "E_acc_mm_s2_ref": ea_r,
        "E_vel_mm_s_robot": ev_o,
        "E_acc_mm_s2_robot": ea_o,
        "frac_fail_orientation": float(np.mean(fail_ori)),
        "frac_fail_height": frac_fail_height,
        "frac_fail_either": float(np.mean(fail)),
        "success_rate": float(1.0 - np.mean(fail)),
        "mean_abs_dz": mean_abs_dz,
        "mean_ori_err_rad": float(np.mean(ang)),
        "skip_success_height": bool(args.skip_success_height),
        "source": "sonic_state_logger",
    }
    if not math.isnan(ev_r) and not math.isnan(ev_o):
        out["E_vel_diff_mm_s"] = float(abs(ev_r - ev_o))
    if not math.isnan(ea_r) and not math.isnan(ea_o):
        out["E_acc_diff_mm_s2"] = float(abs(ea_r - ea_o))

    print(json.dumps(out, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
