#!/usr/bin/env python3
"""
Build a BeyondMimic-style reference NPZ from a GMR robot motion .pkl, so that
the SAME reference can be consumed by:
  - motion_tracking_controller/scripts/eval/eval_motion_metrics.py
  - gear_sonic_deploy/scripts/eval_sonic_metrics_from_logs.py

Pipeline:
  GMR .pkl --> joint_pos (IsaacLab order, 29) + root pose at target_fps
           --> Pinocchio FK on G1 fixed-base URDF (pelvis root)
           --> body_pos_w (T,B,3) and body_quat_w (T,B,4 wxyz) for --body-names

Output NPZ keys (matching `load_ref_npz` in eval_motion_metrics.py):
  fps, joint_pos, body_pos_w, body_quat_w, joint_names, body_names

Usage:
  python build_ref_npz_from_pkl.py \\
    --pkl <clip>.pkl \\
    --urdf .../g1_29dof.urdf \\
    --out  <clip>.npz

Optional CLI overrides:
  --target-fps 50
  --joint-names <comma-separated 29 names in IsaacLab order>
  --body-names  <comma-separated body names>  (defaults: 14 SONIC-eval bodies)
  --anchor-body pelvis
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# Reuse helpers from gmr_pkl_to_sonic_motion.py (same directory).
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from gmr_pkl_to_sonic_motion import (  # noqa: E402
    ISAACLAB_TO_MUJOCO,
    MUJOCO_SLOT_FOR_ISAACLAB,
    resample_linear,
    resample_root_rot_slerp,
)

# G1 joint names in MuJoCo (hardware) order — taken from policy_parameters.hpp.
G1_JOINT_NAMES_MUJOCO = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def _g1_joint_names_isaaclab() -> list[str]:
    """IsaacLab order = inverse of ISAACLAB_TO_MUJOCO (which maps MJ->IL slot)."""
    names = [""] * 29
    # mujoco_to_isaaclab[i] = j means MJ slot i lives at IL slot j.
    # Equivalently, IL slot MUJOCO_SLOT_FOR_ISAACLAB[j] is MJ slot j.
    # Easiest derivation: for each IL slot j, look up which MJ slot contributes.
    # MUJOCO_SLOT_FOR_ISAACLAB[j_il] = mj_slot.
    for j_il in range(29):
        mj_slot = MUJOCO_SLOT_FOR_ISAACLAB[j_il]
        names[j_il] = G1_JOINT_NAMES_MUJOCO[mj_slot]
    return names


# Default 14-body subset used by SONIC training-time eval (im_eval_callback.py).
DEFAULT_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
    "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
    "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]


def _require_pinocchio():
    import pinocchio as pin
    if not hasattr(pin, "buildModelFromUrdf"):
        raise ImportError(
            "Install robotics Pinocchio: pip install 'pin>=3.9'"
        )
    return pin


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.stack([q[:, 3], q[:, 0], q[:, 1], q[:, 2]], axis=1)


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = q / (np.linalg.norm(q) + 1e-12)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _fill_q_actuated(model, joint_names: list[str], joint_vec: np.ndarray) -> np.ndarray:
    pin = _require_pinocchio()
    q = pin.neutral(model)
    for i, name in enumerate(joint_names):
        jid = model.getJointId(name)
        if jid == 0:
            raise KeyError(f"Joint '{name}' not in URDF")
        jmodel = model.joints[jid]
        if jmodel.nq != 1:
            raise ValueError(f"Joint '{name}' has nq={jmodel.nq}; expected 1-DOF revolute")
        q[jmodel.idx_q] = float(joint_vec[i])
    return q


def _fk_one(model, data, joint_names, joint_vec, body_names, R_root, p_root):
    pin = _require_pinocchio()
    q = _fill_q_actuated(model, joint_names, joint_vec)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    positions = np.zeros((len(body_names), 3), dtype=np.float64)
    quats = np.zeros((len(body_names), 4), dtype=np.float64)  # wxyz
    for bi, bn in enumerate(body_names):
        fid = None
        for cand in (bn, bn.replace("_link", "")):
            try:
                fid = int(model.getFrameId(cand))
                break
            except Exception:
                continue
        if fid is None:
            raise ValueError(f"Body '{bn}' not found in URDF")
        oMf = data.oMf[fid]
        pl = np.asarray(oMf.translation, dtype=np.float64).reshape(3)
        Rl = np.asarray(oMf.rotation, dtype=np.float64).reshape(3, 3)
        p_w = R_root @ pl + p_root
        R_w = R_root @ Rl
        # R -> wxyz quaternion
        m = R_w
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        positions[bi] = p_w
        quats[bi] = np.array([w, x, y, z], dtype=np.float64)
        n = np.linalg.norm(quats[bi]) + 1e-12
        quats[bi] /= n
    return positions, quats


def main() -> int:
    ap = argparse.ArgumentParser(
        description="GMR pkl -> BeyondMimic-style reference NPZ (joint+FK body poses)."
    )
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--urdf", required=True, help="G1 fixed-base URDF (pelvis root)")
    ap.add_argument("--out", required=True, help="Output .npz path")
    ap.add_argument("--target-fps", type=float, default=50.0)
    ap.add_argument("--source-fps", type=float, default=None,
                    help="Override fps from pickle")
    ap.add_argument("--joint-names", default="",
                    help="Comma-separated joint names in IsaacLab order (29). "
                         "Default: derived from G1 hardware order via mujoco_to_isaaclab.")
    ap.add_argument("--body-names", default="",
                    help=f"Comma-separated body names (default: {','.join(DEFAULT_BODY_NAMES)})")
    ap.add_argument("--anchor-body", default="pelvis",
                    help="Anchor body name; must be in --body-names")
    args = ap.parse_args()

    pkl_path = Path(args.pkl).resolve()
    with open(pkl_path, "rb") as f:
        motion = pickle.load(f)
    for k in ("fps", "root_pos", "root_rot", "dof_pos"):
        if k not in motion:
            print(f"Error: pkl missing key '{k}'", file=sys.stderr)
            return 1

    fps_in = float(args.source_fps) if args.source_fps is not None else float(motion["fps"])
    root_pos = np.asarray(motion["root_pos"], dtype=np.float64)         # (T,3)
    root_rot = np.asarray(motion["root_rot"], dtype=np.float64)         # (T,4) xyzw
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float64)           # (T,29) MJ order

    if dof_pos.shape[1] != 29:
        print("Error: dof_pos must be (T,29)", file=sys.stderr); return 1

    # Convert MJ-order dofs to IL-order joint_pos (per visualize_motion.py / SONIC convention).
    T = dof_pos.shape[0]
    joint_isaac = np.zeros((T, 29), dtype=np.float64)
    for j in range(29):
        joint_isaac[:, j] = dof_pos[:, MUJOCO_SLOT_FOR_ISAACLAB[j]]

    # Resample to target fps.
    t_src = np.arange(T, dtype=np.float64) / fps_in
    dt_t = 1.0 / float(args.target_fps)
    t_tgt = np.arange(0.0, float(t_src[-1]) + 1e-9, dt_t, dtype=np.float64)
    if len(t_tgt) < 2:
        print("Error: trajectory too short", file=sys.stderr); return 1

    joint_tgt = resample_linear(t_src, t_tgt, joint_isaac)
    root_p_tgt = resample_linear(t_src, t_tgt, root_pos)
    root_q_xyzw = resample_root_rot_slerp(t_src, t_tgt, root_rot)
    root_q_wxyz = _quat_xyzw_to_wxyz(root_q_xyzw)

    # Joint / body name lists.
    if args.joint_names.strip():
        joint_names = [s.strip() for s in args.joint_names.split(",") if s.strip()]
        if len(joint_names) != 29:
            print(f"Error: --joint-names must have 29 entries (got {len(joint_names)})",
                  file=sys.stderr); return 1
    else:
        joint_names = _g1_joint_names_isaaclab()

    if args.body_names.strip():
        body_names = [s.strip() for s in args.body_names.split(",") if s.strip()]
    else:
        body_names = list(DEFAULT_BODY_NAMES)
    if args.anchor_body not in body_names:
        print(f"Error: --anchor-body '{args.anchor_body}' not in body_names",
              file=sys.stderr); return 1

    # FK per frame.
    pin = _require_pinocchio()
    model = pin.buildModelFromUrdf(args.urdf)
    data = model.createData()

    Tn = joint_tgt.shape[0]
    body_pos_w = np.zeros((Tn, len(body_names), 3), dtype=np.float64)
    body_quat_w = np.zeros((Tn, len(body_names), 4), dtype=np.float64)
    for t in range(Tn):
        R_root = _quat_wxyz_to_R(root_q_wxyz[t])
        p, q = _fk_one(model, data, joint_names, joint_tgt[t],
                       body_names, R_root, root_p_tgt[t])
        body_pos_w[t] = p
        body_quat_w[t] = q

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_path),
        fps=np.array(args.target_fps, dtype=np.float64),
        joint_pos=joint_tgt.astype(np.float64),
        body_pos_w=body_pos_w.astype(np.float64),
        body_quat_w=body_quat_w.astype(np.float64),
        joint_names=np.array(joint_names),
        body_names=np.array(body_names),
    )
    print(f"[build_ref_npz] wrote {out_path}  T={Tn} fps={args.target_fps}  "
          f"joints={len(joint_names)} bodies={len(body_names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
