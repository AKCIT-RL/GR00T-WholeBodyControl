#!/usr/bin/env bash
# Batch SONIC deploy + metrics from a folder of GMR motion .pkl files.
#
# For every <clip>.pkl in PKL_DIR:
#   1. Convert to SONIC CSV reference   (gmr_pkl_to_sonic_motion.py)
#   2. Build BeyondMimic-style NPZ ref  (build_ref_npz_from_pkl.py + Pinocchio FK)
#   3. cd into per-clip log dir, launch deploy.sh sim with stdin pipe autostart
#      (`]` start_control, `T` play_motion, `O` stop). End-of-clip detected via
#      stdout "Motion ... completed" or duration timeout.
#   4. Compute metrics with eval_sonic_metrics_from_logs.py
#
# Usage:
#   ./run_sonic_batch.sh [OPTIONS] <pkl_dir> <out_dir>
#
# Options:
#   --urdf PATH            URDF (default: decoupled_wbc/.../g1_29dof.urdf)
#   --checkpoint PATH      passed to deploy.sh --cp
#   --obs-config PATH      passed to deploy.sh --obs-config
#   --planner PATH         passed to deploy.sh --planner
#   --target-fps N         (default 50)
#   --warmup-sec N         seconds to wait before pressing start (default 18)
#   --extra-sec N          extra seconds after motion completes (default 5)
#   --max-sec N            absolute timeout per clip (default 600)
#   --anchor-body NAME     default 'pelvis'
#   --body-names CSV       override 14-body default
#   --skip-existing        skip clips with metrics file already present
#
# Output layout under <out_dir>:
#   sonic_ref/<clip>/{joint_pos.csv,joint_vel.csv,body_pos.csv,body_quat.csv,...}
#   ref_npz/<clip>.npz
#   sonic_logs/<clip>/{deploy.log, logs/<date>/<time>/{q.csv,...}}
#   metrics/<clip>.json
#   metrics/summary.csv

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$DEPLOY_DIR/.." && pwd)"

URDF_DEFAULT="$REPO_ROOT/decoupled_wbc/control/robot_model/model_data/g1/g1_29dof.urdf"
URDF="$URDF_DEFAULT"
CHECKPOINT_OPT=()
OBS_OPT=()
PLANNER_OPT=()
TARGET_FPS=50
WARMUP_SEC=18
EXTRA_SEC=5
MAX_SEC=600
ANCHOR_BODY="pelvis"
BODY_NAMES_OPT=()
SKIP_EXISTING=0

PYTHON_BIN="${PYTHON_BIN:-python3}"

positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --urdf)         URDF="$(realpath "$2")"; shift 2 ;;
    --checkpoint)   CHECKPOINT_OPT=(--checkpoint "$2"); shift 2 ;;
    --obs-config)   OBS_OPT=(--obs-config "$2"); shift 2 ;;
    --planner)      PLANNER_OPT=(--planner "$2"); shift 2 ;;
    --target-fps)   TARGET_FPS="$2"; shift 2 ;;
    --warmup-sec)   WARMUP_SEC="$2"; shift 2 ;;
    --extra-sec)    EXTRA_SEC="$2"; shift 2 ;;
    --max-sec)      MAX_SEC="$2"; shift 2 ;;
    --anchor-body)  ANCHOR_BODY="$2"; shift 2 ;;
    --body-names)   BODY_NAMES_OPT=(--body-names "$2"); shift 2 ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    -h|--help)
      sed -n '1,40p' "$0"; exit 0 ;;
    -*)
      echo "error: unknown option $1" >&2; exit 1 ;;
    *)
      positional+=("$1"); shift ;;
  esac
done

if [[ ${#positional[@]} -lt 2 ]]; then
  echo "usage: $0 [opts] <pkl_dir> <out_dir>" >&2; exit 1
fi
PKL_DIR="$(realpath "${positional[0]}")"
OUT_DIR="$(mkdir -p "${positional[1]}" && realpath "${positional[1]}")"

REF_NPZ_DIR="$OUT_DIR/ref_npz"
SONIC_REF_DIR="$OUT_DIR/sonic_ref"
LOG_DIR_ROOT="$OUT_DIR/sonic_logs"
METRICS_DIR="$OUT_DIR/metrics"
mkdir -p "$REF_NPZ_DIR" "$SONIC_REF_DIR" "$LOG_DIR_ROOT" "$METRICS_DIR"

SUMMARY="$METRICS_DIR/summary.csv"
if [[ ! -f "$SUMMARY" ]]; then
  echo "clip,joint_rmse_rad,E_mpjpe_m,success_rate,frames_used,E_vel_mm_s_ref,E_vel_mm_s_robot,E_acc_mm_s2_ref,E_acc_mm_s2_robot" > "$SUMMARY"
fi

shopt -s nullglob
pkl_files=("$PKL_DIR"/*.pkl)
if [[ ${#pkl_files[@]} -eq 0 ]]; then
  echo "error: no .pkl in $PKL_DIR" >&2; exit 1
fi

run_one_deploy() {
  local clip="$1"
  local sonic_clip_dir="$2"   # absolute dir holding clip subdir for --motion-data parent
  local clip_log_dir="$3"
  local duration="$4"

  local fifo
  fifo="$(mktemp -u "$clip_log_dir/.input_fifo.XXXXXX")"
  mkfifo "$fifo"

  pushd "$clip_log_dir" >/dev/null

  # Run deploy.sh with stdin from the FIFO; logs (StateLogger) land under ./logs/<date>/<time>/
  ( bash "$DEPLOY_DIR/deploy.sh" \
      "${CHECKPOINT_OPT[@]:-}" \
      "${OBS_OPT[@]:-}" \
      "${PLANNER_OPT[@]:-}" \
      --motion-data "$sonic_clip_dir/" \
      sim ) <"$fifo" >"$clip_log_dir/deploy.log" 2>&1 &
  local deploy_pid=$!
  exec 3>"$fifo"

  # Best-effort cleanup
  cleanup() {
    set +e
    exec 3>&- || true
    if kill -0 "$deploy_pid" 2>/dev/null; then
      kill "$deploy_pid" 2>/dev/null
      sleep 0.5
      kill -9 "$deploy_pid" 2>/dev/null
    fi
    rm -f "$fifo"
  }
  trap cleanup EXIT INT TERM

  # Warmup, then start_control + play_motion.
  sleep "$WARMUP_SEC"
  if ! kill -0 "$deploy_pid" 2>/dev/null; then
    echo "[run_sonic_batch] deploy died during warmup ($clip); see deploy.log" >&2
    cleanup
    trap - EXIT INT TERM
    popd >/dev/null
    return 1
  fi
  printf "]" >&3
  sleep 0.4
  printf "T" >&3
  echo "[run_sonic_batch] $clip: pressed ] T (start, play)"

  # Wait for "Motion ... completed" OR timeout (duration + EXTRA + MAX cap).
  local deadline
  deadline=$(( $(date +%s) + ${MAX_SEC} ))
  local timed_deadline
  timed_deadline=$(( $(date +%s) + ${duration%.*} + EXTRA_SEC + 2 ))
  local end=$deadline
  if (( timed_deadline < deadline )); then end=$timed_deadline; fi

  local done=0
  while [[ "$(date +%s)" -lt "$end" ]]; do
    if grep -q -m1 "Motion .* completed" "$clip_log_dir/deploy.log" 2>/dev/null; then
      done=1
      break
    fi
    if ! kill -0 "$deploy_pid" 2>/dev/null; then
      break
    fi
    sleep 0.5
  done

  # Extra time so logger flushes the last frames.
  sleep "$EXTRA_SEC"
  printf "O" >&3 || true
  sleep 0.5

  cleanup
  trap - EXIT INT TERM
  popd >/dev/null
  if [[ "$done" -eq 1 ]]; then
    return 0
  fi
  echo "[run_sonic_batch] $clip: motion-completed not detected; logs may still be valid" >&2
  return 0
}

resolve_state_logger_dir() {
  local clip_log_dir="$1"
  # StateLogger writes to logs/<dd-MM-yy>/<HH-MM-SS>/<csvs>
  local newest
  newest="$(find "$clip_log_dir/logs" -mindepth 2 -maxdepth 2 -type d 2>/dev/null \
            | sort | tail -1 || true)"
  if [[ -n "$newest" && -f "$newest/q.csv" ]]; then
    echo "$newest"
    return 0
  fi
  return 1
}

for pkl in "${pkl_files[@]}"; do
  clip="$(basename "$pkl" .pkl)"
  metrics_file="$METRICS_DIR/$clip.json"
  if [[ "$SKIP_EXISTING" -eq 1 && -f "$metrics_file" ]]; then
    echo "[run_sonic_batch] skip $clip (metrics exist)"; continue
  fi

  echo "==================== $clip ===================="

  sonic_clip_root="$SONIC_REF_DIR/$clip"   # gmr_pkl_to_sonic_motion creates $sonic_clip_root/<clip>/
  ref_npz="$REF_NPZ_DIR/$clip.npz"
  clip_log_dir="$LOG_DIR_ROOT/$clip"
  rm -rf "$clip_log_dir"
  mkdir -p "$clip_log_dir" "$sonic_clip_root"

  # 1) Build SONIC CSV reference. Output is $sonic_clip_root/<clip>/...
  "$PYTHON_BIN" "$SCRIPT_DIR/gmr_pkl_to_sonic_motion.py" \
    --pkl "$pkl" \
    --out "$sonic_clip_root" \
    --target-fps "$TARGET_FPS"

  # 2) Build NPZ reference (joints in IL order + FK body poses).
  "$PYTHON_BIN" "$SCRIPT_DIR/build_ref_npz_from_pkl.py" \
    --pkl "$pkl" \
    --urdf "$URDF" \
    --out "$ref_npz" \
    --target-fps "$TARGET_FPS" \
    --anchor-body "$ANCHOR_BODY" \
    "${BODY_NAMES_OPT[@]:-}"

  # 3) Compute clip duration (sec) from NPZ.
  duration="$("$PYTHON_BIN" -c "import numpy as np; d=np.load('$ref_npz'); print(len(d['joint_pos'])/float(np.asarray(d['fps']).ravel()[0]))")"
  echo "[run_sonic_batch] $clip duration=${duration}s"

  # 4) Run deploy with stdin autostart (sonic_clip_root contains <clip>/ inside).
  if ! run_one_deploy "$clip" "$sonic_clip_root" "$clip_log_dir" "$duration"; then
    echo "[run_sonic_batch] $clip: deploy failed; skipping metrics" >&2
    continue
  fi

  # 5) Resolve StateLogger output dir.
  if ! state_dir="$(resolve_state_logger_dir "$clip_log_dir")"; then
    echo "[run_sonic_batch] $clip: no StateLogger dir under $clip_log_dir/logs; skipping metrics" >&2
    continue
  fi

  # 6) Compute metrics.
  "$PYTHON_BIN" "$SCRIPT_DIR/eval_sonic_metrics_from_logs.py" \
    --ref-npz "$ref_npz" \
    --state-logger-dir "$state_dir" \
    --urdf "$URDF" \
    --anchor-body "$ANCHOR_BODY" \
    --skip-success-height \
    "${BODY_NAMES_OPT[@]:-}" \
    --output "$metrics_file"

  # 7) Append to summary.
  "$PYTHON_BIN" - "$metrics_file" "$clip" >> "$SUMMARY" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1]))
clip = sys.argv[2]
def g(k):
    v = m.get(k, "")
    if isinstance(v, float): return f"{v:.6f}"
    return str(v)
print(",".join([clip, g("joint_rmse_rad"), g("E_mpjpe_m"), g("success_rate"),
                g("frames_used"), g("E_vel_mm_s_ref"), g("E_vel_mm_s_robot"),
                g("E_acc_mm_s2_ref"), g("E_acc_mm_s2_robot")]))
PYEOF
done

echo "==================== batch done ===================="
echo "metrics: $METRICS_DIR"
echo "summary: $SUMMARY"
