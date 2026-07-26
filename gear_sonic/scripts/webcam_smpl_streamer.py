#!/usr/bin/env python3
"""Webcam SMPL streamer: bridges GEM (GENMO) webcam pose estimation to SONIC deployment.

Receives per-frame SMPL parameters from ``gem_webcam_zmq_publisher.py`` (ZMQ SUB,
default port 5558), converts them to the SONIC pose-streaming protocol (same wire
format as ``pico_manager_thread_server.py``), and publishes on the ZMQ PUB socket
(default port 5556) consumed by the C++ deployment (``deploy.sh --input-type
zmq_manager``).

Pipeline:
    GEM webcam demo (GENMO venv, GPU)  --ZMQ 5558-->  this bridge (.venv_teleop, CPU)
    --ZMQ 5556-->  C++ SONIC controller (sim or real G1)

Keyboard controls (this terminal):
    s : START streaming in POSE mode (robot imitates your motion)
    p : toggle POSE <-> PLANNER-IDLE (pause imitation, robot stands)
    o : STOP control and exit
    q : same as 'o'

Run (inside .venv_teleop, from repo root):
    python gear_sonic/scripts/webcam_smpl_streamer.py
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import defaultdict, deque
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch
import zmq
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.isaac_utils.rotations import (  # noqa: E402
    remove_smpl_base_rot,
    smpl_root_ytoz_up,
)
from gear_sonic.trl.utils.rotation_conversion import decompose_rotation_aa  # noqa: E402
from gear_sonic.trl.utils.torch_transform import (  # noqa: E402
    angle_axis_to_quaternion,
    compute_human_joints,
    quat_apply,
    quat_inv,
    quaternion_to_angle_axis,
    quaternion_to_rotation_matrix,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    build_planner_message,
    pack_pose_message,
)

HUMAN_JOINTS_INFO_PATH = str(REPO_ROOT / "gear_sonic" / "data" / "human" / "human_joints_info.pkl")


class StreamMode(IntEnum):
    OFF = 0
    POSE = 1
    PLANNER_IDLE = 2


# ---------------------------------------------------------------------------
#  SMPL processing (mirrors pico_manager_thread_server.process_smpl_joints)
# ---------------------------------------------------------------------------


@torch.no_grad()
def process_smpl_joints(body_pose: torch.Tensor, global_orient: torch.Tensor) -> dict:
    """Process SMPL parameters to compute local joints.

    Args:
        body_pose: Body pose tensor (axis-angle), shape (T, 63) or (T, 69)
        global_orient: Global orientation tensor (axis-angle, y-up SMPL world), shape (T, 3)

    Returns:
        Dict with smpl_pose, smpl_joints_local, global_orient_quat (z-up, base rot removed).
    """
    global_orient_quat = angle_axis_to_quaternion(global_orient)
    global_orient_quat = smpl_root_ytoz_up(global_orient_quat)
    global_orient_new = quaternion_to_angle_axis(global_orient_quat)

    joints = compute_human_joints(
        body_pose=body_pose[..., :63],
        global_orient=global_orient_new,
        human_joints_info_path=HUMAN_JOINTS_INFO_PATH,
    )  # (T, 24, 3)

    global_orient_quat = remove_smpl_base_rot(global_orient_quat, w_last=False)

    global_orient_quat_inv = quat_inv(global_orient_quat).unsqueeze(1).repeat(1, joints.shape[1], 1)
    smpl_joints_local = quat_apply(global_orient_quat_inv, joints)
    global_orient_mat = quaternion_to_rotation_matrix(global_orient_quat)
    global_orient_6d = global_orient_mat[..., :2].reshape(1, 6)

    return {
        "smpl_pose": body_pose,
        "smpl_joints_local": smpl_joints_local,
        "global_orient_quat": global_orient_quat,
        "global_orient_6d": global_orient_6d,
    }


def _quat_lerp_normalized(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Lerp two quaternions (shape (4,)) with shortest-path sign flip and renormalize."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
    q = (1.0 - alpha) * q0 + alpha * q1
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    return q


def _interp_pose_axis_angle(
    prev_pose: np.ndarray, curr_pose: np.ndarray, alpha: float
) -> np.ndarray:
    """Interpolate (21,3) axis-angle poses via per-joint quaternion lerp."""
    prev_quats = R.from_rotvec(prev_pose.reshape(-1, 3)).as_quat()
    curr_quats = R.from_rotvec(curr_pose.reshape(-1, 3)).as_quat()
    out_quats = np.empty_like(prev_quats)
    for i in range(prev_quats.shape[0]):
        out_quats[i] = _quat_lerp_normalized(prev_quats[i], curr_quats[i], alpha)
    return R.from_quat(out_quats).as_rotvec().reshape(prev_pose.shape)


def compute_g1_wrist_joint_pos(body_pose_21x3: np.ndarray) -> np.ndarray:
    """Map SMPL elbow/wrist rotations onto the G1 29-dof wrist joints.

    Mirrors the "From @Jiefeng" block in pico_manager_thread_server.SmplStream.run_once.

    Args:
        body_pose_21x3: (21, 3) axis-angle body pose (SMPL, no root)

    Returns:
        joint_pos: (29,) with only wrist entries populated
    """
    joint_pos = np.zeros(29)
    body_pose = body_pose_21x3.reshape(-1, 21, 3)

    SMPL_L_ELBOW_IDX = 17
    SMPL_L_WRIST_IDX = 19
    SMPL_R_ELBOW_IDX = 18
    SMPL_R_WRIST_IDX = 20

    G1_L_WRIST_ROLL_IDX = 23
    G1_L_WRIST_PITCH_IDX = 25
    G1_L_WRIST_YAW_IDX = 27
    G1_R_WRIST_ROLL_IDX = 24
    G1_R_WRIST_PITCH_IDX = 26
    G1_R_WRIST_YAW_IDX = 28

    smpl_l_elbow_aa = body_pose[:, SMPL_L_ELBOW_IDX]
    smpl_l_wrist_aa = body_pose[:, SMPL_L_WRIST_IDX]
    smpl_r_elbow_aa = body_pose[:, SMPL_R_ELBOW_IDX]
    smpl_r_wrist_aa = body_pose[:, SMPL_R_WRIST_IDX]

    # Guard against exactly-zero rotations (decompose_rotation_aa divides by the angle)
    def _safe_aa(aa: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        aa = aa.copy()
        norms = np.linalg.norm(aa, axis=-1)
        aa[norms < eps, 0] = eps
        return aa

    smpl_l_elbow_aa = _safe_aa(smpl_l_elbow_aa)
    smpl_r_elbow_aa = _safe_aa(smpl_r_elbow_aa)

    elbow_axis = np.array([0, 1, 0])
    _, g1_l_elbow_q_swing = decompose_rotation_aa(smpl_l_elbow_aa, elbow_axis)
    _, g1_r_elbow_q_swing = decompose_rotation_aa(smpl_r_elbow_aa, elbow_axis)

    l_elbow_swing_euler = R.from_quat(g1_l_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_elbow_swing_euler = R.from_quat(g1_r_elbow_q_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )

    l_wrist_euler = R.from_rotvec(smpl_l_wrist_aa).as_euler("XYZ", degrees=False)
    r_wrist_euler = R.from_rotvec(smpl_r_wrist_aa).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_elbow_swing_euler[:, 0] + l_wrist_euler[:, 0]
    g1_l_wrist_pitch = -l_wrist_euler[:, 1]
    g1_l_wrist_yaw = l_elbow_swing_euler[:, 2] + l_wrist_euler[:, 2]

    g1_r_wrist_roll = -(r_elbow_swing_euler[:, 0] + r_wrist_euler[:, 0])
    g1_r_wrist_pitch = -r_wrist_euler[:, 1]
    g1_r_wrist_yaw = r_elbow_swing_euler[:, 2] + r_wrist_euler[:, 2]

    joint_pos[G1_L_WRIST_ROLL_IDX] = g1_l_wrist_roll[0]
    joint_pos[G1_L_WRIST_PITCH_IDX] = -g1_l_wrist_pitch[0]
    joint_pos[G1_L_WRIST_YAW_IDX] = g1_l_wrist_yaw[0]

    joint_pos[G1_R_WRIST_ROLL_IDX] = g1_r_wrist_roll[0]
    joint_pos[G1_R_WRIST_PITCH_IDX] = g1_r_wrist_pitch[0]
    joint_pos[G1_R_WRIST_YAW_IDX] = g1_r_wrist_yaw[0]

    return joint_pos


# ---------------------------------------------------------------------------
#  GEM receiver (background thread)
# ---------------------------------------------------------------------------


class GemReceiver:
    """Background SUB that receives GEM frames and pre-computes SMPL joint outputs.

    Keeps the two most recent processed frames so the main loop can interpolate
    (GEM runs at ~15-30 fps; the controller consumes 50 Hz).
    """

    def __init__(self, host: str, port: int):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._prev: dict | None = None
        self._curr: dict | None = None
        self._n_received = 0
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._sock.close(0)

    @property
    def n_received(self) -> int:
        return self._n_received

    def _run(self):
        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            events = dict(poller.poll(timeout=100))
            if self._sock not in events:
                continue
            try:
                sample = self._sock.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            processed = self._process(sample)
            if processed is None:
                continue
            with self._lock:
                self._prev = self._curr
                self._curr = processed
                self._n_received += 1

    def _process(self, sample: dict) -> dict | None:
        try:
            body_pose = torch.from_numpy(
                np.asarray(sample["body_pose"], dtype=np.float32)
            ).reshape(1, -1)
            global_orient = torch.from_numpy(
                np.asarray(sample["global_orient"], dtype=np.float32)
            ).reshape(1, 3)
        except (KeyError, ValueError) as e:
            print(f"[Bridge] Bad GEM sample: {e}")
            return None

        out = process_smpl_joints(body_pose, global_orient)
        return {
            "timestamp_ns": int(sample.get("timestamp_ns", time.monotonic_ns())),
            "timestamp_realtime": float(sample.get("timestamp_realtime", time.time())),
            "dt": float(sample.get("dt", 0.0)),
            "fps": float(sample.get("fps", 0.0)),
            "smpl_pose_np": out["smpl_pose"].numpy()[:, :63].reshape(-1, 21, 3)[0].astype(
                np.float32
            ),
            "smpl_joints_np": out["smpl_joints_local"].numpy()[0].astype(np.float32),
            "body_quat_np": out["global_orient_quat"].numpy()[0].astype(np.float32),
        }

    def get_pair(self) -> tuple[dict | None, dict | None]:
        with self._lock:
            return self._prev, self._curr


# ---------------------------------------------------------------------------
#  Keyboard (raw, non-blocking, this terminal)
# ---------------------------------------------------------------------------


class KeyboardListener:
    """Non-blocking single-key reader using termios cbreak mode."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)

    def restore(self):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)

    def get_key(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="GEM webcam -> SONIC pose bridge")
    parser.add_argument("--gem_host", type=str, default="localhost", help="GEM publisher host")
    parser.add_argument("--gem_port", type=int, default=5558, help="GEM publisher port")
    parser.add_argument("--port", type=int, default=5556, help="SONIC ZMQ PUB port")
    parser.add_argument("--target_fps", type=int, default=50, help="Output rate (Hz)")
    parser.add_argument(
        "--num_frames_to_send", type=int, default=5, help="Frames per pose message"
    )
    parser.add_argument(
        "--record_dir", type=str, default="", help="Save sent batches as npz (default: off)"
    )
    parser.add_argument(
        "--auto_start",
        action="store_true",
        help="Send START (pose mode) automatically once GEM frames arrive",
    )
    args = parser.parse_args()

    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{args.port}")
    time.sleep(0.2)
    print(f"[Bridge] SONIC PUB bound to tcp://*:{args.port}")

    receiver = GemReceiver(args.gem_host, args.gem_port)
    receiver.start()
    print(f"[Bridge] Listening for GEM frames on tcp://{args.gem_host}:{args.gem_port}")
    print(
        "[Bridge] Keys: [s]=START pose mode  [p]=toggle POSE/PLANNER-IDLE  "
        "[o]/[q]=STOP+exit"
    )

    mode = StreamMode.OFF
    frame_buffer: dict[str, deque] = defaultdict(lambda: deque(maxlen=args.num_frames_to_send))
    buffer_cleared = True
    step = 0
    record_idx = 0
    frame_time = 1.0 / args.target_fps
    left_hand_joints = np.zeros((1, 7), dtype=np.float32)
    right_hand_joints = np.zeros((1, 7), dtype=np.float32)
    fps_counter = 0
    last_fps_report = time.time()
    warned_no_gem = False

    kb = KeyboardListener()

    def clear_buffer():
        nonlocal buffer_cleared
        for k in list(frame_buffer.keys()):
            frame_buffer[k].clear()
        buffer_cleared = True

    def send_command(new_mode: StreamMode):
        if new_mode == StreamMode.POSE:
            pub.send(build_command_message(start=True, stop=False, planner=False))
        elif new_mode == StreamMode.PLANNER_IDLE:
            pub.send(build_command_message(start=True, stop=False, planner=True))
        elif new_mode == StreamMode.OFF:
            pub.send(build_command_message(start=False, stop=True, planner=True))

    try:
        frame_start = time.time()
        while True:
            # --- Keyboard ---
            key = kb.get_key()
            if key in ("o", "q", "\x03"):
                print("\n[Bridge] STOP requested")
                send_command(StreamMode.OFF)
                break
            elif key == "s" and mode == StreamMode.OFF:
                if receiver.n_received == 0:
                    print("[Bridge] No GEM frames yet — wait for the webcam warmup.")
                else:
                    mode = StreamMode.POSE
                    clear_buffer()
                    print("[Bridge] -> POSE mode (robot imitates you). Command will be sent.")
            elif key == "p" and mode != StreamMode.OFF:
                if mode == StreamMode.POSE:
                    mode = StreamMode.PLANNER_IDLE
                    send_command(mode)
                    print("[Bridge] -> PLANNER-IDLE (imitation paused, robot stands)")
                else:
                    mode = StreamMode.POSE
                    clear_buffer()
                    print("[Bridge] -> POSE mode (imitation resumed)")

            if args.auto_start and mode == StreamMode.OFF and receiver.n_received > 0:
                mode = StreamMode.POSE
                clear_buffer()
                print("[Bridge] auto_start: -> POSE mode")

            if mode == StreamMode.PLANNER_IDLE:
                pub.send(
                    build_planner_message(0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], -1.0, -1.0)
                )

            elif mode == StreamMode.POSE:
                prev, curr = receiver.get_pair()
                if curr is None:
                    if not warned_no_gem:
                        print("[Bridge] Waiting for GEM frames...")
                        warned_no_gem = True
                elif prev is None or curr["timestamp_ns"] <= prev["timestamp_ns"]:
                    pass  # need two distinct frames to interpolate
                else:
                    # Playback delayed by one source interval for smooth interpolation
                    src_interval = curr["timestamp_ns"] - prev["timestamp_ns"]
                    playback_ns = time.monotonic_ns() - src_interval
                    alpha = (playback_ns - prev["timestamp_ns"]) / float(src_interval)
                    alpha = min(max(alpha, 0.0), 1.0)

                    use_pose = _interp_pose_axis_angle(
                        prev["smpl_pose_np"], curr["smpl_pose_np"], alpha
                    ).astype(np.float32)
                    use_joints = (
                        (1.0 - alpha) * prev["smpl_joints_np"] + alpha * curr["smpl_joints_np"]
                    ).astype(np.float32)
                    use_body_quat = _quat_lerp_normalized(
                        prev["body_quat_np"], curr["body_quat_np"], alpha
                    ).astype(np.float32)

                    joint_pos = compute_g1_wrist_joint_pos(use_pose)

                    frame_buffer["smpl_pose"].append(use_pose)
                    frame_buffer["smpl_joints"].append(use_joints)
                    frame_buffer["body_quat_w"].append(use_body_quat)
                    frame_buffer["frame_index"].append(int(step))
                    frame_buffer["joint_pos"].append(joint_pos)

                    N = len(frame_buffer["frame_index"])
                    buffer_is_full = N >= args.num_frames_to_send
                    if buffer_is_full and buffer_cleared:
                        buffer_cleared = False
                        send_command(StreamMode.POSE)
                        print("[Bridge] Buffer full — streaming pose + START command sent")

                    if buffer_is_full and not buffer_cleared:
                        numpy_data = {
                            "smpl_pose": np.stack(frame_buffer["smpl_pose"], axis=0),
                            "smpl_joints": np.stack(frame_buffer["smpl_joints"], axis=0),
                            "body_quat_w": np.stack(frame_buffer["body_quat_w"], axis=0),
                            "joint_pos": np.stack(frame_buffer["joint_pos"], axis=0),
                            "joint_vel": np.zeros((N, 29)),
                            "frame_index": np.array(
                                frame_buffer["frame_index"], dtype=np.int64
                            ),
                            "left_trigger": np.array([0.0], dtype=np.float32),
                            "right_trigger": np.array([0.0], dtype=np.float32),
                            "left_grip": np.array([0.0], dtype=np.float32),
                            "right_grip": np.array([0.0], dtype=np.float32),
                            "pico_dt": np.array([curr["dt"]], dtype=np.float32),
                            "pico_fps": np.array([curr["fps"]], dtype=np.float32),
                            "timestamp_realtime": np.array(
                                [curr["timestamp_realtime"]], dtype=np.float64
                            ),
                            "timestamp_monotonic": np.array(
                                [curr["timestamp_ns"] * 1e-9], dtype=np.float64
                            ),
                            "left_hand_joints": left_hand_joints.reshape(-1),
                            "right_hand_joints": right_hand_joints.reshape(-1),
                            "toggle_data_collection": np.array([False], dtype=bool),
                            "toggle_data_abort": np.array([False], dtype=bool),
                            "heading_increment": np.array([0.0], dtype=np.float32),
                        }
                        pub.send(pack_pose_message(numpy_data, topic="pose"))

                        if args.record_dir:
                            out_path = os.path.join(
                                args.record_dir, f"pose_{record_idx:06d}.npz"
                            )
                            np.savez_compressed(out_path, **numpy_data)
                            record_idx += 1

                    step += 1
                    fps_counter += 1

            now = time.time()
            if now - last_fps_report >= 5.0:
                fps = fps_counter / (now - last_fps_report)
                print(
                    f"[Bridge] mode={mode.name} out_fps={fps:.1f} "
                    f"gem_frames={receiver.n_received} step={step}"
                )
                fps_counter = 0
                last_fps_report = now

            elapsed = time.time() - frame_start
            if elapsed < frame_time:
                time.sleep(frame_time - elapsed)
            frame_start = time.time()

    except KeyboardInterrupt:
        print("\n[Bridge] Interrupted — sending STOP")
        send_command(StreamMode.OFF)
    finally:
        kb.restore()
        receiver.stop()
        time.sleep(0.1)
        pub.close(0)
        print("[Bridge] Shutdown complete")


if __name__ == "__main__":
    main()
