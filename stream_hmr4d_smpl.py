#!/usr/bin/env python3
"""
Stream HMR4D hmr4d_results.pt as SONIC ZMQ Protocol v3 (SMPL + joints).

Usage (from repo root, .venv_sim active):
    python stream_hmr4d_smpl.py                             # default file
    python stream_hmr4d_smpl.py --pt hmr4d_results.pt      # explicit path
    python stream_hmr4d_smpl.py --fps 30 --loop             # 30 fps, looping
    python stream_hmr4d_smpl.py --inspect                   # only print shapes, don't stream

Steps:
    Terminal 1 (host):   source .venv_sim/bin/activate && python gear_sonic/scripts/run_sim_loop.py
    Terminal 2 (docker): bash deploy.sh --input-type zmq --zmq-host localhost sim
    Terminal 3 (host):   source .venv_sim/bin/activate && python stream_hmr4d_smpl.py
    Then in Terminal 2:  press ] to start, 9 in MuJoCo to drop robot, ENTER to enable ZMQ streaming
"""

import argparse
import sys
import time

import numpy as np
import torch
import zmq


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Stream HMR4D .pt → SONIC ZMQ v3")
    p.add_argument("--pt", default="hmr4d_results.pt", help="Path to hmr4d_results.pt")
    p.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30)")
    p.add_argument("--host", default="*", help="ZMQ bind host (default: * = all interfaces)")
    p.add_argument("--port", type=int, default=5556, help="ZMQ port (default: 5556)")
    p.add_argument("--topic", default="pose", help="ZMQ topic (default: pose)")
    p.add_argument("--loop", action="store_true", help="Loop the sequence indefinitely")
    p.add_argument("--inspect", action="store_true", help="Print shapes and exit (no streaming)")
    p.add_argument("--smpl-source", choices=["global", "incam"], default="global",
                   help="Which smpl_params to use: global (default) or incam")
    p.add_argument("--neutral-heading", action="store_true", default=True,
                   help="Send identity body_quat (no heading snap when streaming starts). Default: ON")
    p.add_argument("--no-neutral-heading", dest="neutral_heading", action="store_false",
                   help="Send processed body_quat from SMPL global_orient")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load and convert
# ---------------------------------------------------------------------------

def load_and_convert(pt_path: str, smpl_source: str, inspect: bool):
    """
    Load hmr4d_results.pt and return smpl_pose [N,21,3] and smpl_joints [N,24,3].

    hmr4d_results.pt structure:
        smpl_params_global / smpl_params_incam:
            body_pose:     (N, 63)   — 21 joints × axis-angle (3)
            global_orient: (N, 3)    — root orientation axis-angle
            transl:        (N, 3)    — root translation
            betas:         (N, 10)
    """
    sys.path.insert(0, ".")
    from gear_sonic.scripts.pico_manager_thread_server import process_smpl_joints

    print(f"[load] Loading {pt_path} ...")
    data = torch.load(pt_path, map_location="cpu", weights_only=False)

    key = f"smpl_params_{smpl_source}"
    if key not in data:
        available = [k for k in data if k.startswith("smpl_params")]
        raise KeyError(f"Key '{key}' not found. Available: {available}")

    params = data[key]
    body_pose_raw = torch.as_tensor(params["body_pose"], dtype=torch.float32)  # (N, 63)
    global_orient  = torch.as_tensor(params["global_orient"], dtype=torch.float32)  # (N, 3)
    transl         = torch.as_tensor(params["transl"], dtype=torch.float32)         # (N, 3)

    N = body_pose_raw.shape[0]
    print(f"[load] Source: {key} | Frames: {N}")
    print(f"[load] body_pose:     {tuple(body_pose_raw.shape)}")
    print(f"[load] global_orient: {tuple(global_orient.shape)}")
    print(f"[load] transl:        {tuple(transl.shape)}")

    # process_smpl_joints expects body_pose with shape (1, 69) — processes one frame at a time.
    # HMR4D gives 63 dims (21 joints). Pad 6 zeros for the 2 missing joints.
    pad = torch.zeros((N, 6), dtype=torch.float32)
    body_pose_69 = torch.cat([body_pose_raw, pad], dim=1)  # (N, 69)

    print(f"[convert] Computing SMPL forward kinematics for {N} frames ...")
    smpl_pose_list, smpl_joints_list, body_quat_list = [], [], []
    for i in range(N):
        out = process_smpl_joints(
            body_pose_69[i:i+1],   # (1, 69)
            global_orient[i:i+1],  # (1, 3)
            transl[i:i+1],         # (1, 3)
        )
        smpl_pose_list.append(out["smpl_pose"].detach().cpu().numpy()[:, :63].reshape(1, 21, 3))
        smpl_joints_list.append(out["smpl_joints_local"].detach().cpu().numpy().reshape(1, 24, 3))
        body_quat_list.append(out["global_orient_quat"].detach().cpu().numpy().reshape(1, 4))
        if (i + 1) % 100 == 0 or i + 1 == N:
            print(f"  {i+1}/{N} frames done")

    smpl_pose   = np.concatenate(smpl_pose_list,   axis=0).astype(np.float32)   # (N, 21, 3)
    smpl_joints = np.concatenate(smpl_joints_list, axis=0).astype(np.float32)   # (N, 24, 3)
    body_quat   = np.concatenate(body_quat_list,   axis=0).astype(np.float32)   # (N, 4)

    print(f"[convert] smpl_pose:   {smpl_pose.shape}   (need [N,21,3]) ✓")
    print(f"[convert] smpl_joints: {smpl_joints.shape} (need [N,24,3]) ✓")

    if inspect:
        print("\n[inspect] Sample smpl_pose[0]:\n", smpl_pose[0])
        print("[inspect] Sample smpl_joints[0]:\n", smpl_joints[0])
        return None, None, None, N

    return smpl_pose, smpl_joints, body_quat, N


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

def stream(args, smpl_pose, smpl_joints, body_quat, N):
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    bind_addr = f"tcp://{args.host}:{args.port}"
    pub.bind(bind_addr)
    print(f"\n[zmq] Publisher bound to {bind_addr} | topic: {args.topic}")
    print("[zmq] Waiting 1s for subscriber to connect ...")
    time.sleep(1.0)

    # joint_pos and joint_vel are all zeros (v3: only wrist joints matter, we have no retargeting here)
    joint_pos = np.zeros((1, 29), dtype=np.float32)
    joint_vel = np.zeros((1, 29), dtype=np.float32)

    # Identity quaternion [w,x,y,z] — avoids heading snap when streaming starts.
    # The policy will track the SMPL body pose but keep the robot facing its current direction.
    IDENTITY_QUAT = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    dt = 1.0 / args.fps
    frame_counter = 0
    seq_idx = 0

    print(f"[stream] Streaming {N} frames at {args.fps} fps | loop={args.loop}")
    print(f"[stream] neutral_heading={args.neutral_heading} (set --no-neutral-heading to use SMPL global_orient as heading)")
    print("[stream] Press Ctrl+C to stop\n")
    print("=" * 55)
    print("  SEQUÊNCIA DE TECLAS NO DEPLOY (Terminal 2):")
    print("  1. Pressione  ]   → inicia o controle")
    print("  2. Na janela MuJoCo pressione  9  → solta o robô")
    print("  3. Pressione ENTER → ZMQ STREAMING MODE: ENABLED")
    print("  4. O robô deve começar a imitar as poses SMPL")
    print("  5. Pressione  O  para parar (emergência)")
    print("=" * 55 + "\n")

    try:
        while True:
            bq = IDENTITY_QUAT if args.neutral_heading else body_quat[seq_idx:seq_idx+1]
            msg = {
                "body_quat":   bq,                                         # [1, 4]  (w,x,y,z)
                "frame_index": np.array([frame_counter], dtype=np.int64),
                "joint_pos":   joint_pos,                                  # [1, 29] zeros
                "joint_vel":   joint_vel,                                  # [1, 29] zeros
                "smpl_joints": smpl_joints[seq_idx:seq_idx+1],             # [1, 24, 3]
                "smpl_pose":   smpl_pose[seq_idx:seq_idx+1],               # [1, 21, 3]
            }
            pub.send(pack_pose_message(msg, topic=args.topic, version=3))

            frame_counter += 1
            seq_idx += 1

            if seq_idx >= N:
                if args.loop:
                    seq_idx = 0
                    print(f"[stream] Loop restart (frame_counter={frame_counter})")
                else:
                    print(f"[stream] Sequence finished ({N} frames sent).")
                    break

            time.sleep(dt)

    except KeyboardInterrupt:
        print("\n[stream] Interrupted by user.")
    finally:
        pub.close()
        ctx.term()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    smpl_pose, smpl_joints, body_quat, N = load_and_convert(
        args.pt, args.smpl_source, args.inspect
    )

    if args.inspect:
        print("\n[inspect] Done. Run without --inspect to start streaming.")
        return

    stream(args, smpl_pose, smpl_joints, body_quat, N)


if __name__ == "__main__":
    main()
