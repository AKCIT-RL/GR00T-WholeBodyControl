#!/usr/bin/env python3
"""GEM (GENMO) webcam demo with ZMQ publishing of per-frame SMPL parameters.

Wraps ``external_dependencies/GENMO/scripts/demo/demo_webcam.py`` and publishes
each ready frame on a ZMQ PUB socket (default port 5558) for consumption by
``gear_sonic/scripts/webcam_smpl_streamer.py``.

IMPORTANT: run this with the GENMO virtual environment (GPU required):
    source external_dependencies/GENMO/.venv/bin/activate
    python gear_sonic/scripts/gem_webcam_zmq_publisher.py --no_imgfeat \
        --render --render_mode opencv

Published payload (pickle via send_pyobj), one message per inference frame:
    body_pose         (63,)  float32  axis-angle, 21 SMPL body joints (in-camera decode)
    global_orient     (3,)   float32  axis-angle, GLOBAL y-up world frame (from rollout)
    transl            (3,)   float32  global translation (y-up world) — informational
    betas             (10,)  float32  (optional)
    frame_index       int
    timestamp_ns      int    time.monotonic_ns() at publish
    timestamp_realtime float time.time()
    dt                float  seconds since previous published frame
    fps               float  smoothed inference fps
"""

# ruff: noqa: E402, I001
import argparse
import queue as _queue_mod
import sys
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GENMO_ROOT = REPO_ROOT / "external_dependencies" / "GENMO"
GENMO_DEMO_DIR = GENMO_ROOT / "scripts" / "demo"

if not GENMO_ROOT.exists():
    sys.exit(
        f"GENMO not found at {GENMO_ROOT}.\n"
        "Clone it first: git clone --depth 1 https://github.com/NVlabs/GENMO.git "
        "external_dependencies/GENMO"
    )

for p in (str(GENMO_ROOT), str(GENMO_DEMO_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import zmq

# demo_webcam sets up cuDNN preloading, torch.load shims, and sys.path on import.
import demo_webcam as _dw
import cv2


class ZmqWebcamGEMSMPLDemo(_dw.WebcamGEMSMPLDemo):
    """WebcamGEMSMPLDemo with a ZMQ PUB socket for per-frame SMPL output."""

    def __init__(self, args):
        super().__init__(args)
        self._zmq_ctx = zmq.Context.instance()
        self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
        self._zmq_pub.setsockopt(zmq.SNDHWM, 3)
        self._zmq_pub.setsockopt(zmq.LINGER, 0)
        self._zmq_pub.bind(f"tcp://*:{args.zmq_port}")
        self._pub_count = 0
        self._last_pub_t = None
        self._headless = bool(getattr(args, "headless", False))
        self._preview_pub = None
        if getattr(args, "preview_port", 0):
            self._preview_pub = self._zmq_ctx.socket(zmq.PUB)
            self._preview_pub.setsockopt(zmq.SNDHWM, 1)
            self._preview_pub.setsockopt(zmq.LINGER, 0)
            self._preview_pub.bind(f"tcp://127.0.0.1:{args.preview_port}")
            print(f"[ZMQ] Preview JPEG on tcp://127.0.0.1:{args.preview_port}")
        print(f"[ZMQ] Publishing SMPL frames on tcp://*:{args.zmq_port}")

    def _emit_preview_raw(self, frame_bgr):
        """Publish the raw camera frame so the UI shows video before tracking starts."""
        if self._preview_pub is None:
            return
        ok, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            try:
                self._preview_pub.send(jpeg.tobytes(), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

    def _emit_display(self, disp) -> bool:
        """Show or publish a display frame. Returns True if the user quit."""
        if self._preview_pub is not None:
            ok, jpeg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                try:
                    self._preview_pub.send(jpeg.tobytes(), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
        if not self._headless:
            cv2.imshow("GEM-SMPL Webcam", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return True
        return False

    def _publish(self, result, avg_fps: float):
        incam = result["body_params_incam"]
        glob = result["body_params_global"]

        now_ns = time.monotonic_ns()
        dt = 0.0
        if self._last_pub_t is not None:
            dt = (now_ns - self._last_pub_t) * 1e-9
        self._last_pub_t = now_ns

        payload = {
            "body_pose": incam["body_pose"].reshape(-1).numpy().astype(np.float32),
            "global_orient": glob["global_orient"].reshape(-1).numpy().astype(np.float32),
            "transl": glob["transl"].reshape(-1).numpy().astype(np.float32),
            "frame_index": int(self.frame_index),
            "timestamp_ns": now_ns,
            "timestamp_realtime": time.time(),
            "dt": dt,
            "fps": float(avg_fps),
        }
        if "betas" in incam:
            payload["betas"] = incam["betas"].reshape(-1).numpy().astype(np.float32)

        try:
            self._zmq_pub.send_pyobj(payload, flags=zmq.NOBLOCK)
            self._pub_count += 1
        except zmq.Again:
            pass

    def run(self):
        """Main loop: identical to the upstream demo, plus ZMQ publishing."""
        _dw.Log.info(
            f"[Run+ZMQ] {self.source_name} | window={self.context_frames} | "
            f"denoiser={self.denoiser_backend} | no_imgfeat={self.no_imgfeat}"
        )

        fps_history = deque(maxlen=60)
        n_frames = 0

        try:
            while True:
                ok, frame_bgr = self.cap.read()
                if not ok:
                    break

                result = self.process_frame(frame_bgr)
                n_frames += 1

                if result is None:
                    if self._display_queue is not None:
                        try:
                            disp = self._display_queue.get_nowait()
                            if self._emit_display(disp):
                                break
                        except _queue_mod.Empty:
                            if not self._headless:
                                cv2.waitKey(1)
                            self._emit_preview_raw(frame_bgr)
                    else:
                        self._emit_preview_raw(frame_bgr)
                    print(f"\rFrame {self.frame_index}: no person detected", end="")
                    continue

                if self._render_queue is not None and result["ready"]:
                    paired = result.get("_frame_bgr")
                    if paired is None:
                        paired = frame_bgr
                    try:
                        self._render_queue.put_nowait({
                            "frame_bgr": paired,
                            "body_params_incam": result["body_params_incam"],
                            "body_params_global": result["body_params_global"],
                            "K_fullimg": self._K_fullimg_cpu,
                        })
                    except _queue_mod.Full:
                        pass

                if self._display_queue is not None:
                    try:
                        disp = self._display_queue.get_nowait()
                        if self._emit_display(disp):
                            break
                    except _queue_mod.Empty:
                        pass

                t = result["timing"]

                if not result["ready"]:
                    self._emit_preview_raw(frame_bgr)
                    print(f"\rWarmup {result['warmup']} | tot={t['total']*1000:.0f}ms", end="")
                    continue

                fps = 1.0 / max(t["total"], 1e-6)
                fps_history.append(fps)
                avg_fps = sum(fps_history) / len(fps_history)

                # --- Publish to bridge ---
                self._publish(result, avg_fps)

                print(
                    f"\rFrame {self.frame_index:5d} | FPS {fps:5.1f} (avg {avg_fps:5.1f}) | "
                    f"tot={t['total']*1000:4.0f}ms | published={self._pub_count}",
                    end="",
                )

        except KeyboardInterrupt:
            print("\n[Interrupted]")
        finally:
            self.cap.release()
            if self._denoiser_executor is not None:
                self._denoiser_executor.shutdown(wait=True, cancel_futures=True)
            if self._preproc_executor is not None:
                self._preproc_executor.shutdown(wait=True, cancel_futures=True)
            if self._render_queue is not None:
                self._render_queue.put(None)
            if self._render_proc is not None:
                self._render_proc.join(timeout=3)
                if self._render_proc.is_alive():
                    self._render_proc.terminate()
            if self._display_queue is not None and not self._headless:
                cv2.destroyAllWindows()
            if self._preview_pub is not None:
                self._preview_pub.close(0)
            self._zmq_pub.close(0)
            print()
            _dw.Log.info(f"[Done] {n_frames} frames | {self._pub_count} published via ZMQ")


def parse_args():
    parser = argparse.ArgumentParser(description="GEM-SMPL Webcam Demo + ZMQ publisher")
    parser.add_argument("--camera_id", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--video", type=str, default=None, help="Video file (overrides camera)")
    parser.add_argument("--zmq_port", type=int, default=5558, help="ZMQ PUB port for SMPL frames")
    parser.add_argument(
        "--context_frames", type=int, default=120,
        help="Sliding window length (must match the exported denoiser seq_len)",
    )
    parser.add_argument("--yolo_period", type=int, default=5)
    parser.add_argument("--vitpose_period", type=int, default=1)
    parser.add_argument(
        "--no_imgfeat", action="store_true",
        help="Skip HMR2 features (faster; uses the no-imgfeat ONNX denoiser)",
    )
    parser.add_argument("--render", action="store_true", help="Enable background rendering")
    parser.add_argument(
        "--render_mode", type=str, default="opencv", choices=["viser", "opencv"],
    )
    parser.add_argument("--render_port", type=int, default=8012)
    parser.add_argument(
        "--preview_port", type=int, default=0,
        help="If set, publish rendered frames as JPEG on this ZMQ PUB port (for the web UI)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="No OpenCV window (use with --preview_port)",
    )
    parser.add_argument("--async_pipeline", action="store_true", default=True)
    parser.add_argument("--no_async_pipeline", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = ZmqWebcamGEMSMPLDemo(args)
    demo.run()
