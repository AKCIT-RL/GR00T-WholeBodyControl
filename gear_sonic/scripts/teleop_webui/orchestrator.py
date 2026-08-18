#!/usr/bin/env python3
"""Process orchestrator for the webcam teleop stack (sim | C++ | GEM | bridge).

Spawns each component in its own PTY (pexpect) so that:
  - readiness markers can be detected in real time (no pipe buffering);
  - keyboard commands can be sent programmatically (']' / Enter / 'o' to the
    C++ controller, 's'/'p'/'o' to the bridge) — exactly what a human does
    in the manual runbook (gear_sonic/scripts/README_webcam_teleop.md).
"""

from __future__ import annotations

import glob
import os
import re
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import pexpect

REPO_ROOT = Path(__file__).resolve().parents[3]

PREVIEW_PORT = 5559
CHECK_PORTS = (5556, 5557, 5558, PREVIEW_PORT)

KEY_MAP = {"enter": "\r", "]": "]", "o": "o", "i": "i", "f": "f", "s": "s", "p": "p"}
ALLOWED_KEYS = {
    "cpp": {"]", "enter", "o", "i", "f"},
    "bridge": {"s", "p", "n", "o"},
}


@dataclass
class ComponentSpec:
    name: str
    argv: list[str]
    cwd: str
    ready_pattern: re.Pattern | None = None
    ready_timeout: float = 120.0
    alive_ready_secs: float = 0.0  # if no pattern: ready after N s alive
    parsers: list[tuple[re.Pattern, str]] = field(default_factory=list)  # (regex, metric prefix)


class Component:
    def __init__(self, spec: ComponentSpec):
        self.spec = spec
        self.proc: pexpect.spawn | None = None
        self.state = "idle"  # idle|starting|ready|exited|error
        self.lines: deque[str] = deque(maxlen=120)
        self.metrics: dict = {}
        self.ready_event = threading.Event()
        self._reader: threading.Thread | None = None
        self.started_at = 0.0

    def start(self, env: dict):
        self.lines.clear()
        self.metrics.clear()
        self.ready_event.clear()
        self.state = "starting"
        self.started_at = time.monotonic()
        self.proc = pexpect.spawn(
            self.spec.argv[0],
            args=self.spec.argv[1:],
            cwd=self.spec.cwd,
            env=env,
            echo=False,
            timeout=None,
            maxread=8192,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        buf = b""
        proc = self.proc
        while proc is not None and proc.isalive():
            try:
                chunk = proc.read_nonblocking(8192, timeout=0.3)
            except pexpect.TIMEOUT:
                continue
            except (pexpect.EOF, OSError):
                break
            buf += chunk
            parts = re.split(rb"[\r\n]", buf)
            buf = parts.pop()
            for raw in parts:
                line = raw.decode("utf-8", "replace").strip()
                if line:
                    self._on_line(line)
        if self.state in ("starting", "ready"):
            self.state = "exited"

    def _on_line(self, line: str):
        self.lines.append(line)
        if self.spec.ready_pattern and self.spec.ready_pattern.search(line):
            self.ready_event.set()
        for pattern, prefix in self.spec.parsers:
            m = pattern.search(line)
            if m:
                for key, val in m.groupdict().items():
                    if val is not None:
                        self.metrics[f"{prefix}{key}"] = val
                self.metrics[f"{prefix}ts"] = time.time()

    def wait_ready(self) -> bool:
        spec = self.spec
        deadline = time.monotonic() + spec.ready_timeout
        while time.monotonic() < deadline:
            if self.proc is None or not self.proc.isalive():
                self.state = "error"
                return False
            if spec.ready_pattern is not None:
                if self.ready_event.wait(timeout=0.5):
                    self.state = "ready"
                    return True
            elif time.monotonic() - self.started_at >= spec.alive_ready_secs:
                self.state = "ready"
                return True
        self.state = "error"
        return False

    def send(self, text: str):
        if self.proc is not None and self.proc.isalive():
            self.proc.send(text)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.isalive()

    def stop(self, sig=signal.SIGINT, grace: float = 5.0):
        proc = self.proc
        if proc is None:
            return
        if proc.isalive():
            try:
                os.killpg(proc.pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
            deadline = time.monotonic() + grace
            while proc.isalive() and time.monotonic() < deadline:
                time.sleep(0.2)
            if proc.isalive():
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        try:
            proc.close(force=True)
        except Exception:
            pass
        self.proc = None
        if self.state != "error":
            self.state = "exited"

    def status(self) -> dict:
        return {
            "state": self.state,
            "pid": self.proc.pid if self.alive() else None,
            "lines": list(self.lines)[-25:],
            "metrics": dict(self.metrics),
        }


def _make_specs(mode: str, camera_id: int, upper_body: bool = False) -> dict[str, ComponentSpec]:
    root = str(REPO_ROOT)
    specs = {}
    if mode == "sim":
        specs["sim"] = ComponentSpec(
            name="sim",
            argv=[f"{root}/.venv_sim/bin/python", "gear_sonic/scripts/run_sim_loop.py"],
            cwd=root,
            alive_ready_secs=6.0,
            ready_timeout=30.0,
        )
    if mode == "sim":
        cpp_argv = [
            "just", "run", "g1_deploy_onnx_ref", "lo",
            "policy/release/model_decoder.onnx", "reference/example/",
            "--obs-config", "policy/release/observation_config.yaml",
            "--encoder-file", "policy/release/model_encoder.onnx",
            "--planner-file", "planner/target_vel/V2/planner_sonic.onnx",
            "--input-type", "zmq_manager", "--output-type", "all",
            "--zmq-host", "localhost", "--disable-crc-check",
        ]
    else:  # deploy.sh auto-detects the robot network interface (and rebuilds)
        cpp_argv = [f"{root}/gear_sonic_deploy/deploy.sh", "--input-type", "zmq_manager", mode]
    specs["cpp"] = ComponentSpec(
        name="cpp",
        argv=cpp_argv,
        cwd=f"{root}/gear_sonic_deploy",
        ready_pattern=re.compile(r"Init Done"),
        ready_timeout=600.0,
    )
    specs["gem"] = ComponentSpec(
        name="gem",
        argv=[
            f"{root}/external_dependencies/GENMO/.venv/bin/python",
            "gear_sonic/scripts/gem_webcam_zmq_publisher.py",
            "--no_imgfeat", "--render", "--render_mode", "opencv",
            "--headless", "--preview_port", str(PREVIEW_PORT),
            "--camera_id", str(camera_id),
        ],
        cwd=root,
        ready_pattern=re.compile(r"\[ZMQ\] Publishing SMPL frames"),
        ready_timeout=180.0,
        parsers=[
            (re.compile(r"Warmup (?P<warmup>\d+)"), "gem_"),
            (re.compile(r"FPS\s+[\d.]+\s+\(avg\s+(?P<fps>[\d.]+)\)"), "gem_"),
            (re.compile(r"(?P<no_person>no person detected)"), "gem_"),
        ],
    )
    bridge_argv = [
        f"{root}/.venv_teleop/bin/python",
        "gear_sonic/scripts/webcam_smpl_streamer.py", "--auto_start",
        # web UI has explicit Arm/E-STOP controls, so no demo session timer
        "--session_timeout", "0",
    ]
    if upper_body:
        bridge_argv.append("--upper_body")
    specs["bridge"] = ComponentSpec(
        name="bridge",
        argv=bridge_argv,
        cwd=root,
        ready_pattern=re.compile(r"Buffer full"),
        ready_timeout=600.0,  # includes GEM warmup + waiting for a person
        parsers=[
            (
                re.compile(r"mode=(?P<mode>\w+) out_fps=(?P<out_fps>[\d.]+).*?gem_frames=(?P<gem_frames>\d+)"),
                "bridge_",
            ),
        ],
    )
    return specs


class Orchestrator:
    START_ORDER = ["sim", "cpp", "gem", "bridge"]

    def __init__(self):
        self.components: dict[str, Component] = {}
        self.global_state = "idle"  # idle|preflight|starting|running|stopping|error
        self.detail = ""
        self.mode = "sim"
        self.preflight_results: list[dict] = []
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------ preflight
    def preflight(self, mode: str) -> list[dict]:
        checks = []

        def add(name, ok, detail, hard=True):
            checks.append({"name": name, "ok": bool(ok), "detail": detail, "hard": hard})

        for label, path in [
            ("venv sim", REPO_ROOT / ".venv_sim/bin/python"),
            ("venv bridge", REPO_ROOT / ".venv_teleop/bin/python"),
            ("venv GEM", REPO_ROOT / "external_dependencies/GENMO/.venv/bin/python"),
            ("deploy.sh", REPO_ROOT / "gear_sonic_deploy/deploy.sh"),
        ]:
            if label == "venv sim" and mode != "sim":
                continue
            add(label, os.access(path, os.X_OK), str(path))

        cams = sorted(glob.glob("/dev/video*"))
        add("webcam", bool(cams), ", ".join(cams) or "no /dev/video* device")

        for port in CHECK_PORTS:
            free = self._port_free(port)
            add(f"port {port}", free, "free" if free else "IN USE — kill leftover processes (Cleanup button)")

        if mode == "sim":
            route_ok = self._multicast_ok()
            add(
                "multicast (lo)", route_ok,
                "ok" if route_ok else
                "run: sudo ip link set lo multicast on && sudo ip route add 224.0.0.0/4 dev lo",
            )

        used = self._gpu_used_mib()
        if used is not None:
            add("GPU memory", used < 6000, f"{used} MiB in use", hard=False)

        self.preflight_results = checks
        return checks

    @staticmethod
    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", port))
                return True
            except OSError:
                return False

    @staticmethod
    def _multicast_ok() -> bool:
        try:
            out = subprocess.run(
                ["ip", "route", "get", "224.0.0.1"], capture_output=True, text=True, timeout=5
            ).stdout
            return "dev lo" in out
        except Exception:
            return False

    @staticmethod
    def _gpu_used_mib() -> int | None:
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().splitlines()
            return int(out[0]) if out else None
        except Exception:
            return None

    # ------------------------------------------------------------------ lifecycle
    def start(self, mode: str = "sim", camera_id: int = 0, upper_body: bool = False) -> bool:
        with self._lock:
            if self.global_state in ("starting", "running", "stopping"):
                return False
            self.mode = mode
            self.global_state = "preflight"
            self.detail = ""
            self._worker = threading.Thread(
                target=self._start_sequence, args=(mode, camera_id, upper_body), daemon=True
            )
            self._worker.start()
            return True

    def _start_sequence(self, mode: str, camera_id: int, upper_body: bool = False):
        checks = self.preflight(mode)
        hard_fail = [c for c in checks if c["hard"] and not c["ok"]]
        if hard_fail:
            self.global_state = "error"
            self.detail = "Pre-flight failed: " + "; ".join(c["name"] for c in hard_fail)
            return

        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        specs = _make_specs(mode, camera_id, upper_body)
        self.components = {name: Component(spec) for name, spec in specs.items()}
        self.global_state = "starting"

        for name in self.START_ORDER:
            comp = self.components.get(name)
            if comp is None:
                continue
            self.detail = f"starting {name}…"
            comp.start(env)
            if not comp.wait_ready():
                self.global_state = "error"
                self.detail = f"{name} failed to become ready (see its log)"
                self._stop_all()
                return

        time.sleep(1.0)
        self.detail = "sending ']' to C++ (start policy)"
        self.components["cpp"].send("]")
        self.global_state = "running"
        self.detail = "policy started — if the robot stands but does not imitate, press 'Toggle stream'"

        threading.Thread(target=self._monitor, daemon=True).start()

    def _monitor(self):
        while self.global_state == "running":
            for name, comp in self.components.items():
                if comp.state == "exited" or (comp.state == "ready" and not comp.alive()):
                    self.global_state = "error"
                    self.detail = f"component '{name}' exited unexpectedly"
                    return
            time.sleep(1.0)

    def stop(self) -> bool:
        with self._lock:
            if self.global_state == "stopping":
                return False
            self.global_state = "stopping"
            self.detail = ""
            self._worker = threading.Thread(target=self._stop_sequence, daemon=True)
            self._worker.start()
            return True

    def _stop_sequence(self):
        bridge = self.components.get("bridge")
        cpp = self.components.get("cpp")
        if bridge is not None and bridge.alive():
            bridge.send("o")  # sends the STOP command to the controller
            time.sleep(1.0)
        if cpp is not None and cpp.alive():
            cpp.send("o")
            time.sleep(0.5)
        self._stop_all()
        self.global_state = "idle"
        self.detail = "stopped"

    def _stop_all(self):
        for name in reversed(self.START_ORDER):
            comp = self.components.get(name)
            if comp is not None:
                comp.stop()
        subprocess.run(["pkill", "-9", "-f", "g1_deploy_onnx_ref"], capture_output=True)

    def estop(self):
        cpp = self.components.get("cpp")
        if cpp is not None and cpp.alive():
            cpp.send("o")
            self.detail = "E-STOP sent to controller"

    def cleanup(self):
        """Kill leftover processes from previous runs (frees the ZMQ ports)."""
        for pat in ("g1_deploy_onnx_ref", "run_sim_loop", "webcam_smpl_streamer", "gem_webcam_zmq_publisher"):
            subprocess.run(["pkill", "-9", "-f", pat], capture_output=True)
        time.sleep(1.0)

    def send_key(self, component: str, key: str) -> bool:
        key = key.lower()
        if key not in ALLOWED_KEYS.get(component, set()):
            return False
        comp = self.components.get(component)
        if comp is None or not comp.alive():
            return False
        comp.send(KEY_MAP[key])
        return True

    def status(self) -> dict:
        return {
            "global_state": self.global_state,
            "detail": self.detail,
            "mode": self.mode,
            "components": {name: comp.status() for name, comp in self.components.items()},
            "preflight": self.preflight_results,
        }
