#!/usr/bin/env python3
"""Edge agent: the laptop half of the split ACT deployment.

Owns everything at the robot side so the WAN link is never inside a control
loop and never between the stop button and the motors:

  - SO-101 bus + two cameras via lerobot (local, 10Hz)
  - the app backend the Flutter app talks to (commands + /video_feed, one port)
  - a websocket client that streams observations OUT to the ACT policy server
    and replays the returned action chunks locally
  - safety: per-step delta clamp, absolute joint limits, chunk-consumption cap,
    watchdog (hold -> home on link loss), local home primitive
  - auto-stop (zero-velocity / chunk-stability), ported from main_act.py but
    time-based instead of tick-based because the inference rate is adaptive

Inference pacing is "one request in flight": a new observation is sent the
moment the previous chunk arrives, so the effective rate adapts to the link.
Chunks are indexed by (now - t_obs) on the laptop's own monotonic clock (the
server echoes t_obs untouched), which skips exactly the steps the network
delay already consumed.

First run per cable-plug: `python -m so101_pipeline.runtime.main_edge
--identify` - camera indices shuffle on every reconnect, so the mapping is
confirmed by eye and stored in edge_cameras.json.

Test without GPU/robot motion: run `act_policy_server --mock` locally and
start this agent with --dry_run (never enables torque, never writes goals).
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import msgpack
import numpy as np

from so101_pipeline.interfaces.command_bridge import normalize_command
from so101_pipeline.interfaces.edge_app_backend import CommandStore, build_app, start_app_server
from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.utils.robot_utils import precise_sleep

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
HOME_POSITION_DEG = np.asarray([-3.2, -104.8, 105.8, 78.6, 0.3, 2.3], dtype=np.float32)
HOME_MOVE_DURATION_S = 2.5
HOME_MOVE_HZ = 30
ACT_ALLOWED_TASKS = (
    "pick the banana and place it in the green basket",
    "pick the banana and place it in the yellow basket",
    "pick the banana and place it in the blue basket",
)
# Absolute command limits (deg; gripper is 0-100). Home pose must fit inside.
JOINT_LIMITS = {
    "shoulder_pan": (-115.0, 115.0),
    "shoulder_lift": (-115.0, 115.0),
    "elbow_flex": (-115.0, 115.0),
    "wrist_flex": (-115.0, 115.0),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (0.0, 100.0),
}
LIMIT_LOW = np.asarray([JOINT_LIMITS[n][0] for n in JOINT_NAMES], dtype=np.float32)
LIMIT_HIGH = np.asarray([JOINT_LIMITS[n][1] for n in JOINT_NAMES], dtype=np.float32)


# ---------------------------------------------------------------- cameras

def camera_config_path(args) -> Path:
    if args.camera_config:
        return Path(args.camera_config)
    return Path(__file__).resolve().parents[2] / "edge_cameras.json"


def identify_cameras(args) -> None:
    """Probe indices, save preview frames, ask which is which, store mapping."""
    out_dir = camera_config_path(args).parent
    found = []
    for index in range(args.identify_max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(8):  # let exposure settle
            cap.read()
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        preview = out_dir / f"edge_camera_preview_{index}.png"
        cv2.imwrite(str(preview), frame)
        found.append(index)
        print(f"  index {index}: preview saved -> {preview}")
    if len(found) < 2:
        raise SystemExit(f"need at least 2 cameras, found indices {found}")
    top = int(input(f"top camera index {found}: ").strip())
    wrist = int(input(f"wrist camera index {found}: ").strip())
    if top == wrist or top not in found or wrist not in found:
        raise SystemExit("invalid selection")
    path = camera_config_path(args)
    path.write_text(json.dumps({"top": top, "wrist": wrist}, indent=2))
    print(f"saved: {path}")


def load_camera_mapping(args) -> dict:
    path = camera_config_path(args)
    if not path.exists():
        raise SystemExit(f"{path} not found - run with --identify first (indices shuffle on every reconnect)")
    mapping = json.loads(path.read_text())
    return {"top": int(mapping["top"]), "wrist": int(mapping["wrist"])}


# ---------------------------------------------------------------- policy link

class PolicyClient(threading.Thread):
    """One request in flight; reconnects with backoff; laptop dials out."""

    def __init__(self, url: str, on_chunk, on_error):
        super().__init__(name="policy-client", daemon=True)
        self.url = url
        self.on_chunk = on_chunk
        self.on_error = on_error
        self._requests: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self.connected = threading.Event()

    def submit(self, request: dict, snapshot: dict) -> bool:
        """Non-blocking; False if a request is already in flight/queued."""
        try:
            self._requests.put_nowait((request, snapshot))
            return True
        except queue.Full:
            return False

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from websockets.sync.client import connect

        backoff = 0.5
        while not self._stop.is_set():
            try:
                with connect(self.url, max_size=32 * 1024 * 1024, open_timeout=5) as ws:
                    self.connected.set()
                    backoff = 0.5
                    print(f"policy server connected: {self.url}")
                    while not self._stop.is_set():
                        try:
                            request, snapshot = self._requests.get(timeout=0.2)
                        except queue.Empty:
                            continue
                        ws.send(msgpack.packb(request))
                        reply = msgpack.unpackb(ws.recv(timeout=10.0), raw=False)
                        if reply.get("type") == "chunk":
                            self.on_chunk(reply, snapshot)
                        else:
                            self.on_error(f"server error: {reply.get('message')}")
            except Exception as exc:  # noqa: BLE001 - any link failure -> reconnect
                self.connected.clear()
                self.on_error(f"policy link down ({type(exc).__name__}: {exc}); retrying in {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
        self.connected.clear()


# ---------------------------------------------------------------- agent

class EdgeAgent:
    def __init__(self, args):
        self.args = args
        self.store = CommandStore()
        mapping = load_camera_mapping(args)
        print(f"cameras: top=idx{mapping['top']} wrist=idx{mapping['wrist']} (from {camera_config_path(args)})")
        self.robot = make_robot_from_config(
            SOFollowerRobotConfig(
                id=args.robot_id,
                port=args.robot_port,
                calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
                disable_torque_on_disconnect=True,
                use_degrees=True,
                cameras={
                    name: OpenCVCameraConfig(
                        index_or_path=index, fps=30, width=640, height=480, backend=Cv2Backends.MSMF
                    )
                    for name, index in mapping.items()
                },
            )
        )
        self._lock = threading.Lock()
        self._chunk = None            # np (n_steps, 6) in JOINT_NAMES order
        self._chunk_t0 = 0.0          # laptop monotonic time of the source observation
        self._chunk_fps = 10
        self._chunk_arrival = 0.0
        self._chunk_obs_joints = None
        self._prev_chunk = None
        self._prev_chunk_t0 = 0.0
        self._zero_since = None
        self._stable_since = None
        self._infer_ms_log = []
        self._episode_id = 0
        self._app_jpeg = None
        self._configured = False
        self._last_submit_t = 0.0
        self.client = PolicyClient(args.server_url, self._handle_chunk, lambda msg: print(f"[link] {msg}"))

    # ---- hardware ----

    def connect(self) -> None:
        self.robot.bus.connect()
        if not self.robot.bus.is_calibrated:
            raise SystemExit("motor calibration mismatch - check --calibration_dir / --robot_id")
        for cam in self.robot.cameras.values():
            cam.connect()
        print(f"bus connected on {self.args.robot_port}; cameras up; dry_run={self.args.dry_run}")

    def read_joints(self) -> np.ndarray:
        positions = self.robot.bus.sync_read("Present_Position")
        return np.asarray([positions[name] for name in JOINT_NAMES], dtype=np.float32)

    def write_joints(self, target: np.ndarray) -> None:
        if self.args.dry_run:
            return
        self.robot.bus.sync_write("Goal_Position", {name: float(target[i]) for i, name in enumerate(JOINT_NAMES)})

    def clamp(self, target: np.ndarray, reference: np.ndarray) -> np.ndarray:
        step = np.asarray(
            [self.args.max_step_deg] * 5 + [self.args.max_step_gripper], dtype=np.float32
        )
        clamped = np.clip(target, reference - step, reference + step)
        return np.clip(clamped, LIMIT_LOW, LIMIT_HIGH)

    def move_home(self, reason: str) -> None:
        print(f"returning home ({reason})...")
        if self.args.dry_run:
            print("dry_run: home skipped")
            return
        current = self.read_joints()
        steps = max(1, round(HOME_MOVE_DURATION_S * HOME_MOVE_HZ))
        for step in range(1, steps + 1):
            alpha = step / steps
            self.write_joints(current + alpha * (HOME_POSITION_DEG - current))
            precise_sleep(1.0 / HOME_MOVE_HZ)
        print("home pose reached")

    # ---- chunk handling (called from the client thread) ----

    def _handle_chunk(self, reply: dict, snapshot: dict) -> None:
        names = reply["joint_names"]
        raw = np.asarray(reply["chunk"], dtype=np.float32)
        try:
            order = [names.index(f"{n}.pos") if f"{n}.pos" in names else names.index(n) for n in JOINT_NAMES]
        except ValueError:
            print(f"[link] chunk joint names {names} don't match {JOINT_NAMES}; dropping")
            return
        chunk = raw[:, order]
        with self._lock:
            if reply["episode_id"] != self._episode_id:
                return  # stale reply from a previous task
            self._prev_chunk, self._prev_chunk_t0 = self._chunk, self._chunk_t0
            self._chunk = chunk
            self._chunk_t0 = reply["t_obs"]
            self._chunk_fps = int(reply["fps"])
            self._chunk_arrival = time.monotonic()
            self._chunk_obs_joints = snapshot["joints"]
            self._infer_ms_log.append(reply.get("infer_ms", 0.0))
        self._update_auto_stop(chunk, reply["t_obs"], snapshot["joints"])

    def _update_auto_stop(self, chunk, t0, obs_joints) -> None:
        args, now = self.args, time.monotonic()
        horizon = min(args.auto_stop_horizon, len(chunk))
        zero_delta = np.abs(chunk[:horizon] - obs_joints[None, :])
        is_zero = zero_delta.max() <= args.zero_velocity_max_delta_deg and zero_delta.mean() <= args.zero_velocity_mean_delta_deg
        self._zero_since = (self._zero_since or now) if is_zero else None

        is_stable = False
        with self._lock:
            prev, prev_t0 = self._prev_chunk, self._prev_chunk_t0
        if prev is not None:
            offset = round((t0 - prev_t0) * self._chunk_fps)
            aligned = min(horizon, len(chunk), len(prev) - offset)
            if 0 < offset and aligned >= 1:
                delta = np.abs(chunk[:aligned] - prev[offset:offset + aligned])
                is_stable = delta.max() <= args.chunk_stability_max_delta_deg and delta.mean() <= args.chunk_stability_mean_delta_deg
        self._stable_since = (self._stable_since or now) if is_stable else None

    def auto_stop_reason(self, elapsed_s: float) -> str | None:
        now, args = time.monotonic(), self.args
        if (
            self._zero_since is not None
            and elapsed_s >= args.zero_velocity_min_duration_s
            and now - self._zero_since >= args.auto_stop_sustain_s
        ):
            return "zero-velocity chunk"
        if (
            self._stable_since is not None
            and elapsed_s >= args.chunk_stability_min_duration_s
            and now - self._stable_since >= args.auto_stop_sustain_s
        ):
            return "stable chunk"
        return None

    # ---- observation upload ----

    def maybe_submit_observation(self, task: str, joints: np.ndarray, frames: dict) -> None:
        if time.monotonic() - self._last_submit_t < self.args.min_infer_period_s:
            return
        encode = [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
        images = {}
        for name, frame in frames.items():
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode)
            if not ok:
                return
            images[name] = jpeg.tobytes()
        request = {
            "type": "predict",
            "episode_id": self._episode_id,
            "t_obs": time.monotonic(),
            "task": task,
            "joints": {name: float(joints[i]) for i, name in enumerate(JOINT_NAMES)},
            "images": images,
        }
        if self.client.submit(request, {"joints": joints.copy()}):
            self._last_submit_t = time.monotonic()

    # ---- app video ----

    def publish_app_frame(self, frames: dict) -> None:
        ordered = [frames[n] for n in ("top", "wrist") if n in frames]
        if not ordered:
            return
        combined = np.hstack(ordered) if len(ordered) > 1 else ordered[0]
        ok, jpeg = cv2.imencode(
            ".jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), self.args.app_jpeg_quality]
        )
        if ok:
            self._app_jpeg = jpeg.tobytes()

    def app_jpeg(self):
        return self._app_jpeg

    # ---- episode ----

    def run_episode(self, task: str, consumed_stop_ts) -> str | None:
        """Returns the stop ts if the app stopped us, else None. Home is caller's job."""
        args = self.args
        with self._lock:
            self._episode_id += 1
            self._chunk = self._prev_chunk = self._chunk_obs_joints = None
            self._infer_ms_log = []
        self._zero_since = self._stable_since = None
        if not args.dry_run:
            if not self._configured:
                # Same motor setup robot.connect() would do in main_act: position
                # mode, P=16 (default 32 causes shakiness), gripper protections.
                self.robot.configure()
                self._configured = True
                print("servo gains configured (P=16, matching main_act)")
            self.robot.bus.enable_torque()
        last_sent = self.read_joints()
        start = time.monotonic()
        tick, chunks_used, clamp_hits, holds = 0, 0, 0, 0
        print(f"episode {self._episode_id} start: {task!r}")

        while time.monotonic() - start < args.duration_s:
            loop_t = time.monotonic()
            joints = self.read_joints()
            frames = {name: cam.async_read() for name, cam in self.robot.cameras.items()}
            self.publish_app_frame(frames)
            self.maybe_submit_observation(task, joints, frames)

            stop = self.store.snapshot()["stop"]
            if stop["requested"] and stop["ts"] != consumed_stop_ts:
                print("app stop received")
                return stop["ts"]

            with self._lock:
                chunk, t0, fps, arrival = self._chunk, self._chunk_t0, self._chunk_fps, self._chunk_arrival
                prev, prev_t0 = self._prev_chunk, self._prev_chunk_t0
            now = time.monotonic()
            if chunk is None:
                if now - start > args.first_chunk_timeout_s:
                    print(f"no chunk within {args.first_chunk_timeout_s}s - aborting episode")
                    return None
            else:
                index = int((now - t0) * fps)
                usable = min(len(chunk), args.consume_cap_steps)
                if index < 0:
                    index = 0
                if index < usable:
                    step_target = chunk[index]
                    if args.blend_prev_chunk and prev is not None and now - prev_t0 < 2.0:
                        # Poor-man's temporal ensemble: the checkpoint was rolled out
                        # with per-tick ensembling; averaging the two most recent
                        # chunks at the same wall-clock step recovers most of the
                        # smoothing without per-tick inference.
                        prev_index = int((now - prev_t0) * fps)
                        if 0 <= prev_index < len(prev):
                            step_target = 0.5 * step_target + 0.5 * prev[prev_index]
                    target = self.clamp(step_target, last_sent)
                    if not np.allclose(target, step_target, atol=1e-3):
                        clamp_hits += 1
                    self.write_joints(target)
                    last_sent = target
                    chunks_used += 1
                else:
                    holds += 1
                    if now - arrival > args.link_lost_home_s:
                        print(f"link stale for {now - arrival:.1f}s - safety stop")
                        return None

            reason = self.auto_stop_reason(now - start)
            if reason:
                print(f"auto stop: {reason}")
                self.store.record_success(task)
                return None

            if args.print_every > 0 and tick % args.print_every == 0:
                age = (now - t0) if chunk is not None else float("nan")
                infer = self._infer_ms_log[-1] if self._infer_ms_log else float("nan")
                print(
                    f"[edge {tick:04d}] chunk_age={age:5.2f}s steps={chunks_used} holds={holds} "
                    f"clamps={clamp_hits} infer={infer:.0f}ms link={'up' if self.client.connected.is_set() else 'DOWN'}"
                )
            tick += 1
            precise_sleep(max(1.0 / args.control_fps - (time.monotonic() - loop_t), 0.0))

        print(f"episode time limit ({args.duration_s}s) reached")
        return None

    # ---- main ----

    def run(self) -> None:
        args = self.args
        self.connect()
        start_app_server(build_app(self.store, self.app_jpeg), args.app_host, args.app_port)
        print(f"app backend: http://{args.app_host}:{args.app_port}  (video: /video_feed)")
        self.client.start()

        state = self.store.snapshot()
        consumed_instruction_ts = (state["instruction"] or {}).get("ts")
        consumed_stop_ts = state["stop"].get("ts")
        print("waiting for app command. allowed tasks:")
        for allowed in ACT_ALLOWED_TASKS:
            print(f"  - {allowed}")

        try:
            while True:
                # idle: keep the app camera view alive at ~10fps
                frames = {name: cam.async_read() for name, cam in self.robot.cameras.items()}
                self.publish_app_frame(frames)

                state = self.store.snapshot()
                instruction = state["instruction"] or {}
                ts, text = instruction.get("ts"), (instruction.get("text") or "").strip()
                if ts and ts != consumed_instruction_ts:
                    consumed_instruction_ts = ts
                    task = normalize_command(text)
                    if task in ACT_ALLOWED_TASKS:
                        if task != text:
                            print(f"command normalized: {text!r} -> {task!r}")
                        stop_ts = self.run_episode(task, consumed_stop_ts)
                        if stop_ts:
                            consumed_stop_ts = stop_ts
                        self.move_home("episode finished")
                        print("ready for next app command")
                    else:
                        print(f"unsupported app command ignored: {text!r}")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("interrupted")
        finally:
            self.client.stop()
            if not args.dry_run and self.robot.bus.is_connected:
                self.move_home("shutdown")
                try:
                    self.robot.bus.disable_torque()
                    print("torque disabled")
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: failed to disable torque: {exc}")
            if self.robot.bus.is_connected:
                self.robot.bus.disconnect()
            for cam in self.robot.cameras.values():
                try:
                    cam.disconnect()
                except Exception:  # noqa: BLE001
                    pass


def main() -> None:
    import os

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identify", action="store_true", help="Probe cameras, save previews, store the index mapping.")
    parser.add_argument("--identify_max_index", type=int, default=5)
    parser.add_argument("--camera_config", default=os.getenv("EDGE_CAMERA_CONFIG"))
    parser.add_argument("--server_url", default=os.getenv("ACT_SERVER_URL", "ws://127.0.0.1:8765"))
    parser.add_argument("--robot_port", default=os.getenv("ROBOT_PORT", "COM3"))
    parser.add_argument("--robot_id", default=os.getenv("ROBOT_ID", "my_follower"))
    parser.add_argument("--calibration_dir", default=os.getenv("LEROBOT_CALIBRATION_DIR"))
    parser.add_argument("--app_host", default="0.0.0.0")
    parser.add_argument("--app_port", type=int, default=8000)
    parser.add_argument("--control_fps", type=int, default=10)
    parser.add_argument("--jpeg_quality", type=int, default=70, help="Policy observation JPEG quality.")
    parser.add_argument("--app_jpeg_quality", type=int, default=70)
    parser.add_argument("--duration_s", type=float, default=30.0)
    parser.add_argument("--consume_cap_steps", type=int, default=10, help="Max chunk steps replayed before holding (1.0s at 10Hz).")
    parser.add_argument("--first_chunk_timeout_s", type=float, default=5.0)
    parser.add_argument("--link_lost_home_s", type=float, default=5.0)
    parser.add_argument("--min_infer_period_s", type=float, default=0.0, help="Throttle observation uploads so several chunk steps replay between inferences (0 = adaptive, one-in-flight).")
    parser.add_argument("--blend_prev_chunk", action=argparse.BooleanOptionalAction, default=True, help="Average the two most recent chunks at the aligned step (recovers temporal-ensemble smoothing).")
    parser.add_argument("--max_step_deg", type=float, default=6.0, help="Per-tick clamp for the five arm joints.")
    parser.add_argument("--max_step_gripper", type=float, default=25.0)
    parser.add_argument("--auto_stop_horizon", type=int, default=30)
    parser.add_argument("--auto_stop_sustain_s", type=float, default=2.0)
    parser.add_argument("--zero_velocity_min_duration_s", type=float, default=12.0)
    parser.add_argument("--zero_velocity_max_delta_deg", type=float, default=3.0)
    parser.add_argument("--zero_velocity_mean_delta_deg", type=float, default=0.8)
    parser.add_argument("--chunk_stability_min_duration_s", type=float, default=10.0)
    parser.add_argument("--chunk_stability_max_delta_deg", type=float, default=6.0)
    parser.add_argument("--chunk_stability_mean_delta_deg", type=float, default=1.5)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--dry_run", action="store_true", help="Never enable torque or write goals; log only.")
    args = parser.parse_args()

    if args.identify:
        identify_cameras(args)
        return
    EdgeAgent(args).run()


if __name__ == "__main__":
    main()
