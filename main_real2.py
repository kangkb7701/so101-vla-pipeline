import os
import sys
import time
import threading
import re
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from PIL import Image

from application.command_bridge import CommandBridgeConfig, UserCommandBridge
from application.camera_source import open_dual_camera

current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.join(current_dir, "proto")
if proto_dir not in sys.path:
    sys.path.append(proto_dir)


# ===================================================================
# Task Parsing
# ===================================================================
def _clean_task_phrase(value: str | None) -> str:
    if not value:
        return ""
    text = value.strip().strip("\"'` ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[.?!]+$", "", text).strip()
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text


def parse_pick_place_task(task_description: str) -> tuple[str | None, str | None]:
    """Extract target object and placement location from common pick-and-place text."""
    if not task_description:
        return None, None

    text = _clean_task_phrase(task_description).lower()
    patterns = [
        r"^pick(?:\s+up)?\s+(?P<object>.+?)\s+(?:and\s+)?place\s+(?:it|them|the\s+object)?\s*(?:in|into|on|onto|to)\s+(?P<location>.+)$",
        r"^pick(?:\s+up)?\s+(?P<object>.+?)\s+(?:and\s+|then\s+)?put\s+(?:it|them|the\s+object)?\s*(?:in|into|on|onto|to)\s+(?P<location>.+)$",
        r"^move\s+(?P<object>.+?)\s+(?:in|into|on|onto|to)\s+(?P<location>.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            target = _clean_task_phrase(match.group("object"))
            location = _clean_task_phrase(match.group("location"))
            if target and location:
                return target, location
    return None, None


# ===================================================================
# Config
# ===================================================================
USE_DIRECT_CAMERA_SOURCE = True
CAM_TOP_INDEX = 0
CAM_WRIST_INDEX = 2
JETSON_TOP_STREAM_URL = "http://127.0.0.1:8080/top"
JETSON_WRIST_STREAM_URL = "http://127.0.0.1:8080/wrist"

DEFAULT_URDF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "so101_new_calib.urdf")
URDF_PATH = os.getenv("ROBOT_URDF", DEFAULT_URDF)

CONTROL_HZ = 10.0
CONTROL_PERIOD = 1.0 / CONTROL_HZ
ACTION_DIM = 7
SHOW_VIZ = os.getenv("SHOW_VIZ", "true").lower() in {"1", "true", "yes"}
MAX_STEPS = 2000
FIRST_CHUNK_TIMEOUT = 15.0

HZ_WINDOW_N = 20
HZ_REPORT_EVERY = 20
LOG_DETAIL_TIMING = True

# Match the fine-tuning action convention by default: Octo outputs real one-step
# EE deltas. Set ACTION_SCALING=0.3 / JOINT_SMOOTHING_ALPHA=0.9 to recover the
# previous conservative runtime behavior.
ACTION_SCALING = float(os.getenv("ACTION_SCALING", "1.0"))
JOINT_SMOOTHING_ALPHA = float(os.getenv("JOINT_SMOOTHING_ALPHA", "1.0"))

TASK_DESCRIPTION = os.getenv(
    "TASK_DESCRIPTION", "pick the banana and place it in the yellow basket"
)
USE_USER_COMMAND_TASK_BRIDGE = os.getenv("USE_USER_COMMAND_TASK_BRIDGE", "false").lower() in {"1", "true", "yes"}
USER_COMMAND_ENDPOINT = os.getenv("USER_COMMAND_ENDPOINT", "http://127.0.0.1:8000/command/latest")
USER_COMMAND_TIMEOUT_S = float(os.getenv("USER_COMMAND_TIMEOUT_S", "1.0"))
USER_COMMAND_POLL_S = float(os.getenv("USER_COMMAND_POLL_S", "0.5"))
VP_TARGET_OBJECT_OVERRIDE = os.getenv("VP_TARGET_OBJECT")
VP_TARGET_LOCATION_OVERRIDE = os.getenv("VP_TARGET_LOCATION")
QWEN_MODEL_ID = os.getenv("QWEN_MODEL_ID", "Qwen/Qwen2.5-VL-3B-Instruct")

POLICY_BACKEND = os.getenv("POLICY_BACKEND", "vp_vla").strip().lower()
if POLICY_BACKEND not in {"octo", "vp_vla"}:
    raise ValueError(f"Unsupported POLICY_BACKEND={POLICY_BACKEND!r}; use octo or vp_vla")
OCTO_SERVER_HOST = os.getenv("OCTO_SERVER_HOST", "localhost")
OCTO_SERVER_PORT = int(os.getenv("OCTO_SERVER_PORT", "50051"))
VP_VLA_SERVER_HOST = os.getenv("VP_VLA_SERVER_HOST", "127.0.0.1")
VP_VLA_SERVER_PORT = int(os.getenv("VP_VLA_SERVER_PORT", "10093"))
VP_VLA_STATS_PATH = os.getenv(
    "VP_VLA_STATS_PATH",
    "/home/aivlab/kkb_capstone/outputs/train/vp_vla/"
    "so101_file000_005_ee7_vp_qwen_oft_fast3090/dataset_statistics.json",
)
VP_VLA_UNNORM_KEY = os.getenv("VP_VLA_UNNORM_KEY") or None

USE_VP_VISUAL_PROMPT = os.getenv("USE_VP_VISUAL_PROMPT", "true").lower() in {"1", "true", "yes"}
VP_SAM3_HOST = os.getenv("VP_SAM3_HOST", "127.0.0.1")
VP_SAM3_PORT = int(os.getenv("VP_SAM3_PORT", "10094"))
VP_SAM3_TIMEOUT_S = float(os.getenv("VP_SAM3_TIMEOUT_S", "5.0"))
VP_SAM3_THRESHOLD = float(os.getenv("VP_SAM3_THRESHOLD", "0.5"))
VP_SAM3_MASK_THRESHOLD = float(os.getenv("VP_SAM3_MASK_THRESHOLD", "0.5"))
VP_PICK_SAM_INTERVAL_S = float(os.getenv("VP_PICK_SAM_INTERVAL_S", "0.5"))
VP_PLACE_SAM_INTERVAL_S = float(os.getenv("VP_PLACE_SAM_INTERVAL_S", "2.0"))
VP_PREFETCH_PLACE_ON_PICK = os.getenv("VP_PREFETCH_PLACE_ON_PICK", "true").lower() in {"1", "true", "yes"}
VP_READY_TIMEOUT_S = float(os.getenv("VP_READY_TIMEOUT_S", "5.0"))
VP_REQUIRE_OVERLAY = os.getenv("VP_REQUIRE_OVERLAY", "true").lower() in {"1", "true", "yes"}

USE_VP_EVENT_TRIGGER = os.getenv("USE_VP_EVENT_TRIGGER", "false").lower() in {"1", "true", "yes"}
VP_EVENT_TRIGGER_POLL_S = 1.0
VP_EVENT_TRIGGER_FRAME_STALE_S = 2.0
VP_EVENT_TRIGGER_MIN_PICK_S = 2.0
VP_EVENT_TRIGGER_CONFIDENCE = 0.85
VP_EVENT_TRIGGER_VOTE_WINDOW = 3
VP_EVENT_TRIGGER_VOTES_REQUIRED = 2
VP_REQUIRE_EVENT_TRIGGER = os.getenv("VP_REQUIRE_EVENT_TRIGGER", "true").lower() in {"1", "true", "yes"}

# Keep the full VP-VLA horizon so overlapping chunks cover its inference latency.
EXECUTE_CHUNK_STEPS = int(
    os.getenv("EXECUTE_CHUNK_STEPS", "16" if POLICY_BACKEND == "vp_vla" else "4")
)

# VP-VLA-style temporal ensembling over overlapping action chunks. Each new chunk
# is aligned by control tick; the current action is a similarity-weighted average
# of all historical chunk predictions that refer to the current tick.
ACTION_TEMPORAL_ENSEMBLE_DEFAULT = POLICY_BACKEND == "vp_vla"
ACTION_TEMPORAL_ENSEMBLE = os.getenv(
    "ACTION_TEMPORAL_ENSEMBLE",
    "true" if ACTION_TEMPORAL_ENSEMBLE_DEFAULT else "false",
).lower() in {"1", "true", "yes"}
ACTION_ENSEMBLE_ALPHA = float(os.getenv("ACTION_ENSEMBLE_ALPHA", "2.0"))
ACTION_ENSEMBLE_MAX_HISTORY = int(os.getenv("ACTION_ENSEMBLE_MAX_HISTORY", "4"))
ACTION_ENSEMBLE_WEIGHT_DIMS = int(os.getenv("ACTION_ENSEMBLE_WEIGHT_DIMS", "6"))

# Binary gripper checkpoints output open intent: 1=open, 0=close. Close is
# converted to a gradual absolute-position command in IKController.
GRIPPER_CONTROL_MODE = os.getenv("GRIPPER_CONTROL_MODE", "binary").lower()
GRIPPER_BINARY_OPEN_POS = float(os.getenv("GRIPPER_BINARY_OPEN_POS", "40.0"))
GRIPPER_BINARY_CLOSE_MIN_POS = float(os.getenv("GRIPPER_BINARY_CLOSE_MIN_POS", "5.0"))
GRIPPER_BINARY_CLOSE_STEP_DEG = float(os.getenv("GRIPPER_BINARY_CLOSE_STEP_DEG", "2.0"))
GRIPPER_BINARY_OPEN_THRESHOLD = float(os.getenv("GRIPPER_BINARY_OPEN_THRESHOLD", "0.5"))
GRIPPER_BINARY_CLOSE_MOTION_SCALE = float(os.getenv("GRIPPER_BINARY_CLOSE_MOTION_SCALE", "0.25"))

APP_VIDEO_SERVER_ENABLED = os.getenv("APP_VIDEO_SERVER_ENABLED", "false").lower() in {"1", "true", "yes"}
APP_VIDEO_SERVER_HOST = "0.0.0.0"
APP_VIDEO_SERVER_PORT = 8010
APP_VIDEO_STREAM_SLEEP_S = 0.03
APP_VIDEO_JPEG_QUALITY = 70


# ===================================================================
# Shared State
# ===================================================================
class ChunkBuffer:
    def __init__(
        self,
        action_dim: int = 7,
        temporal_ensemble: bool = False,
        ensemble_alpha: float = 0.1,
        max_history: int = 16,
        weight_dims: int = 6,
    ):
        self.lock = threading.Lock()
        self.action_dim = action_dim
        self.temporal_ensemble = temporal_ensemble
        self.ensemble_alpha = float(ensemble_alpha)
        self.weight_dims = max(1, min(int(weight_dims), action_dim))
        self.chunk = None
        self.idx = 0
        self.control_tick = 0
        self.history = deque(maxlen=max(1, int(max_history)))
        self.first_chunk_event = threading.Event()

    def replace(self, new_chunk):
        new_chunk = np.asarray(new_chunk, dtype=np.float32)
        with self.lock:
            if self.temporal_ensemble:
                # start_tick is the control tick for the next pop(). Older chunks
                # remain useful while their offset still overlaps this tick.
                self.history.append((self.control_tick, new_chunk.copy()))
                self._drop_expired_locked()
            else:
                self.chunk = new_chunk
                self.idx = 0
        self.first_chunk_event.set()

    def clear(self):
        with self.lock:
            self.chunk = None
            self.idx = 0
            self.control_tick = 0
            self.history.clear()

    def _drop_expired_locked(self):
        if not self.history:
            return
        kept = deque(maxlen=self.history.maxlen)
        for start_tick, chunk in self.history:
            if self.control_tick - start_tick < len(chunk):
                kept.append((start_tick, chunk))
        self.history = kept

    def _ensemble_candidates(self, candidates: list[np.ndarray]) -> np.ndarray:
        if len(candidates) == 1:
            return candidates[0].copy()
        preds = np.stack(candidates, axis=0).astype(np.float32)
        ref = preds[-1, : self.weight_dims]
        prev = preds[:, : self.weight_dims]
        denom = np.linalg.norm(prev, axis=1) * np.linalg.norm(ref) + 1e-7
        cos_similarity = np.sum(prev * ref[None, :], axis=1) / denom
        weights = np.exp(self.ensemble_alpha * cos_similarity)
        weights = weights / np.sum(weights)
        ensembled = np.sum(weights[:, None] * preds, axis=0).astype(np.float32)
        # Do not average binary gripper intent; use the newest prediction.
        ensembled[self.weight_dims :] = preds[-1, self.weight_dims :]
        return ensembled

    def _stale_action(self) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float32)
        if self.action_dim > 6:
            action[6] = 1.0  # binary gripper: stale/no-op must keep gripper open, not close
        return action

    def pop(self):
        with self.lock:
            if not self.temporal_ensemble:
                if self.chunk is None or self.idx >= len(self.chunk):
                    return self._stale_action(), True
                action = self.chunk[self.idx].copy()
                self.idx += 1
                return action, False

            self._drop_expired_locked()
            candidates = []
            for start_tick, chunk in self.history:
                offset = self.control_tick - start_tick
                if 0 <= offset < len(chunk):
                    candidates.append(chunk[offset])

            self.control_tick += 1
            if not candidates:
                return self._stale_action(), True
            return self._ensemble_candidates(candidates), False

    def wait_first(self, timeout: float = 10.0) -> bool:
        return self.first_chunk_event.wait(timeout)


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.q_deg = None
        self.ready_event = threading.Event()

    def set(self, q_deg):
        with self.lock:
            self.q_deg = list(q_deg)
        self.ready_event.set()

    def get(self):
        with self.lock:
            return None if self.q_deg is None else list(self.q_deg)


class SharedFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set(self, frame_bgr):
        with self.lock:
            self.frame = frame_bgr

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class SharedFrames:
    """Raw top/wrist RGB frames for Qwen VP event trigger."""

    def __init__(self):
        self.lock = threading.Lock()
        self.top = None
        self.wrist = None
        self.t = None

    def set(self, top_rgb, wrist_rgb, t_monotonic):
        with self.lock:
            self.top = top_rgb
            self.wrist = wrist_rgb
            self.t = t_monotonic

    def get(self):
        with self.lock:
            if self.top is None or self.wrist is None or self.t is None:
                return None, None, float("inf")
            top = self.top.copy()
            wrist = self.wrist.copy()
            age_s = time.monotonic() - self.t
        return top, wrist, age_s


class SharedAppFrame:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None

    def set(self, frame_bgr):
        with self.lock:
            self.frame = frame_bgr

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class SharedCameraFrames:
    def __init__(self):
        self.lock = threading.Lock()
        self.top_bgr = None
        self.wrist_bgr = None
        self.t = None
        self.ready_event = threading.Event()

    def set(self, top_bgr, wrist_bgr):
        with self.lock:
            self.top_bgr = top_bgr.copy()
            self.wrist_bgr = wrist_bgr.copy()
            self.t = time.monotonic()
        self.ready_event.set()

    def get(self):
        with self.lock:
            if self.top_bgr is None or self.wrist_bgr is None or self.t is None:
                return None, None, float("inf")
            return self.top_bgr.copy(), self.wrist_bgr.copy(), time.monotonic() - self.t


class SharedVPPhase:
    def __init__(self):
        self.lock = threading.Lock()
        self.phase = "pick"
        self.reason = ""
        self.switched_at_monotonic = None

    def get(self) -> str:
        with self.lock:
            return self.phase

    def set_pick(self, reason: str = "") -> bool:
        with self.lock:
            if self.phase == "pick":
                return False
            self.phase = "pick"
            self.reason = reason
            self.switched_at_monotonic = time.monotonic()
            return True

    def set_place(self, reason: str = "") -> bool:
        with self.lock:
            if self.phase == "place":
                return False
            self.phase = "place"
            self.reason = reason
            self.switched_at_monotonic = time.monotonic()
            return True


class SharedVPEventTrigger:
    def __init__(self, vote_window: int):
        self.lock = threading.Lock()
        self.votes = deque(maxlen=vote_window)
        self.latched = False
        self.last_decision = None

    def record(self, decision, confidence_threshold: float, votes_required: int):
        with self.lock:
            vote = bool(decision.switch_to_place and decision.confidence >= confidence_threshold)
            self.votes.append(vote)
            vote_count = sum(1 for v in self.votes if v)
            if vote_count >= votes_required:
                self.latched = True
            self.last_decision = decision
            return self.latched, vote_count, len(self.votes)


# ===================================================================
# Video Stream
# ===================================================================
def preview_camera_worker(cam_top, cam_wrist, shared_camera_frames, shared_app_frame, stop_event):
    print("📷 추론용 카메라 캡처 시작")
    while not stop_event.is_set():
        ret_top, frame_top = cam_top.read()
        ret_wrist, frame_wrist = cam_wrist.read()
        if not ret_top or not ret_wrist:
            time.sleep(0.02)
            continue
        shared_camera_frames.set(frame_top, frame_wrist)
        combined = np.hstack((frame_top, frame_wrist))
        shared_app_frame.set(combined)
        time.sleep(APP_VIDEO_STREAM_SLEEP_S)
    print("📷 추론용 카메라 캡처 종료")


def wait_for_task_description():
    if TASK_DESCRIPTION:
        return TASK_DESCRIPTION

    command_bridge = UserCommandBridge(
        CommandBridgeConfig(
            enabled=USE_USER_COMMAND_TASK_BRIDGE,
            endpoint=USER_COMMAND_ENDPOINT,
            timeout_s=USER_COMMAND_TIMEOUT_S,
        )
    )
    if not USE_USER_COMMAND_TASK_BRIDGE:
        return ""

    print("⌛ 앱 명령 대기 중: 카메라 화면을 보고 텍스트/음성 명령을 먼저 보내세요.")
    while True:
        task_description = command_bridge.resolve_task_description("")
        if task_description:
            print(f"📝 앱 명령 기반 TASK_DESCRIPTION 적용: '{task_description}'")
            return task_description
        time.sleep(USER_COMMAND_POLL_S)


def start_app_video_server(shared_app_frame, stop_event):
    app = FastAPI(title="main_real2 Video Stream")

    @app.get("/health")
    def health():
        has_frame = shared_app_frame.get() is not None
        return {"ok": True, "has_frame": has_frame}

    @app.get("/video_feed")
    def video_feed():
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), APP_VIDEO_JPEG_QUALITY]

        def gen():
            while not stop_event.is_set():
                frame = shared_app_frame.get()
                if frame is None:
                    time.sleep(0.05)
                    continue
                ok, buf = cv2.imencode(".jpg", frame, encode_param)
                if not ok:
                    time.sleep(0.01)
                    continue
                chunk = buf.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(chunk)).encode() + b"\r\n\r\n" +
                    chunk + b"\r\n"
                )
                time.sleep(APP_VIDEO_STREAM_SLEEP_S)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    def _run():
        import uvicorn
        uvicorn.run(app, host=APP_VIDEO_SERVER_HOST, port=APP_VIDEO_SERVER_PORT, log_level="warning")

    thread = threading.Thread(target=_run, daemon=True, name="AppVideoServerThread")
    thread.start()
    print(f"📺 영상 스트림 서버 시작: http://{APP_VIDEO_SERVER_HOST}:{APP_VIDEO_SERVER_PORT}/video_feed")
    return thread


# ===================================================================
# Workers
# ===================================================================
def inference_worker(
    agent,
    task_description,
    shared_camera_frames,
    shared_state,
    shared_frame,
    shared_app_frame,
    shared_frames,
    vp_phase,
    vp_overlay,
    buffer,
    stop_event,
):
    if not shared_state.ready_event.wait(timeout=5.0):
        print("⚠️ [Infer] 초기 state 대기 실패. 종료.")
        return

    n_infer = 0
    print(f"🔄 [Infer] task: \"{task_description}\"")
    while not stop_event.is_set():
        t0 = time.time()
        if not shared_camera_frames.ready_event.wait(timeout=1.0):
            time.sleep(0.01)
            continue

        frame_top, frame_wrist, frame_age = shared_camera_frames.get()
        if frame_top is None or frame_wrist is None:
            time.sleep(0.01)
            continue

        top_rgb = cv2.cvtColor(frame_top, cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(frame_wrist, cv2.COLOR_BGR2RGB)
        phase = vp_phase.get()

        if vp_overlay is not None and vp_overlay.error is None:
            overlay_tracking_phase = "pick" if agent.backend_name == "vp_vla" else phase
            vp_overlay.submit(top_rgb, overlay_tracking_phase)
            overlay_ready = (
                vp_overlay.has_dual_detection()
                if agent.backend_name == "vp_vla"
                else vp_overlay.has_detection(phase)
            )
            if not overlay_ready:
                ready = (
                    vp_overlay.wait_until_dual_ready(timeout_s=VP_READY_TIMEOUT_S)
                    if agent.backend_name == "vp_vla"
                    else vp_overlay.wait_until_ready(phase, timeout_s=VP_READY_TIMEOUT_S)
                )
                if not ready:
                    msg = (
                        "VP dual overlay not ready"
                        if agent.backend_name == "vp_vla"
                        else f"VP overlay not ready for phase={phase}"
                    )
                    if VP_REQUIRE_OVERLAY:
                        print(f"⚠️ [Infer] {msg}; skipping prediction until overlay is ready")
                        preview_top = frame_top.copy()
                        cv2.putText(preview_top, f"WAITING SAM3 VP OVERLAY | phase={phase}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                        combined_wait = np.hstack((preview_top, frame_wrist))
                        shared_app_frame.set(combined_wait)
                        if SHOW_VIZ:
                            shared_frame.set(combined_wait)
                        continue
                    print(f"⚠️ [Infer] {msg}; using raw top frame this cycle")
            top_model_rgb = (
                vp_overlay.render_both(top_rgb)
                if agent.backend_name == "vp_vla"
                else vp_overlay.render(top_rgb, phase)
            )
        else:
            if vp_overlay is not None and vp_overlay.error is not None and VP_REQUIRE_OVERLAY:
                print(f"🚨 [Infer] VP overlay failed: {vp_overlay.error}")
                stop_event.set()
                return
            top_model_rgb = top_rgb

        if shared_frames is not None:
            shared_frames.set(top_rgb.copy(), wrist_rgb.copy(), time.monotonic())

        model_top_bgr = cv2.cvtColor(top_model_rgb, cv2.COLOR_RGB2BGR)
        if vp_overlay is not None:
            target_ready = vp_overlay.has_detection("pick")
            place_ready = vp_overlay.has_detection("place")
            label = f"VP TOP | phase={phase} | target={int(target_ready)} place={int(place_ready)}"
        else:
            label = "VLA INPUT TOP | raw | VP disabled"
        cv2.putText(model_top_bgr, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

        wrist_viz_bgr = frame_wrist.copy()
        cv2.putText(wrist_viz_bgr, "VLA INPUT WRIST | raw", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        if agent.backend_name == "vp_vla":
            raw_top_viz_bgr = frame_top.copy()
            cv2.putText(raw_top_viz_bgr, "VP-VLA INPUT TOP | raw", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            combined = np.hstack((raw_top_viz_bgr, model_top_bgr, wrist_viz_bgr))
        else:
            combined = np.hstack((model_top_bgr, wrist_viz_bgr))
        shared_app_frame.set(combined)
        if SHOW_VIZ:
            shared_frame.set(combined)

        q_deg = shared_state.get()
        if q_deg is None:
            continue

        try:
            chunk = agent.predict(
                Image.fromarray(top_model_rgb),
                task_description,
                wrist_image=Image.fromarray(wrist_rgb),
                state=q_deg,
                raw_image=Image.fromarray(top_rgb),
            )
        except Exception as exc:
            print(f"⚠️ [Infer] predict 실패: {exc}")
            continue

        if chunk.ndim == 1:
            chunk = chunk.reshape(1, -1)
        if EXECUTE_CHUNK_STEPS > 0 and len(chunk) > EXECUTE_CHUNK_STEPS:
            chunk = chunk[:EXECUTE_CHUNK_STEPS]
        buffer.replace(chunk)

        n_infer += 1
        latency_ms = (time.time() - t0) * 1000.0
        if n_infer == 1 or n_infer % 5 == 0:
            print(f"[Infer #{n_infer:03d}] ⏱️ {latency_ms:5.1f}ms | phase={phase} chunk_len={len(chunk)}")


def vp_event_trigger_worker(
    trigger,
    shared_frames,
    shared_vp_event_trigger,
    vp_phase,
    buffer,
    stop_event,
    poll_s,
    frame_stale_s,
    min_pick_s,
    confidence_threshold,
    votes_required,
    sequence_started_at,
):
    next_poll = time.monotonic() + poll_s
    print(f"🧭 [VPEvent] worker started (poll={poll_s}s, min_pick={min_pick_s}s)")

    while not stop_event.is_set() and vp_phase.get() != "place":
        sleep_s = next_poll - time.monotonic()
        if sleep_s > 0 and stop_event.wait(timeout=sleep_s):
            break
        next_poll += poll_s

        elapsed_s = time.monotonic() - sequence_started_at
        if elapsed_s < min_pick_s:
            continue

        top, wrist, frame_age = shared_frames.get()
        if top is None or wrist is None or frame_age > frame_stale_s:
            print(f"⚠️ [VPEvent] frame stale or missing (age={frame_age if frame_age != float('inf') else 'inf'}s) — skip")
            continue

        decision = trigger.decide(top, wrist, elapsed_s)
        latched, vote_count, vote_total = shared_vp_event_trigger.record(decision, confidence_threshold, votes_required)
        print(
            f"🧭 [VPEvent] switch={decision.switch_to_place} "
            f"conf={decision.confidence:.2f} votes={vote_count}/{vote_total} "
            f"latched={latched} latency={decision.latency_ms:.0f}ms "
            f'reason="{decision.reason}"'
        )

        if latched and vp_phase.set_place(reason=decision.reason):
            buffer.clear()
            print("🧭 [VPEvent] phase switched: pick -> place; action buffer cleared")
            break

    print("🧭 [VPEvent] worker stopped.")


# ===================================================================
# Main
# ===================================================================
def resolve_task_spec():
    task_description = wait_for_task_description()
    if not task_description:
        print(
            "🚨 TASK_DESCRIPTION is required. Example:\n"
            '   TASK_DESCRIPTION="pick the orange cube and place it in the yellow basket." python src/main_real2.py\n'
            '   또는 앱 서버 실행 후 USE_USER_COMMAND_TASK_BRIDGE=true 상태에서 앱 텍스트/음성 명령을 먼저 전송하세요.'
        )
        sys.exit(1)

    parsed_target, parsed_location = parse_pick_place_task(task_description)
    if parsed_target is None or parsed_location is None:
        print(
            "🚨 TASK_DESCRIPTION에서 pick/place target 파싱 실패. "
            "지원 형식 예: 'pick the orange cube and place it in the yellow basket.'"
        )
        sys.exit(1)

    target_object = VP_TARGET_OBJECT_OVERRIDE or parsed_target
    target_location = VP_TARGET_LOCATION_OVERRIDE or parsed_location
    if VP_TARGET_OBJECT_OVERRIDE and VP_TARGET_OBJECT_OVERRIDE != parsed_target:
        print(f"⚠️ VP_TARGET_OBJECT env override differs from task: env={VP_TARGET_OBJECT_OVERRIDE!r}, parsed={parsed_target!r}")
    if VP_TARGET_LOCATION_OVERRIDE and VP_TARGET_LOCATION_OVERRIDE != parsed_location:
        print(f"⚠️ VP_TARGET_LOCATION env override differs from task: env={VP_TARGET_LOCATION_OVERRIDE!r}, parsed={parsed_location!r}")

    print(f"🧩 VP task spec: task='{task_description}' | target={target_object!r} | place={target_location!r}")
    return task_description, target_object, target_location


def main():
    from controllers.ik_ctrl import IKController
    from envs.real_env_client import RealRobotEnvClient

    if POLICY_BACKEND == "vp_vla":
        from agents.vp_vla_remote_agent import VPVLARemoteAgent
        agent = VPVLARemoteAgent(
            target_ip=VP_VLA_SERVER_HOST,
            target_port=VP_VLA_SERVER_PORT,
            stats_path=VP_VLA_STATS_PATH,
            unnorm_key=VP_VLA_UNNORM_KEY,
        )
    else:
        from agents.remote_agent import RemoteAgent
        agent = RemoteAgent(target_ip=OCTO_SERVER_HOST, target_port=OCTO_SERVER_PORT)
    print(f"Policy backend: {POLICY_BACKEND}")

    print("🔌 하드웨어 서버(ZMQ)에 연결을 시도합니다...")
    env = RealRobotEnvClient(target_ip="localhost", target_port=5555, urdf_path=URDF_PATH)

    controller = IKController(
        robot=env.shadow_robot,
        ee_link=env.ee_link,
        action_scaling=ACTION_SCALING,
        smoothing_alpha=JOINT_SMOOTHING_ALPHA,
        freeze_wrist_roll=False,
        wrist_roll_index=4,
        gripper_mode=GRIPPER_CONTROL_MODE,
        gripper_open_pos=GRIPPER_BINARY_OPEN_POS,
        gripper_close_min_pos=GRIPPER_BINARY_CLOSE_MIN_POS,
        gripper_close_step_deg=GRIPPER_BINARY_CLOSE_STEP_DEG,
        gripper_open_threshold=GRIPPER_BINARY_OPEN_THRESHOLD,
        gripper_close_motion_scale=GRIPPER_BINARY_CLOSE_MOTION_SCALE,
    )

    print(f"📷 카메라 연결 중: top({CAM_TOP_INDEX}), wrist({CAM_WRIST_INDEX})")
    cam_top, cam_wrist = open_dual_camera(
        use_direct_camera=USE_DIRECT_CAMERA_SOURCE,
        top_index=CAM_TOP_INDEX,
        wrist_index=CAM_WRIST_INDEX,
        top_stream_url=JETSON_TOP_STREAM_URL,
        wrist_stream_url=JETSON_WRIST_STREAM_URL,
    )
    if not cam_top.isOpened() or not cam_wrist.isOpened():
        print("🚨 카메라를 열 수 없습니다. 인덱스를 확인해 주세요.")
        sys.exit(1)

    for name, cam in [("cam_top  ", cam_top), ("cam_wrist", cam_wrist)]:
        w = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cam.get(cv2.CAP_PROP_FPS)
        buf = cam.get(cv2.CAP_PROP_BUFFERSIZE)
        print(f"📷 {name} actual: {w:.0f}x{h:.0f} @ {fps:.0f}fps, buffersize={buf:.0f}")

    shared_app_frame = SharedAppFrame()
    stop_event = threading.Event()
    shared_camera_frames = SharedCameraFrames()
    app_video_thread = None
    if APP_VIDEO_SERVER_ENABLED:
        app_video_thread = start_app_video_server(shared_app_frame, stop_event)

    preview_thread = threading.Thread(
        target=preview_camera_worker,
        args=(cam_top, cam_wrist, shared_camera_frames, shared_app_frame, stop_event),
        daemon=True,
        name="PreviewCameraThread",
    )
    preview_thread.start()

    task_description, target_object, target_location = resolve_task_spec()

    if hasattr(agent, "reset") and not agent.reset():
        print(f"🚨 {POLICY_BACKEND} policy reset 실패. 실행을 중단합니다.")
        sys.exit(1)

    buffer = ChunkBuffer(
        action_dim=ACTION_DIM,
        temporal_ensemble=ACTION_TEMPORAL_ENSEMBLE,
        ensemble_alpha=ACTION_ENSEMBLE_ALPHA,
        max_history=ACTION_ENSEMBLE_MAX_HISTORY,
        weight_dims=ACTION_ENSEMBLE_WEIGHT_DIMS,
    )
    print(
        f"🧮 action temporal ensemble: enabled={ACTION_TEMPORAL_ENSEMBLE} "
        f"alpha={ACTION_ENSEMBLE_ALPHA} history={ACTION_ENSEMBLE_MAX_HISTORY} "
        f"weight_dims={ACTION_ENSEMBLE_WEIGHT_DIMS} execute_steps={EXECUTE_CHUNK_STEPS}"
    )
    shared_state = SharedState()
    shared_frame = SharedFrame()
    vp_phase = SharedVPPhase()

    vp_overlay = None
    if USE_VP_VISUAL_PROMPT:
        try:
            from vp_runtime_overlay import VisualPromptOverlayRuntime
            vp_overlay = VisualPromptOverlayRuntime(
                target_object=target_object,
                target_location=target_location,
                host=VP_SAM3_HOST,
                port=VP_SAM3_PORT,
                timeout_s=VP_SAM3_TIMEOUT_S,
                threshold=VP_SAM3_THRESHOLD,
                mask_threshold=VP_SAM3_MASK_THRESHOLD,
                pick_interval_s=VP_PICK_SAM_INTERVAL_S,
                place_interval_s=VP_PLACE_SAM_INTERVAL_S,
                prefetch_place_on_pick=VP_PREFETCH_PLACE_ON_PICK,
                ready_timeout_s=VP_READY_TIMEOUT_S,
                enabled=True,
            )
            vp_overlay.start()
            print(f"🎯 VP overlay enabled: target={target_object!r}, place={target_location!r}, sam3={VP_SAM3_HOST}:{VP_SAM3_PORT}")
        except Exception as exc:
            print(f"⚠️ VP overlay init failed ({type(exc).__name__}: {exc})")
            vp_overlay = None
    if USE_VP_VISUAL_PROMPT and vp_overlay is None and VP_REQUIRE_OVERLAY:
        print("🚨 VP overlay is required. Start the SAM3 server or set VP_REQUIRE_OVERLAY=false for debugging.")
        sys.exit(1)
    if not USE_VP_VISUAL_PROMPT:
        print("📷 VP overlay disabled by env. App stream will show raw VLA input cameras.")

    init_state = env.get_state()
    if init_state is None:
        print("🚨 초기 state read 실패. 종료.")
        sys.exit(1)
    shared_state.set(np.rad2deg(init_state["q"]).tolist())

    shared_frames = SharedFrames() if USE_VP_EVENT_TRIGGER else None
    shared_vp_event_trigger = None
    vp_event_trigger = None
    vp_event_thread = None
    if USE_VP_EVENT_TRIGGER:
        try:
            from agents.qwen_vp_event_trigger import QwenVPEventTrigger
            vp_event_trigger = QwenVPEventTrigger(
                task_description=task_description,
                target_object=target_object,
                target_location=target_location,
                model_id=QWEN_MODEL_ID,
            )
            shared_vp_event_trigger = SharedVPEventTrigger(VP_EVENT_TRIGGER_VOTE_WINDOW)
            print(
                f"🧭 VP event trigger initialized: poll={VP_EVENT_TRIGGER_POLL_S}s, "
                f"min_pick={VP_EVENT_TRIGGER_MIN_PICK_S}s, "
                f"votes={VP_EVENT_TRIGGER_VOTES_REQUIRED}/{VP_EVENT_TRIGGER_VOTE_WINDOW}, "
                f"conf>={VP_EVENT_TRIGGER_CONFIDENCE}"
            )
        except Exception as exc:
            print(f"⚠️ VP event trigger init failed ({type(exc).__name__}: {exc}) — phase switch unavailable.")
            vp_event_trigger = None
            shared_vp_event_trigger = None
    if USE_VP_EVENT_TRIGGER and vp_event_trigger is None and VP_REQUIRE_EVENT_TRIGGER:
        print("🚨 VP event trigger is required for automatic pick->place phase switching.")
        sys.exit(1)


    infer_thread = threading.Thread(
        target=inference_worker,
        args=(agent, task_description, shared_camera_frames, shared_state, shared_frame, shared_app_frame, shared_frames, vp_phase, vp_overlay, buffer, stop_event),
        daemon=True,
        name="InferenceThread",
    )
    infer_thread.start()

    print(f"🚀 실물 로봇 제어 시작 | 태스크: '{task_description}'")
    print(f"⏳ 첫 chunk 도착 대기 (max {FIRST_CHUNK_TIMEOUT}s)...")
    if not buffer.wait_first(timeout=FIRST_CHUNK_TIMEOUT):
        print("🚨 첫 chunk 도착 timeout. 종료.")
        stop_event.set()
        infer_thread.join(timeout=2.0)
        sys.exit(1)
    print("✅ 첫 chunk 도착. 10Hz control loop 시작.")
    sequence_started_at = time.monotonic()

    if vp_event_trigger is not None and shared_vp_event_trigger is not None:
        vp_event_thread = threading.Thread(
            target=vp_event_trigger_worker,
            args=(
                vp_event_trigger,
                shared_frames,
                shared_vp_event_trigger,
                vp_phase,
                buffer,
                stop_event,
                VP_EVENT_TRIGGER_POLL_S,
                VP_EVENT_TRIGGER_FRAME_STALE_S,
                VP_EVENT_TRIGGER_MIN_PICK_S,
                VP_EVENT_TRIGGER_CONFIDENCE,
                VP_EVENT_TRIGGER_VOTES_REQUIRED,
                sequence_started_at,
            ),
            daemon=True,
            name="VPEventTriggerThread",
        )
        vp_event_thread.start()

    next_tick = time.time()
    prev_loop_start = None
    cycle_dt_window = []
    drift_count = 0
    state_ms_window = []
    ik_ms_window = []
    step_ms_window = []
    other_ms_window = []

    try:
        for i in range(MAX_STEPS):
            loop_start = time.time()
            if prev_loop_start is not None:
                cycle_dt_ms = (loop_start - prev_loop_start) * 1000.0
                cycle_dt_window.append(cycle_dt_ms)
                if len(cycle_dt_window) > HZ_WINDOW_N:
                    cycle_dt_window.pop(0)
            else:
                cycle_dt_ms = float("nan")
            prev_loop_start = loop_start

            t_a = time.time()
            current_state = env.get_state()
            t_b = time.time()
            state_ms = (t_b - t_a) * 1000.0
            if current_state is None:
                next_tick = max(next_tick + CONTROL_PERIOD, time.time())
                time.sleep(max(0.0, next_tick - time.time()))
                continue

            shared_state.set(np.rad2deg(current_state["q"]).tolist())
            raw_action, stale = buffer.pop()
            gripper_tel = current_state.get("gripper_telemetry") or {}

            t_c = time.time()
            q_target, gripper_target = controller.get_joint_targets(raw_action, current_state)
            t_d = time.time()
            ik_ms = (t_d - t_c) * 1000.0

            env.step(q_target, gripper_target)
            t_e = time.time()
            step_ms = (t_e - t_d) * 1000.0

            work_ms = (time.time() - loop_start) * 1000.0
            other_ms = max(0.0, work_ms - state_ms - ik_ms - step_ms)
            for win, val in ((state_ms_window, state_ms), (ik_ms_window, ik_ms), (step_ms_window, step_ms), (other_ms_window, other_ms)):
                win.append(val)
                if len(win) > HZ_WINDOW_N:
                    win.pop(0)

            stale_mark = "⚠️stale" if stale else "      "
            tel_errors = gripper_tel.get("errors") or {}
            tel_error_str = f" gerr={list(tel_errors.keys())}" if tel_errors else ""
            tel_str = (
                f" gcmd={float(gripper_target):5.1f}"
                f" gpos={gripper_tel.get('pos', None)}"
                f" gcur={gripper_tel.get('current', None)}"
                f" gload={gripper_tel.get('load', None)}"
                f"{tel_error_str}"
            )
            if LOG_DETAIL_TIMING:
                print(
                    f"[Ctrl {i:03d}] work={work_ms:5.1f}ms dt={cycle_dt_ms:6.1f}ms {stale_mark} | "
                    f"state={state_ms:5.1f} ik={ik_ms:5.1f} step={step_ms:5.1f} other={other_ms:4.1f} | "
                    f"a[:3]={np.round(raw_action[:3], 3).tolist()} grip={raw_action[6]:+.2f} |{tel_str}"
                )
            else:
                print(
                    f"[Ctrl {i:03d}] work={work_ms:5.1f}ms dt={cycle_dt_ms:6.1f}ms {stale_mark} | "
                    f"a[:3]={np.round(raw_action[:3], 3).tolist()} grip={raw_action[6]:+.2f} |{tel_str}"
                )

            if i > 0 and i % HZ_REPORT_EVERY == 0 and cycle_dt_window:
                mean_dt = float(np.mean(cycle_dt_window))
                max_dt = float(np.max(cycle_dt_window))
                actual_hz = 1000.0 / mean_dt if mean_dt > 0 else 0.0
                print(
                    f"📊 [HzReport @ step {i}] avg_dt={mean_dt:5.1f}ms "
                    f"({actual_hz:4.2f}Hz, target {CONTROL_HZ:.1f}Hz) max_dt={max_dt:5.1f}ms drift={drift_count}/{i} | "
                    f"breakdown: state={np.mean(state_ms_window):5.1f} ik={np.mean(ik_ms_window):5.1f} "
                    f"step={np.mean(step_ms_window):5.1f} other={np.mean(other_ms_window):4.1f}"
                )

            if SHOW_VIZ:
                frame = shared_frame.get()
                if frame is not None:
                    cv2.imshow("Real Robot Control (Top | Wrist)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("1"):
                        if vp_phase.set_pick(reason="manual keyboard"):
                            buffer.clear()
                            print("⌨️ [ManualPhase] phase switched: place -> pick; action buffer cleared")
                    elif key == ord("2"):
                        if vp_phase.set_place(reason="manual keyboard"):
                            buffer.clear()
                            print("⌨️ [ManualPhase] phase switched: pick -> place; action buffer cleared")

            next_tick += CONTROL_PERIOD
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                drift_count += 1
                next_tick = time.time()

    except KeyboardInterrupt:
        print("\n🛑 키보드 입력으로 실험 강제 종료.")
    except Exception as exc:
        print(f"\n🚨 에러 발생: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        if cycle_dt_window:
            mean_dt = float(np.mean(cycle_dt_window))
            max_dt = float(np.max(cycle_dt_window))
            actual_hz = 1000.0 / mean_dt if mean_dt > 0 else 0.0
            print(f"📊 [HzReport FINAL] avg_dt={mean_dt:5.1f}ms ({actual_hz:4.2f}Hz, target {CONTROL_HZ:.1f}Hz) max_dt={max_dt:5.1f}ms drift={drift_count}")
            print(
                f"📊 [Breakdown FINAL] state={np.mean(state_ms_window):5.1f}ms "
                f"ik={np.mean(ik_ms_window):5.1f}ms step={np.mean(step_ms_window):5.1f}ms other={np.mean(other_ms_window):4.1f}ms"
            )

        stop_event.set()
        if vp_event_thread is not None:
            vp_event_thread.join(timeout=15.0)
            if vp_event_thread.is_alive():
                print("⚠️ VP event trigger thread did not exit within 15s")
        infer_thread.join(timeout=2.0)
        if vp_overlay is not None:
            vp_overlay.close()
        if hasattr(agent, "close"):
            agent.close()

        try:
            print("🏠 홈자세 복귀 명령 전송 중...")
            ok = env.send_go_home(timeout_ms=8000)
            print(f"🏠 홈 복귀 결과: {'성공' if ok else '실패'}")
        except Exception as exc:
            print(f"⚠️ 홈 복귀 명령 전송 실패: {type(exc).__name__}: {exc}")

        if hasattr(cam_top, "release"):
            cam_top.release()
        if hasattr(cam_wrist, "release"):
            cam_wrist.release()
        env.disconnect()
        cv2.destroyAllWindows()
        print("🧹 리소스 정리 완료.")


if __name__ == "__main__":
    main()
