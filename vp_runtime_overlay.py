from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets.sync.client


VP_UTILITY_DIR = (
    Path(__file__).resolve().parents[1]
    / "VP-VLA"
    / "examples"
    / "Robocasa_tabletop"
    / "visual_prompt_utility"
)
sys.path.insert(0, str(VP_UTILITY_DIR))
from msgpack_utils import packb, unpackb  # noqa: E402


@dataclass(frozen=True)
class Detection:
    mask: np.ndarray | None
    box: list[int] | None
    score: float | None
    centroid: list[float] | None
    latency_s: float
    prompt: str
    fallback: bool = False


class SAM3TextClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.uri = f"ws://{host}:{port}"
        self.ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
            ping_interval=None,
            ping_timeout=None,
        )
        self.metadata = unpackb(self.ws.recv())

    def close(self) -> None:
        self.ws.close()

    def segment(
        self,
        image_rgb: np.ndarray,
        text_prompt: str,
        threshold: float,
        mask_threshold: float,
    ) -> tuple[dict[str, Any], float]:
        request = {
            "type": "segment",
            "request_id": f"seg_{time.time()}",
            "image": image_rgb,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }
        t0 = time.perf_counter()
        self.ws.send(packb(request))
        response = unpackb(self.ws.recv())
        elapsed = time.perf_counter() - t0
        if not response.get("ok", False):
            message = response.get("error", {}).get("message", "unknown SAM3 error")
            raise RuntimeError(f"SAM3 segmentation failed for {text_prompt!r}: {message}")
        return response.get("data", {}), elapsed


def mask_bbox(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid(mask: np.ndarray | None) -> list[float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def top_detection(result: dict[str, Any], latency_s: float, prompt: str) -> Detection:
    masks = np.asarray(result.get("masks", np.zeros((0,), dtype=np.uint8)))
    boxes = np.asarray(result.get("boxes", np.zeros((0, 4), dtype=np.float32)))
    scores = np.asarray(result.get("scores", np.zeros((0,), dtype=np.float32)))
    if masks.ndim < 3 or len(masks) == 0:
        return Detection(None, None, None, None, latency_s, prompt)

    idx = int(np.argmax(scores)) if len(scores) else 0
    mask = masks[idx].astype(bool)
    if boxes.ndim == 2 and len(boxes) > idx:
        box = [int(round(v)) for v in boxes[idx].tolist()]
    else:
        box = mask_bbox(mask)
    score = float(scores[idx]) if len(scores) > idx else None
    return Detection(mask, box, score, mask_centroid(mask), latency_s, prompt)


def draw_crosshair(
    image_rgb: np.ndarray,
    center_xy: list[float],
    line_length: int = 16,
    gap: int = 7,
    thickness: int = 2,
) -> None:
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(image_rgb, (x, y), max(2, thickness + 1), (255, 0, 0), -1)
    cv2.line(image_rgb, (x - gap - line_length, y), (x - gap, y), (0, 255, 0), thickness)
    cv2.line(image_rgb, (x + gap, y), (x + gap + line_length, y), (0, 255, 0), thickness)
    cv2.line(image_rgb, (x, y - gap - line_length), (x, y - gap), (0, 255, 0), thickness)
    cv2.line(image_rgb, (x, y + gap), (x, y + gap + line_length), (0, 255, 0), thickness)


def draw_box(image_rgb: np.ndarray, box: list[int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), thickness)


class SharedVPPhase:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.phase = "pick"
        self.switched_at_monotonic: float | None = None
        self.reason = ""

    def get(self) -> str:
        with self.lock:
            return self.phase

    def set_place(self, reason: str = "") -> bool:
        with self.lock:
            if self.phase == "place":
                return False
            self.phase = "place"
            self.reason = reason
            self.switched_at_monotonic = time.monotonic()
            return True


class VisualPromptOverlayRuntime:
    """Asynchronous SAM3 text-prompt overlay for the real-time inference path.

    The 10 Hz control loop must never wait on SAM3. This runtime accepts the latest
    top RGB frame, runs SAM3 in a background thread at a throttled cadence, and
    renders the latest known crosshair/box onto the frame consumed by the policy.
    """

    def __init__(
        self,
        target_object: str,
        target_location: str,
        host: str = "127.0.0.1",
        port: int = 10094,
        timeout_s: float = 5.0,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        pick_interval_s: float = 0.5,
        place_interval_s: float = 2.0,
        place_detect_once: bool = True,
        prefetch_place_on_pick: bool = True,
        ready_timeout_s: float = 5.0,
        crosshair_line_length: int = 16,
        crosshair_gap: int = 7,
        thickness: int = 2,
        enabled: bool = True,
    ) -> None:
        self.target_object = target_object
        self.target_location = target_location
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.threshold = threshold
        self.mask_threshold = mask_threshold
        self.pick_interval_s = pick_interval_s
        self.place_interval_s = place_interval_s
        self.place_detect_once = place_detect_once
        self.prefetch_place_on_pick = prefetch_place_on_pick
        self.ready_timeout_s = ready_timeout_s
        self.crosshair_line_length = crosshair_line_length
        self.crosshair_gap = crosshair_gap
        self.thickness = thickness
        self.enabled = enabled

        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.latest_rgb: np.ndarray | None = None
        self.latest_phase = "pick"
        self.target_detection: Detection | None = None
        self.place_detection: Detection | None = None
        self.last_pick_s = 0.0
        self.last_place_s = 0.0
        self.error: str | None = None
        self.stop_event = threading.Event()
        self.ready_event = threading.Event()
        self.worker: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        self.worker = threading.Thread(target=self._worker, daemon=True, name="VPOverlayThread")
        self.worker.start()

    def close(self) -> None:
        self.stop_event.set()
        with self.cond:
            self.cond.notify_all()
        if self.worker is not None:
            self.worker.join(timeout=2.0)

    def submit(self, image_rgb: np.ndarray, phase: str) -> None:
        if not self.enabled:
            return
        with self.cond:
            self.latest_rgb = image_rgb.copy()
            self.latest_phase = phase
            self.cond.notify()

    def has_detection(self, phase: str) -> bool:
        with self.lock:
            if phase == "place":
                return self.place_detection is not None and self.place_detection.box is not None
            return self.target_detection is not None and self.target_detection.centroid is not None

    def wait_until_ready(self, phase: str, timeout_s: float | None = None) -> bool:
        if not self.enabled:
            return True
        deadline = time.monotonic() + (self.ready_timeout_s if timeout_s is None else timeout_s)
        while time.monotonic() < deadline:
            if self.has_detection(phase):
                return True
            time.sleep(0.02)
        return self.has_detection(phase)

    def has_dual_detection(self) -> bool:
        with self.lock:
            target_ready = (
                self.target_detection is not None
                and self.target_detection.centroid is not None
            )
            place_ready = (
                self.place_detection is not None
                and self.place_detection.box is not None
            )
        return target_ready and place_ready

    def wait_until_dual_ready(self, timeout_s: float | None = None) -> bool:
        if not self.enabled:
            return True
        deadline = time.monotonic() + (self.ready_timeout_s if timeout_s is None else timeout_s)
        while time.monotonic() < deadline:
            if self.has_dual_detection():
                return True
            time.sleep(0.02)
        return self.has_dual_detection()

    def render(self, image_rgb: np.ndarray, phase: str) -> np.ndarray:
        overlay = image_rgb.copy()
        if not self.enabled:
            return overlay

        with self.lock:
            target = self.target_detection
            place = self.place_detection

        if phase == "place":
            if place is not None and place.box is not None:
                draw_box(overlay, place.box, thickness=self.thickness)
        else:
            if target is not None and target.centroid is not None:
                draw_crosshair(
                    overlay,
                    target.centroid,
                    line_length=self.crosshair_line_length,
                    gap=self.crosshair_gap,
                    thickness=self.thickness,
                )
        return overlay

    def render_both(self, image_rgb: np.ndarray) -> np.ndarray:
        """Render the VP-VLA training contract: crosshair and place box together."""
        overlay = image_rgb.copy()
        if not self.enabled:
            return overlay
        with self.lock:
            target = self.target_detection
            place = self.place_detection
        if target is not None and target.centroid is not None:
            draw_crosshair(
                overlay,
                target.centroid,
                line_length=self.crosshair_line_length,
                gap=self.crosshair_gap,
                thickness=self.thickness,
            )
        if place is not None and place.box is not None:
            draw_box(overlay, place.box, thickness=self.thickness)
        return overlay

    def _worker(self) -> None:
        client = None
        try:
            client = SAM3TextClient(self.host, self.port, self.timeout_s)
            print(f"🎯 [VPOverlay] connected to {client.uri}: {client.metadata}")
            while not self.stop_event.is_set():
                with self.cond:
                    self.cond.wait(timeout=0.1)
                    if self.latest_rgb is None:
                        continue
                    image_rgb = self.latest_rgb.copy()
                    phase = self.latest_phase

                now = time.monotonic()
                jobs: list[tuple[str, str]] = []

                if phase == "pick":
                    if now - self.last_pick_s >= self.pick_interval_s:
                        jobs.append(("pick", self.target_object))
                        self.last_pick_s = now
                    with self.lock:
                        has_place = self.place_detection is not None and self.place_detection.box is not None
                    if (
                        self.prefetch_place_on_pick
                        and not has_place
                        and now - self.last_place_s >= self.place_interval_s
                    ):
                        jobs.append(("place", self.target_location))
                        self.last_place_s = now
                else:
                    with self.lock:
                        has_place = self.place_detection is not None and self.place_detection.box is not None
                    if not (self.place_detect_once and has_place):
                        if now - self.last_place_s >= self.place_interval_s:
                            jobs.append(("place", self.target_location))
                            self.last_place_s = now

                if not jobs:
                    continue

                for job_phase, prompt in jobs:
                    if self.stop_event.is_set():
                        break
                    result, latency = client.segment(
                        image_rgb,
                        prompt,
                        threshold=self.threshold,
                        mask_threshold=self.mask_threshold,
                    )
                    det = top_detection(result, latency, prompt)
                    with self.lock:
                        if job_phase == "place":
                            if det.box is not None:
                                self.place_detection = det
                                self.ready_event.set()
                        else:
                            if det.centroid is not None:
                                self.target_detection = det
                                self.ready_event.set()
                    if det.box is not None or det.centroid is not None:
                        score = "None" if det.score is None else f"{det.score:.3f}"
                        print(
                            f"🎯 [VPOverlay] phase={job_phase} prompt={prompt!r} "
                            f"score={score} latency={latency:.2f}s"
                        )

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            print(f"⚠️ [VPOverlay] disabled after error: {self.error}")
            if os.getenv("VP_OVERLAY_STRICT", "false").lower() in {"1", "true", "yes"}:
                raise
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
