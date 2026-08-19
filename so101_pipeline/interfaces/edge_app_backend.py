"""In-process app backend for the edge laptop.

Serves the exact HTTP surface the Flutter app expects on ONE base URL
(the app builds every request from a single ip:port text field):

    GET  /health                     -> {"status": "ok"}
    POST /command/voice  {text}      -> records a new instruction
    POST /command/stop               -> raises the stop flag
    POST /command/success {task}     -> records task success (edge agent calls this in-process)
    GET  /command/latest             -> {instruction, last_command, success, stop}
    GET  /command/history?limit=N    -> {"items": [{kind, text, ts}, ...]}
    GET  /video_feed                 -> MJPEG stream of the combined camera view

The store is plain in-memory state shared with the edge agent running in the
same process. No persistence: a restart starts clean, which is what we want
for a demo box.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Callable, Optional

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


class CommandStore:
    """Thread-safe state shared between the FastAPI thread and the agent loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._instruction: Optional[dict] = None
        self._last_command: Optional[str] = None
        self._success: Optional[dict] = None
        self._stop: dict = {"requested": False, "ts": None}
        self._history: deque = deque(maxlen=200)

    def record_instruction(self, text: str, kind: str = "voice") -> dict:
        entry = {"kind": kind, "text": text, "ts": _now_ts()}
        with self._lock:
            self._instruction = {"text": text, "ts": entry["ts"]}
            self._last_command = text
            self._history.append(entry)
        return entry

    def request_stop(self) -> dict:
        entry = {"kind": "stop", "text": "", "ts": _now_ts()}
        with self._lock:
            self._stop = {"requested": True, "ts": entry["ts"]}
            self._history.append(entry)
        return entry

    def record_success(self, task: str) -> dict:
        entry = {"kind": "success", "text": task, "ts": _now_ts()}
        with self._lock:
            self._success = {"task": task, "ts": entry["ts"]}
            self._history.append(entry)
        return entry

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "instruction": dict(self._instruction) if self._instruction else None,
                "last_command": self._last_command,
                "success": dict(self._success) if self._success else None,
                "stop": dict(self._stop),
            }

    def history_items(self, limit: int) -> list:
        with self._lock:
            items = list(self._history)
        return items[-max(1, limit):]


class VoiceCommand(BaseModel):
    text: str


class SuccessReport(BaseModel):
    task: str


def build_app(store: CommandStore, jpeg_source: Callable[[], Optional[bytes]], mjpeg_period_s: float = 0.033) -> FastAPI:
    """jpeg_source returns the latest combined camera frame as JPEG bytes (or None)."""
    app = FastAPI(title="SO-101 edge app backend")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/command/voice")
    def command_voice(cmd: VoiceCommand):
        entry = store.record_instruction(cmd.text.strip())
        return {"ok": True, "ts": entry["ts"]}

    @app.post("/command/stop")
    def command_stop():
        entry = store.request_stop()
        return {"ok": True, "ts": entry["ts"]}

    @app.post("/command/success")
    def command_success(report: SuccessReport):
        entry = store.record_success(report.task)
        return {"ok": True, "ts": entry["ts"]}

    @app.get("/command/latest")
    def command_latest():
        return store.snapshot()

    @app.get("/command/history")
    def command_history(limit: int = 50):
        return {"items": store.history_items(limit)}

    @app.get("/video_feed")
    def video_feed():
        def gen():
            boundary = b"--frame\r\n"
            while True:
                jpeg = jpeg_source()
                if jpeg is not None:
                    yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
                time.sleep(mjpeg_period_s)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def start_app_server(app: FastAPI, host: str, port: int) -> threading.Thread:
    """Run uvicorn in a daemon thread so the agent loop owns the main thread."""
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="edge-app-backend", daemon=True)
    thread.start()
    return thread
