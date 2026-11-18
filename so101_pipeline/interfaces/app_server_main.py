import os
import time
import threading
from collections import deque
from datetime import datetime
from typing import Optional
from urllib import request

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ===================================================================
# CONFIG
# ===================================================================
# Source of the MJPEG stream this server re-serves to the app. Defaults to
# main_real2.py's app video server on this host; point VIDEO_SOURCE_HOST at the
# control-loop machine when the two run on different boxes.
VIDEO_SOURCE_HOST = os.getenv("VIDEO_SOURCE_HOST", "127.0.0.1")
VIDEO_SOURCE_PORT = int(os.getenv("VIDEO_SOURCE_PORT", "8010"))
REMOTE_VIDEO_FEED_URL = os.getenv(
    "REMOTE_VIDEO_FEED_URL", f"http://{VIDEO_SOURCE_HOST}:{VIDEO_SOURCE_PORT}/video_feed"
)
STREAM_CHUNK_SIZE = 4096
RECONNECT_SLEEP_S = 0.2
APP_SERVER_HOST = os.getenv("APP_SERVER_HOST", "0.0.0.0")
APP_SERVER_PORT = int(os.getenv("APP_SERVER_PORT", "8000"))


class VoiceCommand(BaseModel):
    text: str


class TouchCommand(BaseModel):
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None


class TaskSuccess(BaseModel):
    task: str


class CommandState:
    def __init__(self, max_history: int = 100):
        self._lock = threading.Lock()
        self._commands = deque(maxlen=max_history)
        self._instruction = {"text": None, "ts": None}
        self._stop = {"requested": False, "ts": None}
        self._success = {"task": None, "ts": None}

    def push_command(self, kind: str, payload: dict):
        entry = {
            "kind": kind,
            "payload": payload,
            "ts": datetime.utcnow().isoformat() + "Z",
        }
        with self._lock:
            self._commands.append(entry)
        print(f"[CMD] {kind}: {payload}")
        return entry

    def set_instruction(self, text: str):
        with self._lock:
            self._instruction = {
                "text": text,
                "ts": datetime.utcnow().isoformat() + "Z",
            }
            self._success = {"task": None, "ts": None}
            return dict(self._instruction)

    def request_stop(self):
        with self._lock:
            self._stop = {"requested": True, "ts": datetime.utcnow().isoformat() + "Z"}
            stop_state = dict(self._stop)
        entry = self.push_command("stop", {})
        return stop_state, entry

    def record_success(self, task: str):
        success = {"task": task, "ts": datetime.utcnow().isoformat() + "Z"}
        with self._lock:
            self._success = success
        entry = self.push_command("success", {"task": task})
        return dict(success), entry

    def latest(self):
        with self._lock:
            last = self._commands[-1] if self._commands else None
            return {
                "instruction": dict(self._instruction),
                "stop": dict(self._stop),
                "success": dict(self._success),
                "last_command": last,
            }

    def history(self, limit: int = 20):
        with self._lock:
            items = list(self._commands)[-limit:]
        return {"count": len(items), "items": items}

    def instruction(self):
        with self._lock:
            return dict(self._instruction)


def gen_proxy_frames(remote_video_feed_url: str = REMOTE_VIDEO_FEED_URL):
    while True:
        try:
            with request.urlopen(remote_video_feed_url, timeout=5.0) as resp:
                while True:
                    chunk = resp.read(STREAM_CHUNK_SIZE)
                    if not chunk:
                        break
                    yield chunk
        except Exception:
            time.sleep(RECONNECT_SLEEP_S)


def create_app(state: CommandState | None = None, remote_video_feed_url: str = REMOTE_VIDEO_FEED_URL):
    state = state or CommandState()
    app = FastAPI(title="VLA Command Server (Video Proxy)")
    app.state.command_state = state
    app.state.remote_video_feed_url = remote_video_feed_url

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get('/')
    def index():
        html = f"""
        <html><body style='background:#222;color:white;font-family:sans-serif;text-align:center;'>
          <h2>VLA Command Server (Video Proxy)</h2>
          <p>Proxy source: {remote_video_feed_url}</p>
          <div><img src='/video_feed' width='480'></div>
          <p>POST /command/voice  |  POST /command/touch  |  GET /command/latest</p>
        </body></html>
        """
        return HTMLResponse(html)

    @app.get('/video_feed')
    def video_feed():
        return StreamingResponse(
            gen_proxy_frames(remote_video_feed_url),
            media_type='multipart/x-mixed-replace; boundary=frame',
        )

    @app.post('/command/voice')
    def command_voice(cmd: VoiceCommand):
        text = cmd.text.strip()
        if not text:
            return JSONResponse({"ok": False, "error": "empty text"}, status_code=400)
        instruction = state.set_instruction(text)
        entry = state.push_command("command", {"text": text})
        return {"ok": True, "entry": entry, "instruction": instruction}

    @app.post('/command/touch')
    def command_touch(cmd: TouchCommand):
        entry = state.push_command("touch", cmd.model_dump())
        return {"ok": True, "entry": entry}

    @app.post('/command/stop')
    def command_stop():
        stop_state, entry = state.request_stop()
        return {"ok": True, "stop": stop_state, "entry": entry}

    @app.post('/command/success')
    def command_success(result: TaskSuccess):
        success, entry = state.record_success(result.task.strip())
        return {"ok": True, "success": success, "entry": entry}

    @app.get('/command/latest')
    def command_latest():
        return state.latest()

    @app.get('/command/history')
    def command_history(limit: int = 20):
        return state.history(limit)

    @app.get('/health')
    def health():
        return {
            "ok": True,
            "video_proxy": remote_video_feed_url,
            "instruction": state.instruction(),
            "stop": state.latest().get("stop"),
        }

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=APP_SERVER_HOST, port=APP_SERVER_PORT)
