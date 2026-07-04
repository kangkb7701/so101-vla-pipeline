from __future__ import annotations

import json
import multiprocessing as mp
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import shared_memory

import cv2
import numpy as np


def _video_server_process(shm_name, shape, frame_lock, frame_ready, stop_event, host, port, frame_period_s, jpeg_quality):
    shm = shared_memory.SharedMemory(name=shm_name)
    shared_frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    jpeg_lock = threading.Lock()
    latest = {"jpeg": None}

    def encode_loop():
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        while not stop_event.is_set():
            if not frame_ready.wait(timeout=0.1):
                continue
            with frame_lock:
                frame = shared_frame.copy()
            ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), encode_param)
            if ok:
                with jpeg_lock:
                    latest["jpeg"] = encoded.tobytes()
            time.sleep(frame_period_s)

    encoder = threading.Thread(target=encode_loop, daemon=True)
    encoder.start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            if self.path.startswith("/health"):
                with jpeg_lock:
                    has_frame = latest["jpeg"] is not None
                payload = json.dumps({"ok": True, "has_frame": has_frame}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if not self.path.startswith("/video_feed"):
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while not stop_event.is_set():
                with jpeg_lock:
                    chunk = latest["jpeg"]
                if chunk is None:
                    time.sleep(0.02)
                    continue
                try:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(chunk)).encode() + b"\r\n\r\n" + chunk + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    break
                time.sleep(frame_period_s)

    server = ThreadingHTTPServer((host, port), Handler)
    server.timeout = 0.2
    try:
        while not stop_event.is_set():
            server.handle_request()
    finally:
        server.server_close()
        encoder.join(timeout=1.0)
        shm.close()


class SharedVideoPublisher:
    def __init__(self, shape, *, host, port, frame_period_s, jpeg_quality):
        self.shape = shape
        self._ctx = mp.get_context("spawn")
        self._shm = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)))
        self._frame = np.ndarray(shape, dtype=np.uint8, buffer=self._shm.buf)
        self._lock = self._ctx.Lock()
        self._ready = self._ctx.Event()
        self._stop = self._ctx.Event()
        self._process = self._ctx.Process(target=_video_server_process, args=(self._shm.name, shape, self._lock, self._ready, self._stop, host, port, frame_period_s, jpeg_quality), daemon=True)
        self._process.start()

    def publish(self, frame):
        if frame.shape != self.shape:
            frame = cv2.resize(frame, (self.shape[1], self.shape[0]))
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        with self._lock:
            np.copyto(self._frame, frame)
        self._ready.set()

    def close(self):
        self._stop.set()
        self._process.join(timeout=2.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._shm.close()
        self._shm.unlink()
