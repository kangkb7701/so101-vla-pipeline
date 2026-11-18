"""MJPEG camera server for the robot-side host (Jetson).

Serves the top and wrist cameras as separate MJPEG endpoints so main_real2.py can
consume them over the network via application/camera_source.py:

    USE_DIRECT_CAMERA_SOURCE=false CAMERA_HOST=<this host> python main_real2.py

Deliberately stdlib-only (http.server + cv2, no FastAPI/uvicorn) so it runs on old
JetPack images that are stuck on Python 3.6 as well as current ones.

Each camera is grabbed by exactly one thread that keeps only the newest frame, so
several clients (the control loop, a browser, the phone app) can attach without
multiplying camera reads or accumulating latency.

    python jetson_camera_server.py
    curl -s http://localhost:8080/health
"""
import os
import socketserver
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2

CAM_TOP_INDEX = int(os.getenv("CAM_TOP_INDEX", "0"))
CAM_WRIST_INDEX = int(os.getenv("CAM_WRIST_INDEX", "2"))
CAMERA_SERVER_HOST = os.getenv("CAMERA_SERVER_HOST", "0.0.0.0")
CAMERA_SERVER_PORT = int(os.getenv("CAMERA_SERVER_PORT", "8080"))
FRAME_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
CAPTURE_FPS = float(os.getenv("CAMERA_FPS", "30"))
JPEG_QUALITY = int(os.getenv("CAMERA_JPEG_QUALITY", "80"))
STREAM_FPS = float(os.getenv("CAMERA_STREAM_FPS", "30"))
BOUNDARY = "frame"


class CameraWorker(threading.Thread):
    """Owns one cv2.VideoCapture and publishes the most recent JPEG."""

    def __init__(self, name, index):
        super().__init__(daemon=True, name="Camera-%s" % name)
        self.cam_name = name
        self.index = index
        self._lock = threading.Lock()
        self._jpeg = None
        self._seq = 0
        self._frames = 0
        self._failures = 0
        self._started_at = time.time()
        self._stop = threading.Event()

    def _open(self):
        cap = cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAPTURE_FPS)
        # Keep the driver queue at one frame; otherwise the control loop reads
        # frames that are several hundred ms old.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        cap = self._open()
        if not cap.isOpened():
            print("[%s] index %d 열기 실패" % (self.cam_name, self.index))
        while not self._stop.is_set():
            if not cap.isOpened():
                self._failures += 1
                time.sleep(1.0)
                cap.release()
                cap = self._open()
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                self._failures += 1
                time.sleep(0.05)
                # A handful of dropped reads is normal; a sustained run means the
                # device disappeared (USB reset), so reopen it.
                if self._failures % 40 == 0:
                    cap.release()
                    cap = self._open()
                continue
            ok, buf = cv2.imencode(".jpg", frame, encode_param)
            if not ok:
                continue
            with self._lock:
                self._jpeg = buf.tobytes()
                self._seq += 1
                self._frames += 1
        cap.release()

    def stop(self):
        self._stop.set()

    def snapshot(self):
        with self._lock:
            return self._jpeg, self._seq

    def stats(self):
        with self._lock:
            elapsed = max(1e-6, time.time() - self._started_at)
            return {
                "index": self.index,
                "frames": self._frames,
                "failures": self._failures,
                "fps": round(self._frames / elapsed, 2),
                "ready": self._jpeg is not None,
            }


CAMERAS = {}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args):  # quieter than the default access log
        pass

    def _send_json(self, payload, status=200):
        import json

        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path == "/":
            return self._index()
        if path == "/health":
            return self._send_json(
                {"ok": True, "cameras": {n: c.stats() for n, c in CAMERAS.items()}}
            )
        name = path.lstrip("/")
        if name in CAMERAS:
            return self._stream(CAMERAS[name])
        if name.endswith("/snapshot") and name[: -len("/snapshot")] in CAMERAS:
            return self._snapshot(CAMERAS[name[: -len("/snapshot")]])
        self._send_json({"ok": False, "error": "unknown path", "available": sorted(CAMERAS)}, 404)

    def _index(self):
        imgs = "".join(
            "<div style='display:inline-block;margin:8px'><div>%s</div>"
            "<img src='/%s' width='420'></div>" % (n, n)
            for n in sorted(CAMERAS)
        )
        body = (
            "<html><body style='background:#222;color:#eee;font-family:sans-serif;"
            "text-align:center'><h2>Jetson Camera Server</h2>%s"
            "<p>/top /wrist /top/snapshot /health</p></body></html>" % imgs
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _snapshot(self, cam):
        jpeg, _ = cam.snapshot()
        if jpeg is None:
            return self._send_json({"ok": False, "error": "no frame yet"}, 503)
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(jpeg)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(jpeg)

    def _stream(self, cam):
        self.send_response(200)
        self.send_header("Age", "0")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=%s" % BOUNDARY)
        self.end_headers()

        min_period = 1.0 / STREAM_FPS if STREAM_FPS > 0 else 0.0
        last_seq = -1
        try:
            while True:
                jpeg, seq = cam.snapshot()
                if jpeg is None or seq == last_seq:
                    # Nothing new yet — wait instead of resending a duplicate frame.
                    time.sleep(0.005)
                    continue
                last_seq = seq
                self.wfile.write(b"--" + BOUNDARY.encode() + b"\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n")
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
                if min_period:
                    time.sleep(min_period)
        except (BrokenPipeError, ConnectionResetError):
            pass  # client (control loop / browser) went away


class ThreadingHTTPServerCompat(socketserver.ThreadingMixIn, HTTPServer):
    """http.server.ThreadingHTTPServer is 3.7+; JetPack 4.x ships Python 3.6."""

    daemon_threads = True
    allow_reuse_address = True


def main():
    for name, index in (("top", CAM_TOP_INDEX), ("wrist", CAM_WRIST_INDEX)):
        worker = CameraWorker(name, index)
        worker.start()
        CAMERAS[name] = worker

    # Give the capture threads a moment so /health is meaningful right after boot.
    deadline = time.time() + 5.0
    while time.time() < deadline and not all(c.stats()["ready"] for c in CAMERAS.values()):
        time.sleep(0.1)
    for name, cam in sorted(CAMERAS.items()):
        s = cam.stats()
        mark = "OK" if s["ready"] else "NO FRAME"
        print("[%-5s] index=%d %s" % (name, s["index"], mark))

    if not any(c.stats()["ready"] for c in CAMERAS.values()):
        print("카메라를 하나도 열지 못했습니다. `ls /dev/video*` 및 CAM_TOP_INDEX/CAM_WRIST_INDEX 확인.")

    server = ThreadingHTTPServerCompat((CAMERA_SERVER_HOST, CAMERA_SERVER_PORT), Handler)
    print(
        "카메라 서버 시작: http://%s:%d  (/top /wrist /health)  %dx%d q%d"
        % (CAMERA_SERVER_HOST, CAMERA_SERVER_PORT, FRAME_WIDTH, FRAME_HEIGHT, JPEG_QUALITY)
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n종료 중...")
    finally:
        server.shutdown()
        for cam in CAMERAS.values():
            cam.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
