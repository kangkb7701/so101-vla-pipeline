"""Camera alignment page: live view blended with the training reference frame + basket offsets.

Reads the MJPEG streams of src/scripts/hardware_checks/camera_realtime.py (port 5000, which owns the cameras)
and serves http://<host>:5001 . Red boxes = baskets in the training frame, blue boxes = baskets now.
Numbers per basket: center offset (now - training, px) and size ratio (now / training). Goal: dx, dy ~ 0, size ~ 1.00.
"""

import threading
import time
import urllib.request

import cv2
import numpy as np
from flask import Flask, Response

SRC = "http://localhost:5000"
REF = "/home/aivlab/kkb_capstone/2026-09-paper/reference_frames"
VIEWS = {"top": ("top", cv2.imread(f"{REF}/top.jpg")), "side": ("wrist", cv2.imread(f"{REF}/side.jpg"))}
HSV = {
    "blue": ((95, 120, 60), (130, 255, 255)),
    "yellow": ((18, 80, 120), (35, 255, 255)),
    "green": ((40, 70, 40), (85, 255, 255)),
}
latest: dict[str, np.ndarray] = {}


def baskets(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    out = {}
    for color, (lo, hi) in HSV.items():
        mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, centers = cv2.connectedComponentsWithStats(mask)
        if n < 2:
            continue
        i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[i, cv2.CC_STAT_AREA] >= 3000:
            out[color] = (centers[i], stats[i, :4], stats[i, cv2.CC_STAT_AREA])
    return out


REF_BASKETS = {name: baskets(ref) for name, (_, ref) in VIEWS.items()}


def reader(name, endpoint):
    while True:
        try:
            stream = urllib.request.urlopen(f"{SRC}/{endpoint}", timeout=5)
            buf = b""
            while True:
                buf += stream.read(32768)
                a = buf.find(b"\xff\xd8")
                b = buf.find(b"\xff\xd9", a + 2)
                if a != -1 and b != -1:
                    frame = cv2.imdecode(np.frombuffer(buf[a : b + 2], np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        latest[name] = frame
                    buf = buf[b + 2 :]
        except Exception:
            time.sleep(1)


def render(name, alpha):
    ref = VIEWS[name][1]
    cur = latest.get(name)
    if cur is None:
        return None
    img = cv2.addWeighted(ref, alpha, cur, 1 - alpha, 0)
    now = baskets(cur)
    y = 24
    for color, (rc, (x, yy, w, h), ra) in REF_BASKETS[name].items():
        cv2.rectangle(img, (x, yy), (x + w, yy + h), (0, 0, 255), 2)
        if color in now:
            nc, (x2, y2, w2, h2), na = now[color]
            cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
            text = f"{color}: dx {nc[0] - rc[0]:+.0f} dy {nc[1] - rc[1]:+.0f} size {np.sqrt(na / ra):.2f}"
        else:
            text = f"{color}: not found"
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 24
    return img


app = Flask(__name__)


def stream(name, alpha):
    while True:
        img = render(name, alpha)
        if img is not None:
            ok, jpg = cv2.imencode(".jpg", img)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.15)


@app.route("/<name>/<int:pct>")
def view(name, pct):
    return Response(stream(name, pct / 100), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    return """<html><body style="background:#222;color:#eee;text-align:center;font-family:sans-serif">
<h2>Camera alignment (red = training, blue = now)</h2>
<p>Goal per basket: dx, dy close to 0 and size close to 1.00. Blend: training 50% / now 50%.
 Other blends: <a style="color:#9cf" href="/?a=30">30%</a> <a style="color:#9cf" href="/?a=70">70%</a></p>
<div style="display:flex;justify-content:center;gap:16px">
<div><h3>Top</h3><img id="t" width="640"></div><div><h3>Side</h3><img id="s" width="640"></div></div>
<script>const a=new URLSearchParams(location.search).get('a')||50;
document.getElementById('t').src='/top/'+a;document.getElementById('s').src='/side/'+a;</script>
</body></html>"""


if __name__ == "__main__":
    for name, (endpoint, _) in VIEWS.items():
        threading.Thread(target=reader, args=(name, endpoint), daemon=True).start()
    app.run(host="0.0.0.0", port=5001, threaded=True)
