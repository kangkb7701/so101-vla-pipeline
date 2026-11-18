"""Pre-demo link check: run this on the robot-side host (Jetson) at the venue.

Answers the only question that matters before starting a demo over a hotspot:
is the round trip to the lab GPU server comfortably inside the action-chunk budget?

main_real2.py buffers EXECUTE_CHUNK_STEPS actions and pops one per control tick, so
one inference call covers EXECUTE_CHUNK_STEPS / CONTROL_HZ seconds of motion (16 / 10
= 1.6 s by default). If a round trip exceeds that, ChunkBuffer.pop() starts returning
stale actions: zero deltas with the gripper forced open, i.e. the arm stalls and drops
whatever it was holding. That is the failure mode this script is built to predict.

    VP_VLA_SERVER_HOST=<lab tailnet ip> python check_demo_link.py
    VP_VLA_SERVER_HOST=100.94.154.99 REPS=60 python check_demo_link.py   # longer soak

Sends the real payload shape (three 640x480 RGB frames) through the real client, so
JPEG transport, bandwidth and GPU time are all included in the measured number.
"""
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "VP-VLA"))

HOST = os.getenv("VP_VLA_SERVER_HOST", "127.0.0.1")
PORT = int(os.getenv("VP_VLA_SERVER_PORT", "10093"))
REPS = int(os.getenv("REPS", "30"))
CONTROL_HZ = float(os.getenv("CONTROL_HZ", "10"))
EXECUTE_CHUNK_STEPS = int(os.getenv("EXECUTE_CHUNK_STEPS", "16"))
WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))

BUDGET_MS = EXECUTE_CHUNK_STEPS / CONTROL_HZ * 1000.0
VP_LANGUAGE_PREFIX = (
    "You are given two images: the first is the original robot observation, "
    "and the second has visual prompts overlaid highlighting the target object "
    "and target location. "
)


def probe_images():
    """Real camera frames if available, else noise at the same shape/size.

    Noise is deliberately incompressible, so a synthetic run is a pessimistic
    bandwidth estimate rather than an optimistic one.
    """
    try:
        import cv2

        caps = []
        for idx in (int(os.getenv("CAM_TOP_INDEX", "0")), int(os.getenv("CAM_WRIST_INDEX", "2"))):
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            caps.append(cap)
        got = []
        for cap in caps:
            ok, frame = cap.read()
            cap.release()
            if ok and frame is not None:
                got.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(got) == 2:
            print("probe images: live cameras")
            return [got[0], got[0].copy(), got[1]]
    except Exception:
        pass
    print("probe images: synthetic noise (worst case for compression)")
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8) for _ in range(3)]


def main():
    jpeg_on = os.getenv("VP_VLA_WIRE_JPEG", "").lower() in {"1", "true", "yes"}
    print(f"target      : {HOST}:{PORT}")
    print(f"wire        : {'JPEG q' + os.getenv('VP_VLA_WIRE_JPEG_QUALITY', '90') if jpeg_on else 'RAW (uncompressed!)'}")
    print(f"budget      : {EXECUTE_CHUNK_STEPS} steps / {CONTROL_HZ:g} Hz = {BUDGET_MS:.0f} ms per round trip")
    if not jpeg_on:
        print("             ⚠ VP_VLA_WIRE_JPEG is not set — each request ships ~2.7 MB uncompressed")
    print()

    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

    imgs = probe_images()
    msg = {
        "examples": [{"image": list(imgs), "lang": VP_LANGUAGE_PREFIX + "pick the banana and place it in the yellow basket"}],
        "do_sample": False,
    }

    t0 = time.perf_counter()
    client = WebsocketClientPolicy(host=HOST, port=PORT)
    print(f"connected in {(time.perf_counter() - t0) * 1e3:.0f} ms")

    first = client.predict_action(msg)
    if not first.get("ok"):
        print("server returned an error:", first.get("error"))
        return 1
    horizon = np.asarray(first["data"]["normalized_actions"]).shape
    print(f"action shape: {horizon}  -> {horizon[1]} steps = {horizon[1] / CONTROL_HZ * 1000:.0f} ms of motion\n")

    lat, over = [], 0
    for i in range(REPS):
        t0 = time.perf_counter()
        r = client.predict_action(msg)
        dt = (time.perf_counter() - t0) * 1e3
        if not r.get("ok"):
            print(f"  [{i+1:3d}] ERROR {r.get('error')}")
            continue
        lat.append(dt)
        flag = ""
        if dt > BUDGET_MS:
            over += 1
            flag = "  <-- OVER BUDGET (arm would stall)"
        print(f"  [{i+1:3d}] {dt:7.1f} ms{flag}")
    client.close()

    if not lat:
        print("\nno successful round trips")
        return 1

    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(len(lat_sorted) * q))]
    print("\n" + "=" * 58)
    print(f"round trips  : {len(lat)}   over budget: {over} ({over / len(lat) * 100:.0f}%)")
    print(f"mean / stdev : {statistics.mean(lat):.0f} / {statistics.pstdev(lat):.0f} ms")
    print(f"p50 / p95    : {p(0.50):.0f} / {p(0.95):.0f} ms")
    print(f"min / max    : {min(lat):.0f} / {max(lat):.0f} ms")
    print(f"budget       : {BUDGET_MS:.0f} ms")
    headroom = BUDGET_MS / p(0.95) if p(0.95) else float("inf")
    print(f"headroom@p95 : {headroom:.2f}x")
    print("=" * 58)
    if over == 0 and headroom >= 2.0:
        print("판정: 좋음 — 여유 2배 이상. 시연 진행 가능.")
    elif over == 0:
        print("판정: 빠듯함 — 예산은 지키지만 여유가 적음. 지터가 튀면 끊길 수 있음.")
        print("      EXECUTE_CHUNK_STEPS를 늘리거나 CONTROL_HZ를 낮춰 예산을 키우세요.")
    else:
        print("판정: 위험 — 예산 초과가 발생. 이 상태로 시연하면 팔이 멈추고 그리퍼가 열립니다.")
        print("      VP_VLA_WIRE_JPEG_QUALITY를 낮추거나(80), CONTROL_HZ를 낮추거나,")
        print("      회선을 바꾸세요(다른 통신사 핫스팟 / 유선).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
