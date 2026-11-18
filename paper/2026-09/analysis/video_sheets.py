"""Full-episode video review sheets.

For every evaluation episode:
  overview pages: top | side frame every 1.0 s over the whole episode (18 per page)
  event pages   : every 0.2 s from 1.0 s before to 2.0 s after each gripper close / open event
Each tile is labelled with time, commanded and measured gripper, their gap (about 11 while holding the banana in
the demonstrations, near 0 when the jaws close on nothing) and the commanded gripper height.
Output: vsheets/<repr>/ep<k>_ov<page>.jpg, vsheets/<repr>/ep<k>_ev<n>_<close|open>.jpg (argument "eval2": the
eval2 sessions, into vsheets_eval2/)
"""
import cv2, numpy as np
from lerobot.datasets import LeRobotDataset
import sys
from common import REC, load_all, fk_pos, gripper_events, SESSIONS_OLD, SESSIONS_EVAL2
from pathlib import Path

SESS = SESSIONS_EVAL2 if "eval2" in sys.argv else SESSIONS_OLD
OUTDIR = Path("vsheets_eval2" if "eval2" in sys.argv else "vsheets")

W, H, COLS = 320, 240, 3


def tile(ds, idx, e, t, tag=""):
    it = ds[idx]
    ims = [cv2.cvtColor((it[k].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
           for k in ("observation.images.camera1", "observation.images.camera2")]
    im = np.hstack([cv2.resize(x, (W, H)) for x in ims])
    z = fk_pos(e.action[t:t + 1])[0, 2] * 100
    gc, gm = e.action[t, 5], e.state[t, 5]
    txt = f"{t/10:5.1f}s grip cmd {gc:4.0f} meas {gm:4.0f} gap {gm-gc:4.0f} z {z:4.1f}cm {tag}"
    cv2.rectangle(im, (0, 0), (2 * W, 20), (255, 255, 255), -1)
    cv2.putText(im, txt, (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return im


def page(tiles, header):
    while len(tiles) % COLS:
        tiles.append(np.full_like(tiles[0], 255))
    rows = [np.hstack(tiles[i:i + COLS]) for i in range(0, len(tiles), COLS)]
    img = np.vstack(rows)
    head = np.full((28, img.shape[1], 3), 255, np.uint8)
    cv2.putText(head, header, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    return np.vstack([head, img])


for r, eps in load_all(SESS).items():
    s = eps[0].session
    ds = LeRobotDataset(f"local/rollout_smolvla_{r}_{s}", root=REC / f"smolvla_{r}" / s)
    frm = ds.meta.episodes["dataset_from_index"]
    out = OUTDIR / r; out.mkdir(parents=True, exist_ok=True)
    for e in eps:
        T = len(e.action); f0 = int(frm[e.ep]); p = e.plan
        task = ds[f0]["task"]
        hdr = f"{r} ep{e.ep} trial{e.trial} {p.get('position','')} {p.get('orientation','')} {p.get('distractor','')} | task: {task}"
        ts = list(range(0, T, 10)) + ([T - 1] if (T - 1) % 10 else [])
        for k in range(0, len(ts), 18):
            tl = [tile(ds, f0 + t, e, t) for t in ts[k:k + 18]]
            cv2.imwrite(str(out / f"ep{e.ep:02d}_ov{k//18}.jpg"), page(tl, hdr + f" | overview {k//18}"), [cv2.IMWRITE_JPEG_QUALITY, 85])
        closes, opens = gripper_events(e.state[:, 5], e.action[:, 5])
        evs = sorted([(t, "close") for t in closes] + [(t, "open") for t in opens])
        for n, (te, kind) in enumerate(evs):
            tl = [tile(ds, f0 + t, e, t, f"[{kind}@{te/10:.1f}]") for t in range(max(0, te - 10), min(T, te + 21), 2)]
            cv2.imwrite(str(out / f"ep{e.ep:02d}_ev{n}_{kind}.jpg"), page(tl, hdr + f" | {kind} event at {te/10:.1f}s"), [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(r, e.ep, "events", [(round(t / 10, 1), k) for t, k in evs], flush=True)
