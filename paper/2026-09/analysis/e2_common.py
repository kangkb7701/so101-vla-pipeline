"""Shared helpers for the eval2 analysis: banana position (top camera, first frame -> base XY via the
pixel map fitted on the demonstrations), decisive close, basket release points, outcome signals."""
import json
from functools import lru_cache
import cv2, numpy as np
from lerobot.datasets import LeRobotDataset
from common import EVAL, REC, load_training, fk_pos, gripper_events

BASKET = {k: np.array(v) for k, v in json.load(open("/home/aivlab/kkb_capstone/2026-09-paper/analysis/basket_release_xyz.json")).items()}
Z_GRASP = -0.010


def banana_pixel(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array((18, 90, 110)), np.array((35, 255, 255)))
    m[:, 455:] = 0  # basket column
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if st[i, cv2.CC_STAT_AREA] < 2500:
        return None
    ys, xs = np.where(lab == i)
    w, v = np.linalg.eigh(np.cov(np.vstack([xs, ys])))
    return float(cen[i][0]), float(cen[i][1]), float(np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180)


@lru_cache(None)
def pixel_map():
    starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
    P, XY = [], []
    for e, A, S in load_training():
        c, _ = gripper_events(S[:, 5], A[:, 5])
        if c and e in starts:
            P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(fk_pos(A[c[0]:c[0] + 1])[0, :2])
    M, *_ = np.linalg.lstsq(np.array(P), np.array(XY), rcond=None)
    return M


@lru_cache(None)
def _ds(repr_, session):
    return LeRobotDataset(f"local/rollout_smolvla_{repr_}_{session}", root=REC / f"smolvla_{repr_}" / session)


@lru_cache(None)
def banana_info(repr_, ep, session="eval2"):
    ds = _ds(repr_, session)
    it = ds[int(ds.meta.episodes["dataset_from_index"][ep])]
    img = cv2.cvtColor((it["observation.images.camera1"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return banana_pixel(img)


def banana_xy(repr_, ep, session="eval2"):
    bp = banana_info(repr_, ep, session)
    return None if bp is None else np.array([bp[0], bp[1], 1.0]) @ pixel_map()


def held_mask(action, state):
    return (state[:, 5] - action[:, 5] >= 6) & (action[:, 5] < 10)


def decisive_close(action, state):
    """(t_close, t_hold): t_hold = first frame of >= 1 s holding (gap >= 6); t_close = the close starting it, or
    the first close after the start-up period if nothing was held."""
    hm = held_mask(action, state)
    t_hold = next((t for t in range(len(action) - 10) if hm[t:t + 10].all()), None)
    cmd = action[:, 5]
    drops = [k for k in range(1, len(cmd)) if cmd[k] < 8 <= cmd[k - 1]]
    if t_hold is not None:
        before = [k for k in drops if k <= t_hold]
        return (before[-1] if before else t_hold), t_hold
    later = [k for k in drops if k > 10]
    return (later[0] if later else None), None
