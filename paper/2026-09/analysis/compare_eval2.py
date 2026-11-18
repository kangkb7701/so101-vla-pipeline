"""Old (eval) vs new (eval2) runs of the joint policies: start pose, timing, gripper, grasp, wrist, basket.
All from recorded signals: commanded/measured joints and the first top-camera frame (banana pixel/orientation).
held = measured-commanded gripper gap >= 6 for >= 1 s (matched the video review on all 54 old episodes);
placed_target = after holding, the first reopen happens nearest to the target basket's demonstrated release point."""
import json
import cv2, numpy as np, pandas as pd
from lerobot.datasets import LeRobotDataset
from common import EVAL, REC, load_session, load_training, fk_pos, gripper_events

BASKET = {k: np.array(v) for k, v in json.load(open("basket_release_xyz.json")).items()}
ax = lambda d: ((d + 90) % 180) - 90


def banana_pixel(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array((18, 90, 110)), np.array((35, 255, 255)))
    m[:, 455:] = 0
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


starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
P, XY = [], []
for e, A, S in load_training():
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c and e in starts:
        P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(fk_pos(A[c[0]:c[0] + 1])[0, :2])
M, *_ = np.linalg.lstsq(np.array(P), np.array(XY), rcond=None)
Z_GRASP = -0.010


def reversals(x, amp=4.0):
    n, last, d = 0, x[0], 0
    for v in x[1:]:
        if d >= 0 and v - last < -amp: n += d > 0; d = -1; last = v
        elif d <= 0 and v - last > amp: n += d < 0; d = 1; last = v
        elif (d > 0 and v > last) or (d < 0 and v < last): last = v
    return n


rows = []
for r in ("abs_joint", "rel_joint"):
    for s in ("eval", "eval2"):
        ds = LeRobotDataset(f"local/rollout_smolvla_{r}_{s}", root=REC / f"smolvla_{r}" / s)
        frm = ds.meta.episodes["dataset_from_index"]
        for e in load_session(r, s):
            img = cv2.cvtColor((ds[int(frm[e.ep])]["observation.images.camera1"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            bp = banana_pixel(img)
            target = e.plan.get("color", "")
            a, st = e.action, e.state
            pos = fk_pos(a); mpos = fk_pos(st)
            d0 = np.linalg.norm(mpos - mpos[0], axis=1) * 100
            held_mask = (st[:, 5] - a[:, 5] >= 6) & (a[:, 5] < 10)
            t_hold = next((t for t in range(len(a) - 10) if held_mask[t:t + 10].all()), None)
            closes, opens = gripper_events(a[:, 5], a[:, 5])
            real_closes = [t for t in closes if t > 10]          # skip flickers at the very start
            tc = None
            if t_hold is not None:
                drops = [k for k in range(max(1, t_hold - 30), t_hold + 1) if a[k, 5] < 8 <= a[k - 1, 5]]
                tc = drops[-1] if drops else t_hold
            elif real_closes:
                tc = real_closes[0]
            row = {"repr": r, "session": s, "ep": e.ep, "trial": e.trial, "orient": e.plan.get("orientation", ""),
                   "start_pan": st[0, 0], "start_roll": st[0, 4], "start_grip": st[0, 5],
                   "len_s": len(a) / 10, "leave_s": (np.argmax(d0 > 5) / 10) if d0.max() > 5 else np.nan,
                   "jitter_per_min": (reversals(a[:, 1]) + reversals(a[:, 2])) / (len(a) / 600),
                   "grip_max_first3s": a[:30, 5].max(), "held": t_hold is not None, "n_closes": len(real_closes)}
            if tc is not None:
                row.update(t_close=tc / 10, z_close_cm=(pos[tc, 2] - Z_GRASP) * 100, open_before=a[max(0, tc - 30):tc, 5].max(),
                           roll_change=a[tc, 4] - a[0, 4])
                if bp:
                    b = np.array([bp[0], bp[1], 1.0]) @ M
                    row.update(aim_cm=np.linalg.norm(pos[tc, :2] - b) * 100, roll_err=abs(ax(a[tc, 4] + bp[2] - 22.0)),
                               banana_angle=bp[2])
            if t_hold is not None:
                rel = [t for t in opens if t > t_hold]
                if rel:
                    p = pos[rel[0]]
                    near = min(BASKET, key=lambda c: np.linalg.norm(p[:2] - BASKET[c][:2]))
                    row.update(release_basket=near, placed_target=near == target)
            rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("compare_eval2.csv", index=False)
pd.set_option("display.width", 250)
cols = ["start_pan", "start_roll", "start_grip", "leave_s", "grip_max_first3s", "open_before", "t_close", "z_close_cm",
        "aim_cm", "roll_err", "roll_change", "jitter_per_min", "len_s"]
print(df.groupby(["repr", "session"])[cols].median().round(1).T.to_string())
g = df.groupby(["repr", "session"])
print(pd.DataFrame({"n": g.size(), "held": g.held.sum(), "placed_target": g.placed_target.apply(lambda x: (x == True).sum())}).to_string())
print(df.groupby(["repr", "session", "orient"]).agg(n=("held", "size"), held=("held", "sum"), roll_err=("roll_err", "median")).round(1).to_string())
