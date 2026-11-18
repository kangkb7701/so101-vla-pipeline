"""Where did each policy close the gripper, relative to where the banana actually was?

1. Training data: banana pixel (top camera, episode start) vs FK gripper position at the first grasp (gripper close)
   -> affine map pixel -> robot base XY (least squares), plus the typical grasp height.
2. Evaluation: banana pixel measured in each episode's first top frame -> expected grasp XY; commanded gripper
   position (FK of the commanded joints) at every close event -> miss distance and height above the grasp height.
Only recorded signals are used (commanded/measured joints, camera frames).
"""
import json
import cv2, numpy as np, pandas as pd
from lerobot.datasets import LeRobotDataset
from common import (REC, EVAL, REPRS, LABEL, COLOR, INK2, MUTED, load_all, load_training, fk_pos, gripper_events,
                    setup_matplotlib)


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
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1])) % 180
    return float(cen[i][0]), float(cen[i][1]), float(ang)


# ---- training: pixel -> XY map -------------------------------------------------------------------
starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
P, XY, Z = [], [], []
train_close = []
for e, A, S in load_training():
    closes, _ = gripper_events(S[:, 5], A[:, 5])
    if not closes or e not in starts:
        continue
    pos = fk_pos(A[closes[0]:closes[0] + 1])[0]
    P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(pos[:2]); Z.append(pos[2])
    train_close.append(closes[0])
P, XY, Z = np.array(P), np.array(XY), np.array(Z)
M, *_ = np.linalg.lstsq(P, XY, rcond=None)
res = np.linalg.norm(P @ M - XY, axis=1) * 100
print(f"training grasps {len(P)} | pixel->XY fit residual cm: median {np.median(res):.1f} p90 {np.percentile(res, 90):.1f}")
print(f"training grasp height z (cm): median {np.median(Z)*100:.1f} p5 {np.percentile(Z,5)*100:.1f} p95 {np.percentile(Z,95)*100:.1f}")
print(f"training first close at t = {np.median(train_close)/10:.1f} s (median)")
z_grasp = float(np.median(Z))

# ---- evaluation ------------------------------------------------------------------------------------
vis = pd.read_csv("outcomes_visual.csv")
rows = []
for r, eps in load_all().items():
    s = eps[0].session
    ds = LeRobotDataset(f"local/rollout_smolvla_{r}_{s}", root=REC / f"smolvla_{r}" / s)
    frm = ds.meta.episodes["dataset_from_index"]
    for e in eps:
        it = ds[int(frm[e.ep])]
        bgr = cv2.cvtColor((it["observation.images.camera1"].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bp = banana_pixel(bgr)
        target = (np.array([bp[0], bp[1], 1.0]) @ M) if bp else None
        cmd_pos = fk_pos(e.action)
        closes, opens = gripper_events(e.state[:, 5], e.action[:, 5])
        first = closes[0] if closes else None
        row = {"repr": r, "ep": e.ep, "outcome": vis[(vis.repr == r) & (vis.ep == e.ep)].outcome.iloc[0],
               "banana_px": f"{bp[0]:.0f},{bp[1]:.0f}" if bp else "", "banana_angle": round(bp[2]) if bp else None,
               "n_close": len(closes), "t_first_close_s": first / 10 if first is not None else None}
        if first is not None and target is not None:
            row["miss_cm"] = round(float(np.linalg.norm(cmd_pos[first, :2] - target)) * 100, 1)
            row["height_above_grasp_cm"] = round(float(cmd_pos[first, 2] - z_grasp) * 100, 1)
            # closest approach to the banana (XY) while the gripper was low (within 3 cm of grasp height)
        if target is not None:
            low = cmd_pos[:, 2] < z_grasp + 0.03
            d = np.linalg.norm(cmd_pos[:, :2] - target, axis=1)
            row["closest_low_cm"] = round(float(d[low].min()) * 100, 1) if low.any() else None
            row["min_height_cm"] = round(float(cmd_pos[:, 2].min() - z_grasp) * 100, 1)
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("grasp_analysis.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print()
for r in REPRS:
    d = df[df.repr == r]
    print(f"{r:9s} first-close miss cm median {d.miss_cm.median():5.1f} | height above grasp cm median {d.height_above_grasp_cm.median():5.1f} | "
          f"min height cm median {d.min_height_cm.median():5.1f} | closest-low cm median {d.closest_low_cm.median()} | no close {int((d.n_close==0).sum())}/{len(d)}")

plt = setup_matplotlib()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, r in enumerate(REPRS):
    d = df[df.repr == r]
    ok = d.outcome.str.startswith("success")
    x = np.full(len(d), i) + np.linspace(-0.18, 0.18, len(d))
    ax[0].scatter(x[ok], d.miss_cm[ok], s=40, color=COLOR[r], edgecolor="white", lw=1.2, label="성공" if i == 0 else None)
    ax[0].scatter(x[~ok], d.miss_cm[~ok], s=40, facecolor="white", edgecolor=COLOR[r], lw=1.8, label="실패" if i == 0 else None)
    ax[1].scatter(x[ok], d.height_above_grasp_cm[ok], s=40, color=COLOR[r], edgecolor="white", lw=1.2)
    ax[1].scatter(x[~ok], d.height_above_grasp_cm[~ok], s=40, facecolor="white", edgecolor=COLOR[r], lw=1.8)
for a, t, yl in ((ax[0], "첫 그리퍼 닫힘 위치와 바나나의 수평 거리", "cm"), (ax[1], "첫 그리퍼 닫힘 높이 (학습 데이터 잡기 높이 기준)", "cm")):
    a.set_xticks(range(4)); a.set_xticklabels([LABEL[r] for r in REPRS]); a.set_ylabel(yl)
    a.set_title(t, fontsize=10, color=INK2, loc="left")
ax[1].axhline(0, color=MUTED, lw=1, ls="--")
ax[0].legend(frameon=False, fontsize=8, title="채움=성공, 빈 원=실패", title_fontsize=8)
fig.tight_layout(); fig.savefig("figs/grasp_miss.png", dpi=150)
