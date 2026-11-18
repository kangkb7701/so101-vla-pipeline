"""Camera placement per episode vs the training reference frames (basket centers and sizes in the first frame)."""
import cv2, numpy as np, pandas as pd
from lerobot.datasets import LeRobotDataset
from common import REC, PAPER, load_all, setup_matplotlib, COLOR, LABEL, INK2

HSV = {"blue": ((95, 120, 60), (130, 255, 255)), "yellow": ((18, 80, 120), (35, 255, 255)), "green": ((40, 70, 40), (85, 255, 255))}


def baskets(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    out = {}
    for c, (lo, hi) in HSV.items():
        m = cv2.morphologyEx(cv2.inRange(hsv, np.array(lo), np.array(hi)), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(m)
        if n < 2:
            continue
        i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        if st[i, cv2.CC_STAT_AREA] > 3000:
            out[c] = (cen[i], st[i, cv2.CC_STAT_AREA])
    return out


REF = {k: baskets(cv2.imread(str(PAPER / "reference_frames" / f"{k}.jpg"))) for k in ("top", "side")}


def deviation(bgr, view):
    now, ref = baskets(bgr), REF[view]
    shifts, ratios = [], []
    for c in ("blue", "yellow", "green"):
        if c in now and c in ref:
            shifts.append(np.hypot(*(now[c][0] - ref[c][0])))
            ratios.append(np.sqrt(now[c][1] / ref[c][1]))
    if not shifts:
        return np.nan, np.nan, 0
    return float(np.mean(shifts)), float(np.mean(ratios)), len(shifts)


rows = []
for r, eps in load_all().items():
    s = eps[0].session
    ds = LeRobotDataset(f"local/rollout_smolvla_{r}_{s}", root=REC / f"smolvla_{r}" / s)
    frm = ds.meta.episodes["dataset_from_index"]
    for e in eps:
        it = ds[int(frm[e.ep])]
        row = {"repr": r, "ep": e.ep}
        for view, key in (("top", "observation.images.camera1"), ("side", "observation.images.camera2")):
            bgr = cv2.cvtColor((it[key].permute(1, 2, 0).numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            sh, ra, n = deviation(bgr, view)
            row.update({f"{view}_shift_px": round(sh, 1), f"{view}_size": round(ra, 2), f"{view}_baskets": n})
        rows.append(row)
df = pd.DataFrame(rows)
df.to_csv("camera_shift.csv", index=False)
print(df.groupby("repr")[["top_shift_px", "top_size", "side_shift_px", "side_size"]].agg(["median", "min", "max"]).round(1).to_string())
print(df[["repr", "ep", "side_shift_px", "side_size", "side_baskets"]].to_string(index=False))

plt = setup_matplotlib()
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
for r in ["abs_joint", "rel_joint", "abs_ee", "rel_ee"]:
    d = df[df.repr == r]
    ax[0].plot(d.ep, d.top_shift_px, "o-", color=COLOR[r], label=LABEL[r], ms=5, lw=1.5)
    ax[1].plot(d.ep, d.side_shift_px, "o-", color=COLOR[r], label=LABEL[r], ms=5, lw=1.5)
for a, t in zip(ax, ("상단 카메라", "측면 카메라")):
    a.set_title(f"{t}: 학습 기준 화면 대비 바구니 위치 차이", fontsize=10, color=INK2, loc="left")
    a.set_xlabel("에피소드 (녹화 순서)")
    a.set_ylabel("평균 이동 (px)")
ax[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("figs/camera_shift.png", dpi=150)
