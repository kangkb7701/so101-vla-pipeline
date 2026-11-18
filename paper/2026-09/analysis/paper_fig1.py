"""Figure 1 of the paper (eval2): (a) outcomes per representation, (b) realized / commanded motion per chunk vs
commanded chunk travel. Aspect ratio matches the figure box of the HWPX template (19432 x 11588 HWPUNIT)."""
import glob
import numpy as np, pandas as pd
from common import setup_matplotlib, COLOR, INK, INK2, MUTED

plt = setup_matplotlib()
plt.rcParams.update({"font.size": 19, "axes.labelsize": 19, "xtick.labelsize": 17, "ytick.labelsize": 17,
                     "legend.fontsize": 16, "axes.grid": False, "axes.facecolor": "white", "figure.facecolor": "white"})
R = ["abs_joint", "rel_joint", "abs_ee", "rel_ee"]
LAB = {"abs_joint": "절대 관절각", "rel_joint": "상대 관절각", "abs_ee": "절대 EE", "rel_ee": "상대 EE"}
ov = pd.read_csv("e2/overview.csv"); e2 = ov[ov.session == "eval2"]
succ = [(e2[e2.repr == r].label == "success").sum() for r in R]
rim = [(e2[e2.repr == r].label == "rim").sum() for r in R]
held = [e2[e2.repr == r].held.sum() for r in R]
c = pd.concat([pd.read_csv(f) for f in glob.glob("e2/chunks_per_chunk_*.csv")]); c = c[c.chunk > 0]
c["b"] = pd.cut(c.cmd_travel_cm, [0, 0.5, 1, 2, 5, 50])
fig, ax = plt.subplots(1, 2, figsize=(14.4, 8.59), gridspec_kw={"width_ratios": [1, 1.15]})
x = np.arange(4); w = 0.26
ax[0].bar(x - w, held, w, color="#52514e", label="바나나를 잡음", edgecolor="white", linewidth=2)
ax[0].bar(x, succ, w, color="#2a78d6", label="바구니 안에 놓음", edgecolor="white", linewidth=2)
ax[0].bar(x + w, rim, w, color="#9dc3f0", label="바구니 경계에 놓음", edgecolor="white", linewidth=2)
for xi, v in zip(x - w, held):
    ax[0].text(xi, v + 0.15, str(v), ha="center", va="bottom", fontsize=16, color=INK2)
ax[0].set_xticks(x); ax[0].set_xticklabels([LAB[r] for r in R]); ax[0].set_ylabel("횟수 (20회 중)"); ax[0].set_ylim(0, 8)
ax[0].legend(frameon=False, loc="upper right"); ax[0].set_title("(a) 행동 표현별 평가 결과", loc="left", fontsize=19, color=INK)
for r, m in zip(R, ["o", "s", "^", "D"]):
    g = c[c.repr == r].groupby("b", observed=True).realized.median().reindex(c.b.cat.categories)
    ax[1].plot(range(5), g.values.clip(0, None) * 100, marker=m, ms=10, lw=2.5, color=COLOR[r], label=LAB[r])
ax[1].axhline(100, color=MUTED, lw=1, ls="--")
ax[1].set_xticks(range(5)); ax[1].set_xticklabels(["~0.5", "0.5~1", "1~2", "2~5", "5~"])
ax[1].set_xlabel("1.5초 동안 지시한 그리퍼 이동 거리 (cm)"); ax[1].set_ylabel("실제 이동 비율 (%)")
ax[1].set_ylim(0, 115); ax[1].legend(frameon=False, loc="lower right")
ax[1].set_title("(b) 지시한 이동 거리별 실제 이동 비율", loc="left", fontsize=19, color=INK)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig("figs/paper_fig1.png", dpi=167, facecolor="white")
