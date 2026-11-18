"""Top-down paths of the commanded gripper position (FK of the joints sent to the robot; for EE runs this equals the
model's EE target within ~0.2 mm except the two IK spikes). One small panel per episode: path until the first
close (solid), after it (thin), first close (x), banana position from the top camera via the pixel->XY map (circle).
Also classifies each first close: at the banana (<4 cm and at table height), beside it (table height, >=4 cm), or
in the air (>5 cm above the demonstrated grasp height)."""
import json
import numpy as np, pandas as pd
from common import EVAL, REPRS, LABEL, COLOR, INK2, MUTED, load_all, load_training, fk_pos, gripper_events, setup_matplotlib

starts = {r["ep"]: r for r in json.load(open(EVAL / "train_banana_starts.json"))}
P, XY = [], []
for e, A, S in load_training():
    c, _ = gripper_events(S[:, 5], A[:, 5])
    if c and e in starts:
        P.append([starts[e]["x"], starts[e]["y"], 1.0]); XY.append(fk_pos(A[c[0]:c[0] + 1])[0, :2])
M, *_ = np.linalg.lstsq(np.array(P), np.array(XY), rcond=None)
ga = pd.read_csv("grasp_analysis.csv")
plt = setup_matplotlib()
A = load_all()
rows = []
for r in REPRS:
    eps = A[r]
    n = len(eps); cols = 5; rws = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(2.6 * cols, 2.6 * rws), squeeze=False)
    for ax in axes.flat: ax.set_visible(False)
    for ax, e in zip(axes.flat, eps):
        ax.set_visible(True)
        g = ga[(ga.repr == r) & (ga.ep == e.ep)].iloc[0]
        pos = fk_pos(e.action) * 100
        c, _ = gripper_events(e.state[:, 5], e.action[:, 5])
        tc = c[0] if c else len(pos) - 1
        ax.plot(pos[:tc + 1, 0], pos[:tc + 1, 1], color=COLOR[r], lw=1.6)
        ax.plot(pos[tc:, 0], pos[tc:, 1], color=COLOR[r], lw=0.8, alpha=0.5)
        if isinstance(g.banana_px, str):
            bx, by = map(float, g.banana_px.split(","))
            b = np.array([bx, by, 1.0]) @ M * 100
            ax.scatter(*b, s=90, facecolors="none", edgecolors=INK2, lw=1.5, zorder=3)
        if c:
            ax.scatter(pos[tc, 0], pos[tc, 1], marker="x", color=INK2, s=40, zorder=4)
            kind = "air" if g.height_above_grasp_cm > 5 else ("at_banana" if g.miss_cm < 4 else "beside")
            rows.append({"repr": r, "ep": e.ep, "outcome": g.outcome, "first_close": kind, "miss_cm": g.miss_cm,
                         "height_cm": round(g.height_above_grasp_cm, 1)})
        ax.set_title(f"ep{e.ep} · {g.outcome.replace('fail_', '실패:').replace('success_rim', '성공(걸침)').replace('success', '성공')}", fontsize=8, color=INK2, loc="left")
        ax.set_aspect("equal"); ax.set_xlim(5, 45); ax.set_ylim(-30, 30); ax.tick_params(labelsize=7)
    fig.suptitle(f"{LABEL[r]}: 위에서 본 그리퍼 경로 (cm, 굵은 선=첫 잡기 전, x=첫 잡기, 원=바나나)", fontsize=10, color=INK2, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(f"figs/paths_{r}.png", dpi=130); plt.close(fig)
df = pd.DataFrame(rows); df.to_csv("first_close_kind.csv", index=False)
print(pd.crosstab(df.repr, df.first_close).reindex(REPRS).to_string())
print(pd.crosstab([df.repr, df.first_close], df.outcome.str.startswith("success")).to_string())
