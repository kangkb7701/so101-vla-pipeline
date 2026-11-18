"""Why do the relative policies sit at the rest pose (rel_ee 14-23 s, rel_joint up to 40 s / never)?
Chunks predicted at 3, 6, 9, 12 s while the arm is still at rest (gripper < 5 cm from the start position),
with the recorded state vs gripper state 37 vs the whole demonstration start pose; 3 seeds.
Measure: chunk-end displacement from the current position. Reference: the same measure on demonstration frames
3 s in (offline_preds: the chunk predicted at frame 30 of 10 training episodes, teacher-forced)."""
import sys
import numpy as np, pandas as pd
from replay_lib import Replayer
from common import REPRS, load_all, fk_pos
from counterfactual import set_state, whole_state, TRAIN_START
A = load_all()
rows = []
for r in (sys.argv[1:] or REPRS):
    R = Replayer(r)
    for e in A[r]:
        pos = fk_pos(e.state); d0 = np.linalg.norm(pos - pos[0], axis=1) * 100
        for t in (30, 60, 90, 120):
            if t >= len(e.state) or d0[:t + 1].max() > 5:
                continue
            for cond, mod, sp in (("base", None, e.state[t]), ("grip37", set_state(5, 37.0), np.r_[e.state[t, :5], 37]),
                                  ("trainstart", whole_state(TRAIN_START), TRAIN_START.astype(float))):
                p0 = fk_pos(sp[None])[0]
                for s in range(3):
                    ch = R.chunk_at(e.ep, t, seed=s, modify=mod)
                    rows.append({"repr": r, "ep": e.ep, "t": t / 10, "cond": cond, "seed": s,
                                 "disp_cm": np.linalg.norm(fk_pos(ch[-1:])[0] - p0) * 100, "grip_max": ch[:, 5].max()})
        print(r, e.ep, flush=True)
    R.close()
df = pd.DataFrame(rows); df.to_csv("stall.csv", index=False)
print(df.pivot_table(index="repr", columns="cond", values="disp_cm", aggfunc="median").round(2).to_string())
print(df.groupby("repr").ep.nunique().to_dict())
for r in REPRS:
    z = np.load(f"offline_preds_{r}.npz"); v = []
    for k in z.files:
        if k.endswith("_pred"):
            P, G = z[k], z[k.replace("pred", "gt")]
            v.append((np.linalg.norm(fk_pos(P[44:45])[0] - fk_pos(G[30:31])[0]) * 100, np.linalg.norm(fk_pos(G[44:45])[0] - fk_pos(G[30:31])[0]) * 100))
    print(r, "training frame 30 chunk: pred %.2f cm, demo %.2f cm" % tuple(np.median(v, 0)))
