# Demonstrations: how long the arm idles at the start, and what the start chunks' labels look like.
import numpy as np, pandas as pd
from common import load_training, fk_pos, REPRS
tr = load_training()
leave, lab0, rest_lab = [], [], []
for ep, A, S in tr:
    pm = fk_pos(S); pa = fk_pos(A)
    moved = np.linalg.norm(pm - pm[0], axis=1) * 100
    l = int(np.argmax(moved >= 3)); leave.append(l / 10)
    lab0.append(np.linalg.norm(pa[14] - pm[0]) * 100)
    for t in range(0, l, 15):                      # label chunks that start while still at rest
        rest_lab.append(np.linalg.norm(pa[min(t + 14, len(pa) - 1)] - pm[t]) * 100)
leave, lab0, rest_lab = map(np.array, (leave, lab0, rest_lab))
print("demos:", len(tr), "| time to move 3cm: median", np.median(leave), "p25/p75", np.percentile(leave, [25, 75]))
print("label chunk at t=0: median %.2f cm, share >3cm %.2f" % (np.median(lab0), (lab0 > 3).mean()))
print("label chunks starting at rest: n=%d median %.2f cm share>3cm %.2f" % (len(rest_lab), np.median(rest_lab), (rest_lab > 3).mean()))
# model predictions on 10 demos (teacher forced, offline_preds_*.npz): first chunk displacement
idx = {ep: (A, S) for ep, A, S in tr}
for r in REPRS:
    z = np.load(f"offline_preds_{r}.npz"); out = []
    for k in z.files:
        if not k.endswith("_pred"): continue
        ep = int(k[2:-5]); A, S = idx[ep]; P = z[k]
        pm = fk_pos(S); pp = fk_pos(P)
        moved = np.linalg.norm(pm - pm[0], axis=1) * 100; l = int(np.argmax(moved >= 3))
        for t in range(0, l, 15):
            out.append(np.linalg.norm(pp[min(t + 14, len(pp) - 1)] - pm[t]) * 100)
    out = np.array(out)
    print(f"{r:9s} model on demo rest chunks: n={len(out)} median {np.median(out):.2f} cm share>3cm {(out > 3).mean():.2f}")
