"""Start-up stall as a per-chunk 'leave the rest pose' probability.

The demonstrations idle at the rest pose for about 2.5 s before moving, so an identical rest observation is
labelled both 'stay' and 'go'. While the arm waits nothing in the observation changes, so each new chunk is an
independent draw. p_go = share of 20 noise seeds whose chunk (1.5 s) moves the gripper > 3 cm.
Evaluated on the first frame of every real episode and on frames 0 / 15 of 10 demonstrations (in-distribution).
Expected wait if every chunk is an independent draw: 1.5 s x (1/p_go - 1)."""
import sys
import numpy as np, pandas as pd
from lerobot.datasets import LeRobotDataset
from replay_lib import Replayer, CAMS
from common import REPRS, TRAIN_ROOT, load_all, fk_pos

N = 20
A = load_all()
tr = LeRobotDataset("local/so101_pick_place_fruit_180", root=TRAIN_ROOT)
tfrm = tr.meta.episodes["dataset_from_index"]
rows = []
for r in (sys.argv[1:] or REPRS):
    R = Replayer(r)
    for e in A[r]:
        p0 = fk_pos(e.state[:1])[0]
        go = [np.linalg.norm(fk_pos(R.chunk_at(e.ep, 0, seed=s)[-1:])[0] - p0) * 100 > 3 for s in range(N)]
        rows.append({"repr": r, "src": "real", "ep": e.ep, "t": 0, "p_go": np.mean(go)})
    for ep in range(0, 180, 18):
        obs = []
        for i in range(int(tfrm[ep]), int(tfrm[ep]) + 30):
            it = tr[i]
            o = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
            for k_ds, k in zip(("observation.images.top", "observation.images.wrist"), CAMS):
                o[k] = (it[k_ds].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            obs.append(o)
        R._cache = {-1 - ep: (obs, tr[int(tfrm[ep])]["task"])}
        for t in (0, 15):
            p0 = fk_pos(obs[t]["observation.state"][None].astype(np.float64))[0]
            go = [np.linalg.norm(fk_pos(R.chunk_at(-1 - ep, t, seed=s)[-1:])[0] - p0) * 100 > 3 for s in range(N)]
            rows.append({"repr": r, "src": "train", "ep": ep, "t": t, "p_go": np.mean(go)})
    print(r, flush=True)
    R.close()
df = pd.DataFrame(rows); df.to_csv("escape.csv", index=False)
print(df.groupby(["repr", "src", "t"]).p_go.mean().round(2).unstack(["src", "t"]).reindex(REPRS).to_string())
m = df[df.src == "real"].groupby("repr").p_go.mean()
print("expected wait from real p_go (s):", (1.5 * (1 / m.clip(lower=1 / N) - 1)).reindex(REPRS).round(1).to_dict())
