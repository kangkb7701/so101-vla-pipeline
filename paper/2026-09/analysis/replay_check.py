"""Step 1: can the recorded commands be reproduced by replaying the recorded observations?
Full tick-by-tick replay of every episode with 3 noise seeds. Per chunk (15 frames): distance between replay and
recorded command (arm joints RMSE, gripper, FK position at chunk end) next to the seed-to-seed distance. If the
recorded command sits as close to the replays as the replays sit to each other, the replay is faithful and the
counterfactual tests built on it are valid. Saves replays to replay/<repr>_ep<k>.npz."""
import sys, os
import numpy as np, pandas as pd
from replay_lib import Replayer
from common import REPRS, load_all, fk_pos

os.makedirs("replay", exist_ok=True)
A = load_all()
reprs = sys.argv[1:] or REPRS
rows = []
for r in reprs:
    R = Replayer(r)
    for e in A[r]:
        reps = [R.run(e.ep, seed=s) for s in range(3)]
        np.savez(f"replay/{r}_ep{e.ep}.npz", rec=e.action, state=e.state, **{f"seed{s}": x for s, x in enumerate(reps)})
        rec = e.action
        n0 = len(rows)
        T = len(rec)
        for b in range(0, T, 15):
            sl = slice(b, min(b + 15, T)); last = min(b + 14, T - 1)
            def d(x, y):
                return (np.sqrt(((x[sl, :5] - y[sl, :5]) ** 2).mean()), np.abs(x[sl, 5] - y[sl, 5]).mean(),
                        np.linalg.norm(fk_pos(x[last:last + 1])[0] - fk_pos(y[last:last + 1])[0]) * 100)
            rv = np.array([d(x, rec) for x in reps]).mean(0)
            ss = np.array([d(reps[i], reps[j]) for i, j in ((0, 1), (0, 2), (1, 2))]).mean(0)
            rows.append({"repr": r, "ep": e.ep, "chunk": b // 15, "rec_joint_rmse": rv[0], "rec_grip": rv[1], "rec_pos_cm": rv[2],
                         "seed_joint_rmse": ss[0], "seed_grip": ss[1], "seed_pos_cm": ss[2]})
        print(r, e.ep, "first-chunk rec-vs-replay joint RMSE %.2f, seed-vs-seed %.2f" % (rows[n0]["rec_joint_rmse"], rows[n0]["seed_joint_rmse"]), flush=True)
    R.close()
df = pd.DataFrame(rows)
df.to_csv(f"replay_check_{'_'.join(reprs)}.csv", index=False)
print(df.groupby("repr")[["rec_joint_rmse", "seed_joint_rmse", "rec_grip", "seed_grip", "rec_pos_cm", "seed_pos_cm"]].median().round(2).to_string())
print(df[df.chunk == 0].groupby("repr")[["rec_joint_rmse", "seed_joint_rmse", "rec_pos_cm", "seed_pos_cm"]].median().round(2).to_string())
