"""Does the commanded target follow the measured state? (eval2 recordings, robot not used)
For chunks where the commanded gripper travel was < 1 cm (before the decisive close), shoulder_lift and elbow_flex
measured values are shifted by +3 deg in the input (images unchanged), 3 noise seeds. follow = shift of the
commanded gripper position at the first and last step of the chunk / shift of the measured gripper position
(FK of the shifted state). follow ~ 1: the target moves with the measured state (re-anchored, relative-like);
follow ~ 0: the target is kept regardless of the measured state (absolute target).
Output: e2/target_follow.csv"""
import numpy as np, pandas as pd
import replay_lib
from replay_lib import Replayer
from common import REPRS, load_session, fk_pos
from e2_common import decisive_close

D = np.array([0, 3.0, 3.0, 0, 0, 0], dtype=np.float32)
rows = []
for r in REPRS:
    replay_lib.SESSION[r] = "eval2"
    R = Replayer(r)
    picked = 0
    for e in load_session(r, "eval2"):
        tc, _ = decisive_close(e.action, e.state); T = len(e.action) if tc is None else tc
        pc, pm = fk_pos(e.action), fk_pos(e.state)
        ts = [t for t in range(15, T - 15, 15) if np.linalg.norm(pc[t + 14] - pm[t]) * 100 < 1][:3]
        for t in ts:
            meas_shift = fk_pos((e.state[t] + D)[None].astype(float))[0] - pm[t]
            for sd in range(3):
                a = R.chunk_at(e.ep, t, seed=sd)
                b = R.chunk_at(e.ep, t, seed=sd, modify=lambda k, o: {**o, "observation.state": (o["observation.state"] + D).astype(np.float32)})
                fa, fb = fk_pos(a[[0, -1]]), fk_pos(b[[0, -1]])
                proj = lambda v: float(np.dot(v, meas_shift) / np.dot(meas_shift, meas_shift))
                rows.append({"repr": r, "ep": e.ep, "t": t / 10, "seed": sd,
                             "follow_first": proj(fb[0] - fa[0]), "follow_last": proj(fb[1] - fa[1]),
                             "meas_shift_cm": np.linalg.norm(meas_shift) * 100})
            picked += 1
    print(r, "chunks", picked, flush=True)
    R.close()
df = pd.DataFrame(rows); df.to_csv("e2/target_follow.csv", index=False)
print(df.groupby("repr")[["follow_first", "follow_last", "meas_shift_cm"]].median().reindex(REPRS).round(2).to_string())
print(df.groupby("repr").ep.nunique().reindex(REPRS).to_dict())
