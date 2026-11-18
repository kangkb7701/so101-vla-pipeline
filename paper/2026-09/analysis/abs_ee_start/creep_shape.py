# What do the crept-start frames look like in each session (state only)?
import numpy as np, pandas as pd
from common import load_session, load_training, fk_pos
names = ["pan", "lift", "elbow", "flex", "roll", "grip"]
def crept_rows(eps_states):
    out = []
    for S in eps_states:
        pm = fk_pos(S) * 100; d0 = np.linalg.norm(pm - pm[0], axis=1)
        for t in range(0, len(pm) - 15, 15):
            if d0[t] >= 3: break
            if d0[t] >= 0.5:
                dz = pm[t, 2] - pm[0, 2]; dxy = np.linalg.norm(pm[t, :2] - pm[0, :2])
                out.append({"crept": d0[t], "dz": dz, "dxy": dxy, **{f"d_{n}": S[t, i] - S[0, i] for i, n in enumerate(names)}, "grip": S[t, 5]})
    return pd.DataFrame(out)
tr = load_training()
for name, states in [("demos", [S for _, A, S in tr]),
                     ("abs_joint", [e.state for e in load_session("abs_joint", "eval2")]),
                     ("abs_ee", [e.state for e in load_session("abs_ee", "eval2")])]:
    d = crept_rows(states)
    print(f"{name:9s} n={len(d):3d}", d[["crept", "dz", "dxy", "d_pan", "d_lift", "d_elbow", "d_flex", "d_roll", "grip"]].median().round(2).to_dict())
