# Wrist roll commands while waiting at the start pose
import numpy as np, pandas as pd
from common import REPRS, load_session, fk_pos
for r in REPRS:
    rows = []
    for e in load_session(r, "eval2"):
        pm = fk_pos(e.state) * 100; d0 = np.linalg.norm(pm - pm[0], axis=1)
        for t in range(0, len(pm) - 15, 15):
            if d0[t] >= 3: break
            rows.append({"roll_now": e.state[t, 4], "roll_cmd_minus_now": e.action[t + 14, 4] - e.state[t, 4],
                         "roll_start": e.state[0, 4], "roll_drift": e.state[t, 4] - e.state[0, 4]})
    d = pd.DataFrame(rows)
    print(f"{r:9s} n={len(d):3d} | |cmd-now| roll median {d.roll_cmd_minus_now.abs().median():5.1f} deg, signed {d.roll_cmd_minus_now.median():+5.1f}"
          f" | roll drift from start median {d.roll_drift.median():+5.1f}, p90 |drift| {d.roll_drift.abs().quantile(.9):5.1f}"
          f" | start roll median {d.roll_start.median():6.1f}")
