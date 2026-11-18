# Does a little creeping away from the start make the model start for real? (real eval2 chunks)
import numpy as np, pandas as pd
from common import REPRS, load_session, fk_pos
from e2_common import decisive_close
rows = []
for r in ("abs_joint", "abs_ee"):
    for e in load_session(r, "eval2"):
        pm = fk_pos(e.state) * 100; pc = fk_pos(e.action) * 100
        d0 = np.linalg.norm(pm - pm[0], axis=1)
        for t in range(0, len(pm) - 15, 15):
            if d0[t] >= 3: break
            rows.append({"repr": r, "crept": d0[t], "go": np.linalg.norm(pc[t + 14] - pm[t]) > 3, "lift_cm": pm[t, 2] - pm[0, 2]})
d = pd.DataFrame(rows)
d["bin"] = pd.cut(d.crept, [0, 0.5, 1, 2, 3], include_lowest=True)
print(d.pivot_table(index="bin", columns="repr", values="go", aggfunc=["mean", "count"], observed=False).round(2))
