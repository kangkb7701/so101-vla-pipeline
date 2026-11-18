"""Are the commanded joints (and their FK positions) inside what the demonstrations covered?
Share of frames outside the training 0.5-99.5 percentile band, per joint and for the gripper-frame height."""
import numpy as np, pandas as pd
from common import REPRS, JOINTS, load_all, load_training, fk_pos

tr = np.concatenate([a for _, a, _ in load_training()])
lo, hi = np.percentile(tr, 0.5, 0), np.percentile(tr, 99.5, 0)
trz = fk_pos(tr[::10])[:, 2]
zlo = np.percentile(trz, 0.5)
print("training band:", {j: (round(l), round(h)) for j, l, h in zip(JOINTS, lo, hi)}, "z min", round(zlo * 100, 1), "cm")
rows = []
for r, eps in load_all().items():
    A = np.concatenate([e.action for e in eps])
    out = (A < lo) | (A > hi)
    z = np.concatenate([fk_pos(e.action)[:, 2] for e in eps])
    rows.append({"repr": r, **{j: round(out[:, i].mean() * 100, 1) for i, j in enumerate(JOINTS)},
                 "any_arm_%": round(out[:, :5].any(1).mean() * 100, 1), "below_table_band_%": round((z < zlo).mean() * 100, 1)})
print(pd.DataFrame(rows).set_index("repr").reindex(REPRS).to_string())
