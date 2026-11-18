"""Start-up at the eval2 rest pose: which input keeps the model from starting?
For eval2 chunks predicted while the gripper is still within 3 cm of its start (t = 0, 3, 6 s), 8 noise seeds:
  base      : recorded eval2 observation
  img_train : both camera images replaced by a demonstration's first frame (state = eval2)
  st_train  : joint state replaced by that demonstration's first-frame state (images = eval2)
  all_train : demonstration first frame (images + state), demonstration task text
go = chunk end moves the gripper > 3 cm from the (possibly replaced) state's FK position."""
import sys
import numpy as np, pandas as pd
import replay_lib
from replay_lib import Replayer
from lerobot.datasets import LeRobotDataset
from common import TRAIN_ROOT, REPRS, load_session, fk_pos

tr = LeRobotDataset("local/so101_pick_place_fruit_180", root=TRAIN_ROOT)
frm = tr.meta.episodes["dataset_from_index"]
def demo_first(k):
    it = tr[int(frm[k])]
    img = {c: (it[s].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
           for s, c in (("observation.images.top", "observation.images.camera1"), ("observation.images.wrist", "observation.images.camera2"))}
    return img, it["observation.state"].numpy().astype(np.float32), it["task"]

rows = []
for r in (sys.argv[1:] or REPRS):
    replay_lib.SESSION[r] = "eval2"
    R = Replayer(r)
    for e in load_session(r, "eval2"):
        img, st, task = demo_first((e.ep * 9 + 4) % 180)
        pm = fk_pos(e.state); d0 = np.linalg.norm(pm - pm[0], axis=1) * 100
        for t in (0, 30, 60):
            if t >= len(e.state) or d0[:t + 15].max() > 3:
                continue
            conds = {"base": (None, None), "img_train": (lambda k, o: {**o, **img}, None),
                     "st_train": (lambda k, o: {**o, "observation.state": st.copy()}, None),
                     "all_train": (lambda k, o: {**o, **img, "observation.state": st.copy()}, task)}
            for cond, (mod, tk) in conds.items():
                p0 = fk_pos((st if cond in ("st_train", "all_train") else e.state[t]).astype(float)[None])[0]
                for sd in range(8):
                    ch = R.chunk_at(e.ep, t, seed=sd, modify=mod, task=tk)
                    rows.append({"repr": r, "ep": e.ep, "t": t / 10, "cond": cond, "seed": sd,
                                 "disp_cm": np.linalg.norm(fk_pos(ch[-1:])[0] - p0) * 100, "grip_max": ch[:, 5].max()})
        print(r, e.ep, flush=True)
    R.close()
    pd.DataFrame(rows).to_csv("abs_ee_start/ae_swap.csv", index=False)
