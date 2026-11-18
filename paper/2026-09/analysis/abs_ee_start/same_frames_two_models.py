"""Same inputs, two models: chunks predicted while the arm has crept 0.5-3 cm from its start (eval2), taken from the
abs_joint and abs_ee sessions, fed to both the abs_joint and the abs_ee model (8 seeds). go = chunk end > 3 cm away."""
import numpy as np, pandas as pd
import replay_lib
from replay_lib import Replayer
from lerobot.datasets import LeRobotDataset
from common import REC, load_session, fk_pos
SRC = {}
for s in ("abs_joint", "abs_ee"):
    pts = []
    for e in load_session(s, "eval2"):
        pm = fk_pos(e.state) * 100; d0 = np.linalg.norm(pm - pm[0], axis=1)
        for t in range(0, len(pm) - 15, 15):
            if d0[t] >= 3: break
            if d0[t] >= 0.5: pts.append((e.ep, t, d0[t]))
    SRC[s] = pts
    print(s, "crept chunks", len(pts), flush=True)
rows = []
for model in ("abs_joint", "abs_ee"):
    replay_lib.SESSION[model] = "eval2"
    R = Replayer(model)
    for s, pts in SRC.items():
        R.ds = LeRobotDataset(f"local/rollout_smolvla_{s}_eval2", root=REC / f"smolvla_{s}" / "eval2")
        R.frm, R.to = R.ds.meta.episodes["dataset_from_index"], R.ds.meta.episodes["dataset_to_index"]; R._cache = {}
        for ep, t, crept in pts:
            obs, _ = R.frames(ep); p0 = fk_pos(obs[t]["observation.state"].astype(float)[None])[0]
            for sd in range(8):
                ch = R.chunk_at(ep, t, seed=sd)
                rows.append({"model": model, "frames_from": s, "ep": ep, "t": t, "crept": crept, "seed": sd,
                             "disp_cm": np.linalg.norm(fk_pos(ch[-1:])[0] - p0) * 100})
        print(model, "on", s, "done", flush=True)
    R.close()
d = pd.DataFrame(rows); d["go"] = d.disp_cm > 3
d.to_csv("abs_ee_start/ae_cross.csv", index=False)
print(d.pivot_table(index="frames_from", columns="model", values="go", aggfunc="mean").round(2))
print(d.pivot_table(index="frames_from", columns="model", values="disp_cm", aggfunc="median").round(2))
