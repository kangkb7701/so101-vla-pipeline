"""eval2 pipeline check: replay each recorded episode with the noise logged in eval2.chunks.csv.
If the pipeline on the robot is the same as replay_lib's (= the old lerobot-rollout pipeline), the replayed
commands must equal the recorded commands (up to float rounding of the logged noise, 1e-5)."""
import sys, json
import numpy as np, pandas as pd, torch
from replay_lib import Replayer
import replay_lib
from common import REC, load_session

for r in sys.argv[1:]:
    replay_lib.SESSION[r] = "eval2"
    R = Replayer(r)
    eps = load_session(r, "eval2")
    ch = pd.read_csv(REC / f"smolvla_{r}" / "eval2.chunks.csv")
    res = pd.read_csv(f"/home/aivlab/kkb_capstone/2026-09-paper/eval/eval_plan/results_{r}_eval2.csv")
    for e in eps:
        rows = ch[ch.trial == e.trial]
        # a trial re-recorded (left arrow) leaves earlier rows: keep the last run (ticks restart at 0)
        starts = np.where(rows.tick.values == 0)[0]
        rows = rows.iloc[starts[-1]:]
        noises = [torch.tensor(json.loads(n), device="cuda")[None] for n in rows.noise]
        it = iter(noises)
        R.policy.model.sample_noise = lambda shape, device, it=it: next(it).to(device=device)
        out = R.run(e.ep)
        d = np.abs(out - e.action)
        print(f"{r} ep{e.ep} trial{e.trial}: chunks logged {len(rows)} ticks {rows.tick.tolist()[:4]}... "
              f"max |replay - recorded| arm {d[:, :5].max():.4f} deg, gripper {d[:, 5].max():.4f}", flush=True)
    R.close()
