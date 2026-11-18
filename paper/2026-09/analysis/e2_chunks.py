"""eval2: what the model actually output (chunk log), what was commanded, and what the arm did.

Per episode, from recordings/smolvla_<repr>/eval2.chunks.csv (raw normalized chunk + noise, one row per chunk):
1. Reconstruction: the logged chunks are injected into the same post-processing (unnormalize, relative anchor,
   IK with the recorded joint state as initial guess) by replaying the recorded observations tick by tick;
   reconstructed commands vs recorded commands (should match up to the 1e-5 rounding of the log).
2. Raw output in the model's own representation (unnormalized): per-step magnitude of the predicted action
   (relative: offset from the chunk anchor; EE: position in the base frame).
3. Direction: per chunk, the commanded gripper displacement over the chunk (FK(cmd at chunk end) - FK(measured
   at chunk start)) vs the direction to the banana (before the decisive close) or to the target basket (after a
   hold). Also the measured displacement over the same chunk -> how much of the commanded motion was realized.
Output: e2/chunks_per_chunk.csv, e2/chunks_reconstruction.csv.
"""
import sys, json
import numpy as np, pandas as pd, torch
import replay_lib
from replay_lib import Replayer
from common import REC, EVAL, REPRS, load_session, load_training, fk_pos, gripper_events
from e2_common import banana_xy, decisive_close, BASKET

reprs = sys.argv[1:] or REPRS
rows, rec_rows = [], []
for r in reprs:
    replay_lib.SESSION[r] = "eval2"
    R = Replayer(r)
    unnorm = next(s for s in R.post.steps if "Unnormal" in type(s).__name__)
    st = unnorm._tensor_stats["action"]
    mean, std = st["mean"].float().cpu().numpy(), st["std"].float().cpu().numpy()
    ch_all = pd.read_csv(REC / f"smolvla_{r}" / "eval2.chunks.csv")
    for e in load_session(r, "eval2"):
        rows_t = ch_all[ch_all.trial == e.trial]
        starts = np.where(rows_t.tick.values == 0)[0]
        rows_t = rows_t.iloc[starts[-1]:].reset_index(drop=True)          # last run of a re-recorded trial
        chunks = [np.array(json.loads(c), dtype=np.float32) for c in rows_t.chunk_norm]
        ticks = rows_t.tick.values
        # 1. reconstruction: inject the logged chunks
        it = iter(chunks)
        orig = R.policy._get_action_chunk
        R.policy._get_action_chunk = lambda *a, **k: torch.tensor(next(it), device="cuda")[None]
        try:
            recon = R.run(e.ep)
        finally:
            R.policy._get_action_chunk = orig
        d = np.abs(recon - e.action)
        rec_rows.append({"repr": r, "ep": e.ep, "trial": e.trial, "chunks": len(chunks), "frames": len(e.action),
                         "ticks_ok": bool((ticks == np.arange(len(ticks)) * 15).all()),
                         "max_arm_diff_deg": d[:, :5].max(), "p99_arm_diff_deg": np.percentile(d[:, :5], 99),
                         "max_grip_diff": d[:, 5].max()})
        # 2./3. per chunk
        b = banana_xy(r, e.ep)
        tgt = BASKET[e.plan["color"]][:2]
        tc, t_hold = decisive_close(e.action, e.state)
        pos_cmd, pos_meas = fk_pos(e.action), fk_pos(e.state)
        for k, (t, c) in enumerate(zip(ticks, chunks)):
            if t + 14 >= len(e.action):
                break
            raw = c * std + mean                                             # model's own representation
            cmd_d = pos_cmd[t + 14] - pos_meas[t]
            meas_d = pos_meas[min(t + 16, len(e.action) - 1)] - pos_meas[t]   # +2 frames: tracking lag
            phase = "before_close" if (tc is None or t < tc) else ("holding" if t_hold is not None and t >= t_hold else "after_empty_close")
            goal = b if phase == "before_close" else (tgt if phase == "holding" else None)
            row = {"repr": r, "ep": e.ep, "chunk": k, "t": t / 10, "phase": phase, "infer_ms": rows_t.infer_ms[k],
                   "cmd_travel_cm": np.linalg.norm(cmd_d) * 100, "meas_travel_cm": np.linalg.norm(meas_d) * 100,
                   "realized": float(np.dot(meas_d, cmd_d) / max(np.dot(cmd_d, cmd_d), 1e-9)),
                   "raw_step_norm": float(np.linalg.norm(raw[:, :-1] - (raw[:1, :-1] if r.startswith("abs") else 0), axis=1).mean()),
                   "raw_grip_mean": float(raw[:, -1].mean())}
            if goal is not None and b is not None:
                to_goal = goal - pos_meas[t, :2]
                dist = np.linalg.norm(to_goal)
                row.update(goal_dist_cm=dist * 100,
                           cos_to_goal=float(np.dot(cmd_d[:2], to_goal) / max(np.linalg.norm(cmd_d[:2]) * dist, 1e-9)),
                           progress_cm=float(np.dot(cmd_d[:2], to_goal) / max(dist, 1e-9)) * 100)
            rows.append(row)
        print(r, e.ep, "recon max diff %.4f" % d[:, :5].max(), flush=True)
    R.close()
tag = "_".join(reprs)
pd.DataFrame(rows).to_csv(f"e2/chunks_per_chunk_{tag}.csv", index=False)
pd.DataFrame(rec_rows).to_csv(f"e2/chunks_reconstruction_{tag}.csv", index=False)
