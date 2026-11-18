"""`lerobot-rollout` for the 2026-09-paper checkpoints (same CLI and behaviour as `lerobot-rollout`).

Joint checkpoints run unchanged. For end-effector checkpoints (10-d action) the loaded policy pipelines are
patched with IK (eval/ee_ik_steps.py) right after `build_rollout_context`, so the policy hands 6 joint targets
to the standard rollout loop. IK log CSV: $EE_IK_LOG (default: eval/logs/ik_<time>.csv).
IK joint limits: $EE_IK_JOINT_LIMITS = "urdf" (default) or "data" (recorded range, see ee_ik_steps.py).
Chunk log CSV ($CHUNK_LOG, optional, all checkpoints): one row per predicted chunk with the control tick (= recorded
frame index within the episode), wall time, inference time, the raw chunk as the model output it (normalized, in the
checkpoint's own action representation) and the flow-matching noise it was sampled from (exact replay possible).
$EVAL_TRIAL is copied into every row.
Timing log CSV ($TIMING_LOG, optional, diagnostics): per control tick, the inference engine's get_action split into
preprocessor / policy.select_action / postprocessor (CUDA-synchronised), and whether a new chunk was predicted.

Example:
  python eval/eval_rollout.py --strategy.type=base \
    --policy.path=2026-09-paper/runs/smolvla_abs_ee/checkpoints/012893/pretrained_model \
    --robot.type=so101_follower --robot.port=/dev/ttyACM1 --robot.id=my_follower \
    --robot.cameras="{camera1: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
                      camera2: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}}" \
    --task="pick the banana and place it in the blue basket" --duration=60 --fps=10
"""

import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lerobot.scripts.lerobot_rollout as lerobot_rollout  # noqa: E402
from ee_ik_steps import install_ee_ik  # noqa: E402

_build_rollout_context = lerobot_rollout.build_rollout_context


def build_rollout_context_with_ik(cfg, shutdown_event, **kwargs):
    ctx = _build_rollout_context(cfg, shutdown_event, **kwargs)
    log_path = os.environ.get(
        "EE_IK_LOG", str(Path(__file__).resolve().parent / "logs" / f"ik_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    )
    limits = os.environ.get("EE_IK_JOINT_LIMITS", "urdf")
    if install_ee_ik(ctx.policy.preprocessor, ctx.policy.postprocessor, ctx.policy.policy.config, log_path, limits):
        logging.info("EE checkpoint: IK installed (joint limits: %s, log: %s)", limits, log_path)
    else:
        logging.info("Joint checkpoint: running unchanged")
    if os.environ.get("CHUNK_LOG"):
        install_chunk_log(ctx.policy.policy, os.environ["CHUNK_LOG"], os.environ.get("EVAL_TRIAL", ""))
    if os.environ.get("TIMING_LOG"):
        install_timing_log(ctx.policy.inference, os.environ["TIMING_LOG"])
    return ctx


class _Timed:
    """Callable proxy that adds the (CUDA-synchronised) duration of each call to rec[key]."""

    def __init__(self, inner, rec, key):
        self._inner, self._rec, self._key = inner, rec, key

    def __call__(self, *a, **k):
        import torch
        t0 = time.perf_counter()
        out = self._inner(*a, **k)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._rec[self._key] = self._rec.get(self._key, 0.0) + (time.perf_counter() - t0) * 1e3
        return out

    def __getattr__(self, name):
        return getattr(self._inner, name)


def install_timing_log(engine, path):
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["wall_time", "get_action_ms", "pre_ms", "select_ms", "select_gpu_ms", "kernel_ms", "post_ms", "new_chunk"])
    rec = {}
    engine._preprocessor = _Timed(engine._preprocessor, rec, "pre")
    engine._postprocessor = _Timed(engine._postprocessor, rec, "post")
    policy = engine._policy
    select = policy.select_action
    chunks = [0]
    timed_select = _Timed(select, rec, "select")

    def select_(*a, **k):
        import torch
        rec["new_chunk"] = len(policy._queues["action"]) == 0
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        rec["chunks"] = chunks[0] = chunks[0] + rec["new_chunk"]
        # with $TIMING_PROFILE=1, chunks 3-7 are profiled: summed CUDA kernel time (the profiler slows the loop)
        profile = os.environ.get("TIMING_PROFILE") == "1" and rec["new_chunk"] and 3 <= chunks[0] <= 7
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) if profile else None
        if prof:
            prof.__enter__()
        e0.record()
        out = timed_select(*a, **k)  # synchronises the GPU before returning
        e1.record(); e1.synchronize()
        if prof:
            prof.__exit__(None, None, None)
            rec["kernel"] = sum(ev.self_device_time_total for ev in prof.key_averages()) / 1e3
        rec["select_gpu"] = e0.elapsed_time(e1)  # stream time from e0 to e1 (includes GPU idle gaps)
        return out

    policy.select_action = select_
    get_action = engine.get_action

    def get_action_(*a, **k):
        rec.clear()
        t0 = time.perf_counter()
        out = get_action(*a, **k)
        w.writerow([f"{time.time():.3f}", f"{(time.perf_counter() - t0) * 1e3:.1f}", f"{rec.get('pre', 0):.1f}",
                    f"{rec.get('select', 0):.1f}", f"{rec.get('select_gpu', 0):.1f}", f"{rec.get('kernel', float('nan')):.1f}", f"{rec.get('post', 0):.1f}", int(bool(rec.get("new_chunk")))])
        f.flush()
        return out

    engine.get_action = get_action_
    logging.info("Timing log: %s", path)


def install_chunk_log(policy, path, trial):
    """Record every predicted chunk (and its sampling noise) without changing what the policy does."""
    state = {"tick": 0, "noise": None}
    new = not Path(path).exists()
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["trial", "tick", "wall_time", "infer_ms", "chunk_norm", "noise"])

    reset, select, get_chunk, sample_noise = policy.reset, policy.select_action, policy._get_action_chunk, policy.model.sample_noise

    def reset_(*a, **k):
        state["tick"] = 0
        return reset(*a, **k)

    def select_(*a, **k):
        out = select(*a, **k)
        state["tick"] += 1
        return out

    def sample_noise_(*a, **k):
        n = sample_noise(*a, **k)
        state["noise"] = n
        return n

    def get_chunk_(*a, **k):
        t0 = time.perf_counter()
        c = get_chunk(*a, **k)
        ms = (time.perf_counter() - t0) * 1e3
        enc = lambda x: json.dumps([[round(v, 5) for v in row] for row in x[0].float().cpu().tolist()])
        w.writerow([trial, state["tick"], f"{time.time():.3f}", f"{ms:.1f}", enc(c),
                    enc(state["noise"]) if state["noise"] is not None else ""])
        f.flush()
        state["noise"] = None
        return c

    policy.reset, policy.select_action, policy._get_action_chunk, policy.model.sample_noise = reset_, select_, get_chunk_, sample_noise_
    logging.info("Chunk log: %s", path)


lerobot_rollout.build_rollout_context = build_rollout_context_with_ik

if __name__ == "__main__":
    lerobot_rollout.main()
