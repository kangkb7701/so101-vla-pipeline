"""Tick-by-tick replay of a recorded evaluation episode through the same policy + processors as on the robot.

The real loop (lerobot-rollout, sync engine, interpolation multiplier 1) did at every recorded frame t:
preprocessor(obs_t) -> policy.select_action (new chunk when the queue is empty: t = 0, 15, 30, ...) ->
postprocessor (relative anchor held for the chunk; EE: IK with the measured joints of frame t as initial
guess) -> recorded action_t. Replaying the recorded observations in the same order therefore reproduces the
commands up to the flow-matching noise drawn at each new chunk. `modify(t, obs)` lets a caller change the
observation (counterfactuals) before it enters the pipeline.
"""
import sys, copy
import numpy as np, torch
sys.path.insert(0, "/home/aivlab/kkb_capstone/2026-09-paper/eval")
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.processor import bind_relative_anchor
from lerobot.policies.utils import prepare_observation_for_inference
from ee_ik_steps import install_ee_ik
from common import PAPER, REC, TRAIN_ROOT

CAMS = ("observation.images.camera1", "observation.images.camera2")
ROOTS = {"abs_joint": TRAIN_ROOT, "rel_joint": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_joint"),
         "abs_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_abs_ee"), "rel_ee": TRAIN_ROOT.with_name(TRAIN_ROOT.name + "_rel_ee")}
SESSION = {"abs_joint": "eval", "rel_joint": "eval", "abs_ee": "eval1", "rel_ee": "eval1"}


class Replayer:
    def __init__(self, repr_):
        self.repr = repr_
        ckpt = PAPER / "runs" / f"smolvla_{repr_}" / "checkpoints" / "012893" / "pretrained_model"
        m = LeRobotDataset(f"local/{ROOTS[repr_].name}", root=ROOTS[repr_]).meta
        meta2 = copy.deepcopy(m)
        meta2.info["features"] = {{"observation.images.top": CAMS[0], "observation.images.wrist": CAMS[1]}.get(k, k): v
                                  for k, v in m.features.items()}
        cfg = PreTrainedConfig.from_pretrained(ckpt); cfg.pretrained_path = str(ckpt); cfg.device = "cuda"
        self.policy = make_policy(cfg, ds_meta=meta2)
        self.pre, self.post = make_pre_post_processors(policy_cfg=self.policy.config, pretrained_path=str(ckpt),
                                                       preprocessor_overrides={"device_processor": {"device": "cuda"}})
        bind_relative_anchor(self.policy, self.pre)
        install_ee_ik(self.pre, self.post, self.policy.config, log_path=None)
        s = SESSION[repr_]
        self.ds = LeRobotDataset(f"local/rollout_smolvla_{repr_}_{s}", root=REC / f"smolvla_{repr_}" / s)
        self.frm = self.ds.meta.episodes["dataset_from_index"]
        self.to = self.ds.meta.episodes["dataset_to_index"]
        self._cache = {}

    def frames(self, ep):
        """Recorded observations of an episode (state + both images as uint8 HWC), cached."""
        if ep not in self._cache:
            out = []
            for i in range(int(self.frm[ep]), int(self.to[ep])):
                it = self.ds[i]
                o = {"observation.state": it["observation.state"].numpy().astype(np.float32)}
                for k in CAMS:
                    o[k] = (it[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                out.append(o)
            self._cache = {ep: (out, self.ds[int(self.frm[ep])]["task"])}
        return self._cache[ep]

    def tick(self, obs, task):
        with torch.inference_mode():
            o = prepare_observation_for_inference(dict(obs), torch.device("cuda"), task, "so_follower")  # it mutates the dict
            a = self.policy.select_action(self.pre(o))
            return self.post(a).reshape(-1).cpu().numpy()[:6].astype(np.float64)

    def run(self, ep, seed=0, t_end=None, modify=None, task=None):
        """Replay frames 0..t_end-1 of `ep`; returns (T, 6) commands."""
        obs_list, task0 = self.frames(ep)
        task = task or task0
        T = len(obs_list) if t_end is None else min(t_end, len(obs_list))
        self.policy.reset(); self.pre.reset(); self.post.reset()
        torch.manual_seed(seed)
        out = []
        for t in range(T):
            obs = obs_list[t]
            if modify is not None:
                obs = modify(t, {k: (v.copy() if hasattr(v, "copy") else v) for k, v in obs.items()})
            out.append(self.tick(obs, task))
        return np.array(out)

    def chunk_at(self, ep, t, seed=0, modify=None, task=None):
        """One fresh chunk predicted from frame t (queue emptied first), executed open-loop on frames t..t+14
        (the anchor / IK guess use those frames' recorded states, as on the robot). Returns (<=15, 6)."""
        obs_list, task0 = self.frames(ep)
        task = task or task0
        self.policy.reset(); self.pre.reset(); self.post.reset()
        torch.manual_seed(seed)
        out = []
        for k in range(t, min(t + 15, len(obs_list))):
            obs = obs_list[k]
            if modify is not None:
                obs = modify(k, {kk: (v.copy() if hasattr(v, "copy") else v) for kk, v in obs.items()})
            out.append(self.tick(obs, task))
        return np.array(out)

    def close(self):
        del self.policy; torch.cuda.empty_cache()
