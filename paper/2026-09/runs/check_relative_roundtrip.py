"""Round-trip check of the relative-action processors as lerobot-train builds them (CPU).

preprocessor(raw batch) must turn the action chunk into (action - reference) for non-gripper dims, and
postprocessor(normalized action) must restore the original absolute action chunk.
Run from 2026-09-paper/runs.
"""

import sys

import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_pre_post_processors
from lerobot.policies.factory import make_policy
from lerobot.processor.rename_processor import rename_stats

DATA = "/home/aivlab/kkb_capstone/data/lerobot"
RENAME = {"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}
CASES = {
    "rel_joint": ("bases/smolvla_base_rel_joint", "so101_pick_place_fruit_180_rel_joint", {"use_relative_actions": True}),
    "rel_ee": (
        "bases/smolvla_base_rel_ee",
        "so101_pick_place_fruit_180_rel_ee",
        {"use_relative_ee_actions": True, "ee_urdf_path": "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"},
    ),
}

name = sys.argv[1]
base, ds_name, flags = CASES[name]
cfg = PreTrainedConfig.from_pretrained(base)
cfg.pretrained_path = base
cfg.device = "cpu"
cfg.chunk_size = cfg.n_action_steps = 15
for k, v in flags.items():
    setattr(cfg, k, v)
ds = LeRobotDataset(f"local/{ds_name}", root=f"{DATA}/{ds_name}", delta_timestamps={"action": [i / 10 for i in range(15)]})
policy = make_policy(cfg, ds_meta=ds.meta, rename_map=RENAME)  # fills action_feature_names like training
stats = rename_stats(ds.meta.stats, RENAME)
feats = {**policy.config.input_features, **policy.config.output_features}
pre_over = {
    "device_processor": {"device": "cpu"},
    "normalizer_processor": {"features": feats, "norm_map": policy.config.normalization_mapping, "stats": stats},
    "rename_observations_processor": {"rename_map": RENAME},
}
post_over = {
    "unnormalizer_processor": {
        "features": policy.config.output_features, "norm_map": policy.config.normalization_mapping, "stats": stats,
    }
}
if flags.get("use_relative_actions"):  # same override lerobot_train.py applies
    pre_over["relative_actions_processor"] = {
        "enabled": True, "exclude_joints": policy.config.relative_exclude_joints,
        "action_names": policy.config.action_feature_names,
    }
    post_over["absolute_actions_processor"] = {"enabled": True}
pre, post = make_pre_post_processors(
    policy_cfg=policy.config, pretrained_path=base,
    preprocessor_overrides=pre_over, postprocessor_overrides=post_over,
)
print("pre steps:", [type(s).__name__ for s in pre.steps])
print("post steps:", [type(s).__name__ for s in post.steps])

worst = 0.0
for idx in [0, 500, 12345, 30000]:
    item = ds[idx]
    batch = {k: v for k, v in item.items() if isinstance(v, torch.Tensor)}
    batch["task"] = item["task"]
    raw = batch["action"].clone()
    out = pre(batch)
    restored = post(out["action"]).reshape(raw.shape)  # postprocessor also runs during inference on chunks
    err = (restored - raw).abs().max().item()
    worst = max(worst, err)
    ref = pre.steps[2].get_cached_state()
    print(f"idx {idx}: ref[:3]={np.round(ref.reshape(-1)[:3].numpy(), 4)} raw0[:3]={np.round(raw[0, :3].numpy(), 4)} "
          f"normalized mean|x|={out['action'].abs().mean():.3f} roundtrip max err={err:.2e}")
print("ROUNDTRIP_OK" if worst < 1e-3 else f"ROUNDTRIP_FAIL {worst}")
