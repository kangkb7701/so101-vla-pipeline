"""Run a policy checkpoint the same way lerobot-record does, with dataset frames instead of a robot.

Loading mirrors lerobot_record.record(): PreTrainedConfig.from_pretrained -> make_policy(ds_meta)
-> make_pre_post_processors(pretrained_path, rename overrides) -> predict_action -> make_robot_action.
Only robot.get_observation() is replaced by recorded frames (same cameras, same features).
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies import make_policy, make_pre_post_processors
from lerobot.policies.utils import make_robot_action
from lerobot.processor.rename_processor import rename_stats
from lerobot.utils.device_utils import get_safe_torch_device

p = argparse.ArgumentParser()
p.add_argument("--policy", required=True)
p.add_argument("--rename_map", default="{}", help="passed as --dataset.rename_map would be")
p.add_argument(
    "--camera_names",
    default="{}",
    help="simulate naming the robot cameras differently, e.g. top->camera1 (robot config), no rename_map needed",
)
p.add_argument("--root", default="/home/aivlab/kkb_capstone/data/lerobot/so101_pick_place_fruit")
p.add_argument("--episode", type=int, default=0)
p.add_argument("--steps", type=int, default=45)
p.add_argument("--policy_overrides", default="{}", help='e.g. {"n_action_steps": 15}')
args = p.parse_args()
rename_map = json.loads(args.rename_map)
camera_names = json.loads(args.camera_names)

ds = LeRobotDataset("local/so101_pick_place_fruit", root=args.root, episodes=[args.episode])
# Metadata as lerobot-record would build it from robot features (camera keys follow the robot config).
meta = copy.deepcopy(ds.meta)
meta.info["features"] = {camera_names.get(k, k): v for k, v in ds.meta.features.items()}
meta.stats = {camera_names.get(k, k): v for k, v in ds.meta.stats.items()}
task = ds.meta.tasks.index[ds[0]["task_index"].item()] if "task_index" in ds[0] else ds[0]["task"]

cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.pretrained_path = args.policy
cfg.device = "cuda"
for k, v in json.loads(args.policy_overrides).items():
    setattr(cfg, k, v)

policy = make_policy(cfg, ds_meta=meta)
pre, post = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=cfg.pretrained_path,
    dataset_stats=rename_stats(meta.stats, rename_map),
    preprocessor_overrides={
        "device_processor": {"device": cfg.device},
        "rename_observations_processor": {"rename_map": rename_map},
    },
)
policy.reset()
pre.reset()
post.reset()
device = get_safe_torch_device(cfg.device)
img_keys = [k for k in ds.meta.features if k.startswith("observation.images.")]  # dataset keys
names = ds.meta.features["action"]["names"]

print(f"policy={args.policy}\ntask={task!r}  chunk={cfg.chunk_size} n_action_steps={cfg.n_action_steps}")
print("gpu mem after load: %.2f GB" % (torch.cuda.memory_allocated() / 1e9))

errs, times, first = [], [], None
for t in range(min(args.steps, len(ds))):
    item = ds[t]
    # Same layout build_dataset_frame produces from a robot observation: uint8 HWC images, float32 state.
    obs = {"observation.state": item["observation.state"].numpy().astype(np.float32)}
    for k in img_keys:
        obs[camera_names.get(k, k)] = (item[k].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    t0 = time.perf_counter()
    action = predict_action(obs, policy, device, pre, post, cfg.use_amp, task=task, robot_type="so_follower")
    torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)
    robot_action = make_robot_action(action, meta.features)
    if first is None:
        first = robot_action
    pred = np.array([robot_action[n] for n in names])
    errs.append(np.abs(pred - item["action"].numpy()))

errs = np.array(errs)
print("first robot action:", {k: round(float(v), 1) for k, v in first.items()})
print("MAE vs GT per joint (deg):", dict(zip(names, np.round(errs.mean(0), 1).tolist())))
times = np.array(times) * 1000
print("latency ms: max %.0f (chunk inference), median %.1f (queued step)" % (times.max(), np.median(times)))
print("peak gpu mem: %.2f GB" % (torch.cuda.max_memory_allocated() / 1e9))
print("OK")
