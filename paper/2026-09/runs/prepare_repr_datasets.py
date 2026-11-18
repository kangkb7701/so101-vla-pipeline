"""Build the representation-specific datasets and local SmolVLA base folders (2026-09-paper).

Source: so101_pick_place_fruit_180 (180 episodes, absolute joint actions, joint state).
Outputs (next to the source dataset):
  *_rel_joint : same data, action stats = LeRobot chunk-relative stats (action - state, chunk 15, gripper absolute)
  *_abs_ee    : action = [x, y, z, r1..r6, gripper] = FK(action joints) with 6D rotation, gripper unchanged
  *_rel_ee    : same data as *_abs_ee, action stats = chunk-relative to FK(current joint state)
observation.state stays in joint space for every representation.
Local base folders (smolvla_base weights, processor JSON with the relative steps inserted):
  runs/bases/smolvla_base_rel_joint, runs/bases/smolvla_base_rel_ee
Run from 2026-09-paper/runs (the repo root has a local `datasets` package that shadows HF datasets).
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np

from lerobot.datasets import LeRobotDataset
from lerobot.datasets.compute_stats import compute_relative_action_stats, get_feature_stats
from lerobot.datasets.dataset_tools import modify_features
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor.relative_ee_action_processor import (
    SO101_ARM_JOINTS,
    joints_to_ee_reference,
    pose_to_ee_vector,
)

DATA = Path("/home/aivlab/kkb_capstone/data/lerobot")
SRC = DATA / "so101_pick_place_fruit_180"
URDF = "/home/aivlab/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
CHUNK = 15
EE_NAMES = ["x", "y", "z", "r1", "r2", "r3", "r4", "r5", "r6", "gripper"]
BASES = Path(__file__).resolve().parent / "bases"
QUANTILES = [0.01, 0.10, 0.50, 0.90, 0.99]


def to_jsonable(stats):
    return {k: np.asarray(v).tolist() for k, v in stats.items()}


def hardlink_copy(src: Path, dst: Path):
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, copy_function=os.link)
    # stats.json / info.json are rewritten below; break the hard link so the source stays intact.
    for name in ("stats.json", "info.json"):
        p = dst / "meta" / name
        data = p.read_bytes()
        p.unlink()
        p.write_bytes(data)


def write_action_stats(root: Path, action_stats: dict):
    p = root / "meta" / "stats.json"
    stats = json.loads(p.read_text())
    count = stats["action"]["count"]
    stats["action"] = {**to_jsonable(action_stats), "count": count}
    p.write_text(json.dumps(stats, indent=4))


def columns(ds: LeRobotDataset):
    hf = ds.hf_dataset
    return {
        "action": np.array(hf["action"], dtype=np.float32),
        "observation.state": np.array(hf["observation.state"], dtype=np.float32),
        "episode_index": np.array(hf["episode_index"]),
    }


def main():
    src = LeRobotDataset("local/so101_pick_place_fruit_180", root=SRC)
    cols = columns(src)
    assert src.meta.features["action"]["names"][-1] == "gripper.pos"

    # --- relative joint ---------------------------------------------------------------------------
    rel_joint = DATA / "so101_pick_place_fruit_180_rel_joint"
    hardlink_copy(SRC, rel_joint)
    stats = compute_relative_action_stats(cols, src.meta.features, chunk_size=CHUNK, exclude_joints=["gripper"])
    write_action_stats(rel_joint, stats)
    print("rel_joint action mean", np.round(stats["mean"], 2), "std", np.round(stats["std"], 2))

    # --- absolute EE ------------------------------------------------------------------------------
    kin = RobotKinematics(urdf_path=URDF, target_frame_name="gripper_frame_link", joint_names=SO101_ARM_JOINTS)
    ee_action = np.empty((len(cols["action"]), 10), dtype=np.float32)
    for i, row in enumerate(cols["action"]):
        ee_action[i, :9] = pose_to_ee_vector(kin.forward_kinematics(row[:5].astype(np.float64)))
        ee_action[i, 9] = row[5]

    abs_ee = DATA / "so101_pick_place_fruit_180_abs_ee"
    tmp = DATA / "so101_pick_place_fruit_180_tmp_noaction"
    for p in (abs_ee, tmp):
        if p.exists():
            shutil.rmtree(p)
    no_action = modify_features(src, remove_features="action", output_dir=tmp, repo_id="local/tmp_noaction")
    info = {"dtype": "float32", "shape": [10], "names": EE_NAMES}
    modify_features(
        no_action,
        add_features={"action": (ee_action, info)},
        output_dir=abs_ee,
        repo_id="local/so101_pick_place_fruit_180_abs_ee",
    )
    shutil.rmtree(tmp)
    # Stats: copy the source stats (images/state/etc.) and replace the action entry with exact EE stats.
    stats_src = json.loads((SRC / "meta" / "stats.json").read_text())
    stats_src["action"] = {
        **to_jsonable(get_feature_stats(ee_action, axis=0, keepdims=False, quantile_list=QUANTILES)),
        "count": [len(ee_action)],
    }
    (abs_ee / "meta" / "stats.json").write_text(json.dumps(stats_src, indent=4))

    abs_ds = LeRobotDataset("local/so101_pick_place_fruit_180_abs_ee", root=abs_ee)
    got = np.array(abs_ds.hf_dataset["action"], dtype=np.float32)
    assert np.allclose(got, ee_action, atol=1e-6), "EE action column mismatch"
    assert np.allclose(np.array(abs_ds.hf_dataset["observation.state"]), cols["observation.state"])
    print("abs_ee action mean", np.round(stats_src["action"]["mean"], 3))

    # --- relative EE ------------------------------------------------------------------------------
    rel_ee = DATA / "so101_pick_place_fruit_180_rel_ee"
    hardlink_copy(abs_ee, rel_ee)
    ee_ref = joints_to_ee_reference(kin, cols["observation.state"], len(SO101_ARM_JOINTS))
    fake = {"action": ee_action, "observation.state": ee_ref, "episode_index": cols["episode_index"]}
    stats = compute_relative_action_stats(fake, abs_ds.meta.features, chunk_size=CHUNK, exclude_joints=["gripper"])
    write_action_stats(rel_ee, stats)
    print("rel_ee action mean", np.round(stats["mean"], 3), "std", np.round(stats["std"], 3))

    # --- local SmolVLA bases with relative steps in the processor JSON ---------------------------
    base = Path(next((Path.home() / ".cache/huggingface/hub/models--lerobot--smolvla_base/snapshots").iterdir()))
    rel_steps = {
        "rel_joint": {
            "registry_name": "relative_actions_processor",
            "config": {"enabled": False, "exclude_joints": ["gripper"], "action_names": None},
        },
        "rel_ee": {
            "registry_name": "relative_ee_actions_processor",
            "config": {
                "enabled": True,
                "exclude_joints": ["gripper"],
                "action_names": EE_NAMES,
                "urdf_path": URDF,
                "target_frame": "gripper_frame_link",
                "joint_names": SO101_ARM_JOINTS,
            },
        },
    }
    for name, step in rel_steps.items():
        out = BASES / f"smolvla_base_{name}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for f in base.iterdir():
            if f.suffix == ".safetensors" or f.name == "config.json":
                os.symlink(f.resolve(), out / f.name)
        pre = json.loads((base / "policy_preprocessor.json").read_text())
        names = [s["registry_name"] for s in pre["steps"]]
        pre["steps"].insert(names.index("to_batch_processor") + 1, step)
        (out / "policy_preprocessor.json").write_text(json.dumps(pre, indent=2))
        post = json.loads((base / "policy_postprocessor.json").read_text())
        names = [s["registry_name"] for s in post["steps"]]
        enabled = step["config"]["enabled"]
        post["steps"].insert(names.index("unnormalizer_processor") + 1,
                             {"registry_name": "absolute_actions_processor", "config": {"enabled": enabled}})
        (out / "policy_postprocessor.json").write_text(json.dumps(post, indent=2))
        print("base", out, [s["registry_name"] for s in pre["steps"]])


if __name__ == "__main__":
    main()
