#!/usr/bin/env python3
"""ACT policy websocket service - the server half of the split deployment.

The edge laptop owns the robot, cameras, safety and the app; this process only
answers "here is one observation, give me the action chunk". It is stateless
between requests except for policy.reset() on an episode_id change, so it can
be restarted freely and tested with recorded observations.

The laptop dials OUT to this server (hotspot CGNAT blocks inbound to the
laptop), so this side just listens on --port.

Protocol: one msgpack map per binary websocket frame.
  request : {"type": "predict", "episode_id": int, "t_obs": float,
             "task": str, "joints": {"<joint>": deg, ...},
             "images": {"<cam>": <jpeg bytes>, ...}}
  response: {"type": "chunk", "episode_id": int, "t_obs": float, "fps": int,
             "joint_names": [...], "chunk": [[deg]*n_joints]*n_steps,
             "infer_ms": float}
            {"type": "error", "message": str}
  ping    : {"type": "ping"} -> {"type": "pong"}

t_obs is the laptop's own monotonic timestamp, echoed back untouched: the
laptop aligns chunks to its clock, so no clock sync is needed.

--mock serves hold-position chunks with no policy/GPU/dataset, which lets the
whole edge stack (agent, watchdog, app) be tested end-to-end on the laptop.

Server deps beyond the ACT stack: pip install websockets msgpack
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import msgpack
import numpy as np

_lerobot_src = os.environ.get("LEROBOT_SRC")
if _lerobot_src and Path(_lerobot_src).exists() and _lerobot_src not in sys.path:
    sys.path.insert(0, _lerobot_src)

# Must match the dataset the policy was trained on (same as main_act.py).
GOAL_FEATURE = "observation.environment_state"
GOAL_ONE_HOT = {
    "blue": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "yellow": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "green": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}
TASK_GOAL_COLORS = {
    "pick the banana and place it in the green basket": "green",
    "pick the banana and place it in the yellow basket": "yellow",
    "pick the banana and place it in the blue basket": "blue",
}


class MockPolicy:
    """Returns a hold-position chunk. No torch, no checkpoint."""

    def __init__(self, chunk_size: int, fps: int):
        self.chunk_size = chunk_size
        self.fps = fps

    def predict(self, request: dict) -> dict:
        joints = request["joints"]
        names = list(joints.keys())
        row = [float(joints[n]) for n in names]
        return {
            "joint_names": names,
            "chunk": [row] * self.chunk_size,
            "fps": self.fps,
        }

    def reset(self) -> None:
        pass


class ActPolicy:
    """Wraps the exact policy stack main_act.py builds, minus the robot."""

    def __init__(self, args: argparse.Namespace):
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets import LeRobotDataset
        from lerobot.policies import make_policy, make_pre_post_processors
        from lerobot.policies.utils import prepare_observation_for_inference
        from lerobot.processor import rename_stats
        from lerobot.utils.constants import ACTION, OBS_STR
        from lerobot.utils.device_utils import get_safe_torch_device
        from lerobot.utils.feature_utils import build_dataset_frame

        self._torch = torch
        self._prepare = prepare_observation_for_inference
        self._build_frame = build_dataset_frame
        self._OBS_STR = OBS_STR

        self.dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, download_videos=False)
        policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
        policy_cfg.pretrained_path = Path(args.policy_path)
        policy_cfg.device = args.device
        if policy_cfg.temporal_ensemble_coeff is not None:
            # Chunk-mode serving replays whole chunks open-loop on the edge, so
            # per-tick temporal ensembling is impossible by construction. This was
            # decided in the split-deployment design; disable it at load time.
            print(
                f"WARNING: disabling temporal_ensemble_coeff={policy_cfg.temporal_ensemble_coeff} "
                "from the checkpoint (incompatible with chunk-mode serving; chunks replay open-loop)."
            )
            policy_cfg.temporal_ensemble_coeff = None
        self.policy = make_policy(policy_cfg, ds_meta=self.dataset.meta)
        self.policy.eval()
        self.policy.reset()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=args.policy_path,
            dataset_stats=rename_stats(self.dataset.meta.stats, {}),
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        self.device = get_safe_torch_device(args.device)
        self.use_amp = policy_cfg.use_amp
        self.robot_type = args.robot_type
        self.fps = int(self.dataset.meta.fps)
        self.action_names = list(self.dataset.features[ACTION]["names"])
        print(f"policy loaded: {args.policy_path}")
        print(f"dataset: {args.dataset_repo_id} fps={self.fps} actions={self.action_names}")
        if GOAL_FEATURE in self.dataset.features:
            print(f"goal feature active: {GOAL_FEATURE}")

    def _decode_images(self, images: dict) -> dict:
        import cv2

        out = {}
        for name, jpeg in images.items():
            arr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is None:
                raise ValueError(f"could not decode JPEG for camera {name!r}")
            out[name] = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return out

    def predict(self, request: dict) -> dict:
        torch = self._torch
        task = request["task"]

        obs = {f"{name}.pos": float(v) for name, v in request["joints"].items()}
        obs.update(self._decode_images(request["images"]))
        if GOAL_FEATURE in self.dataset.features:
            color = TASK_GOAL_COLORS.get(task)
            if color is None:
                raise ValueError(f"no goal color for task {task!r}")
            goal_names = self.dataset.features[GOAL_FEATURE].get("names") or []
            obs.update(dict(zip(goal_names, GOAL_ONE_HOT[color], strict=True)))

        observation = self._build_frame(self.dataset.features, obs, prefix=self._OBS_STR)
        with (
            torch.inference_mode(),
            torch.autocast(device_type=self.device.type)
            if self.device.type == "cuda" and self.use_amp
            else torch.no_grad(),
        ):
            observation = self._prepare(observation, self.device, task, self.robot_type)
            observation = self.preprocessor(observation)
            raw_chunk = self.policy.predict_action_chunk(observation)
            chunk = self.postprocessor(raw_chunk)

        chunk = chunk.detach().to("cpu").numpy().squeeze(0)  # (n_steps, n_joints)
        return {
            "joint_names": self.action_names,
            "chunk": chunk.astype(float).tolist(),
            "fps": self.fps,
        }

    def reset(self) -> None:
        self.policy.reset()


def serve(args: argparse.Namespace) -> None:
    from websockets.sync.server import serve as ws_serve

    policy = MockPolicy(args.mock_chunk_size, args.mock_fps) if args.mock else ActPolicy(args)
    state = {"episode_id": None}

    def handle(connection):
        peer = connection.remote_address
        print(f"edge connected: {peer}")
        try:
            for raw in connection:
                request = msgpack.unpackb(raw, raw=False)
                msg_type = request.get("type")
                if msg_type == "ping":
                    connection.send(msgpack.packb({"type": "pong"}))
                    continue
                if msg_type != "predict":
                    connection.send(msgpack.packb({"type": "error", "message": f"unknown type {msg_type!r}"}))
                    continue
                try:
                    if request["episode_id"] != state["episode_id"]:
                        state["episode_id"] = request["episode_id"]
                        policy.reset()
                        print(f"episode {state['episode_id']}: task={request.get('task')!r}")
                    start = time.perf_counter()
                    result = policy.predict(request)
                    infer_ms = (time.perf_counter() - start) * 1e3
                    connection.send(
                        msgpack.packb(
                            {
                                "type": "chunk",
                                "episode_id": request["episode_id"],
                                "t_obs": request["t_obs"],
                                "fps": result["fps"],
                                "joint_names": result["joint_names"],
                                "chunk": result["chunk"],
                                "infer_ms": infer_ms,
                            }
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - report to edge, keep serving
                    print(f"predict failed: {type(exc).__name__}: {exc}")
                    connection.send(msgpack.packb({"type": "error", "message": str(exc)}))
        finally:
            print(f"edge disconnected: {peer}")

    print(f"ACT policy server listening on {args.host}:{args.port}" + (" [MOCK]" if args.mock else ""))
    with ws_serve(handle, args.host, args.port, max_size=32 * 1024 * 1024) as server:
        server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("ACT_SERVER_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("ACT_SERVER_PORT", "8765")))
    parser.add_argument("--policy_path", default=os.getenv("ACT_POLICY_PATH"))
    parser.add_argument("--dataset_repo_id", default=os.getenv("ACT_DATASET_REPO_ID"))
    parser.add_argument("--dataset_root", default=os.getenv("ACT_DATASET_ROOT"))
    parser.add_argument("--device", default=os.getenv("ACT_DEVICE", "cuda"))
    parser.add_argument("--robot_type", default=os.getenv("ACT_ROBOT_TYPE", "so_follower"))
    parser.add_argument("--mock", action="store_true", help="Serve hold-position chunks without loading any policy.")
    parser.add_argument("--mock_chunk_size", type=int, default=100)
    parser.add_argument("--mock_fps", type=int, default=10)
    args = parser.parse_args()

    if not args.mock and (not args.policy_path or not args.dataset_repo_id):
        parser.error("--policy_path and --dataset_repo_id are required unless --mock is set")
    serve(args)


if __name__ == "__main__":
    main()
