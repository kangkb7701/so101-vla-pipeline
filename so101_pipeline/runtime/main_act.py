#!/usr/bin/env python3
"""Run a trained LeRobot ACT policy on the SO-101 follower.

This file is intentionally separate from main_real2.py. ACT outputs absolute
robot joint actions in the LeRobot dataset/action space, while main_real2.py
runs an Octo/VLA EE-delta policy through a ZMQ hardware server and IK. Do not
run the hardware server at the same time as this script, because both would try
to own the SO-101 serial port.
"""

from __future__ import annotations

import argparse
import json
import threading
import os
import sys
import time
from pathlib import Path
from urllib import request
from urllib.error import HTTPError, URLError

import cv2
import numpy as np
import torch

# LeRobot을 소스에서 쓸 경우 LEROBOT_SRC 환경변수로 경로 지정 (미설정 시 설치된 pip 패키지 사용)
_lerobot_src = os.environ.get("LEROBOT_SRC")
if _lerobot_src:
    LEROBOT_SRC = Path(_lerobot_src)
    if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
        sys.path.insert(0, str(LEROBOT_SRC))

from so101_pipeline.interfaces.command_bridge import CommandBridgeConfig, UserCommandBridge  # noqa: E402
from so101_pipeline.interfaces.app_video_process import SharedVideoPublisher  # noqa: E402
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.policies.utils import prepare_observation_for_inference  # noqa: E402
from lerobot.configs import PreTrainedConfig  # noqa: E402
from lerobot.datasets import LeRobotDataset  # noqa: E402
from lerobot.policies import make_policy, make_pre_post_processors, make_robot_action  # noqa: E402
from lerobot.processor import make_default_processors, rename_stats  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from lerobot.utils.device_utils import get_safe_torch_device  # noqa: E402
from lerobot.utils.feature_utils import build_dataset_frame  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
HOME_POSITION_DEG = np.asarray([-3.2, -104.8, 105.8, 78.6, 0.3, 2.3], dtype=np.float32)
HOME_MOVE_DURATION_S = 2.5
HOME_MOVE_HZ = 30
ACT_ALLOWED_TASKS = {
    "green": "pick the banana and place it in the green basket",
    "yellow": "pick the banana and place it in the yellow basket",
    "blue": "pick the banana and place it in the blue basket",
}
GOAL_FEATURE = "observation.environment_state"
GOAL_ONE_HOT = {
    "blue": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
    "yellow": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    "green": np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
}
APP_VIDEO_SERVER_ENABLED = os.getenv("APP_VIDEO_SERVER_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
APP_VIDEO_SERVER_HOST = "0.0.0.0"
APP_VIDEO_SERVER_PORT = int(os.getenv("APP_VIDEO_SERVER_PORT", "8010"))
APP_VIDEO_STREAM_SLEEP_S = float(os.getenv("APP_VIDEO_STREAM_SLEEP_S", "0.03"))
APP_VIDEO_JPEG_QUALITY = int(os.getenv("APP_VIDEO_JPEG_QUALITY", "70"))


def _combined_camera_frame(robot):
    frames = []
    for _, cam in robot.cameras.items():
        try:
            frames.append(cam.read_latest())
        except Exception:
            continue
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    min_h = min(frame.shape[0] for frame in frames)
    resized = [
        frame if frame.shape[0] == min_h else cv2.resize(frame, (int(frame.shape[1] * min_h / frame.shape[0]), min_h))
        for frame in frames
    ]
    return np.hstack(resized)


def camera_preview_worker(robot, video_publisher: SharedVideoPublisher, stop_event: threading.Event) -> None:
    print("ACT camera shared-memory publisher started")
    while not stop_event.is_set():
        frame = _combined_camera_frame(robot)
        if frame is not None:
            video_publisher.publish(frame)
        time.sleep(APP_VIDEO_STREAM_SLEEP_S)
    print("ACT camera shared-memory publisher stopped")

def env_bool(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _camera_index_or_path(value: str) -> int | Path:
    return int(value) if value.isdigit() else Path(value)


def build_robot_config(args: argparse.Namespace) -> SOFollowerRobotConfig:
    cameras = {
        args.camera_name: OpenCVCameraConfig(
            index_or_path=_camera_index_or_path(args.camera),
            fps=args.camera_fps,  # 수정됨: 카메라 하드웨어는 camera_fps (30)를 따름
            width=args.width,
            height=args.height,
            fourcc=args.fourcc or None,
        )
    }
    if args.second_camera:
        cameras[args.second_camera_name] = OpenCVCameraConfig(
            index_or_path=_camera_index_or_path(args.second_camera),
            fps=args.camera_fps,  # 수정됨: 카메라 하드웨어는 camera_fps (30)를 따름
            width=args.width,
            height=args.height,
            fourcc=args.fourcc or None,
        )
    return SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.robot_port,
        calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
        disable_torque_on_disconnect=True,
        max_relative_target=args.max_relative_target,
        cameras=cameras,
        use_degrees=True,
    )


def move_home(robot, *, disable_torque: bool) -> None:
    """Move smoothly to the known home pose, optionally releasing motor torque."""
    try:
        current = robot.bus.sync_read("Present_Position")
        current_position = np.asarray([current[name] for name in JOINT_NAMES], dtype=np.float32)
        num_steps = max(1, round(HOME_MOVE_DURATION_S * HOME_MOVE_HZ))

        print(f"Returning to home pose over {HOME_MOVE_DURATION_S:.1f}s...")
        for step in range(1, num_steps + 1):
            alpha = step / num_steps
            target = current_position + alpha * (HOME_POSITION_DEG - current_position)
            robot.send_action(
                {f"{name}.pos": float(target[index]) for index, name in enumerate(JOINT_NAMES)}
            )
            precise_sleep(1.0 / HOME_MOVE_HZ)
        print(f"Home pose reached: {HOME_POSITION_DEG.tolist()}")
    except Exception as exc:
        print(f"WARNING: Failed to return home: {exc}")
    finally:
        if disable_torque:
            try:
                robot.bus.disable_torque()
                print("Motor torque disabled. The arm can now be moved by hand.")
            except Exception as exc:
                print(f"WARNING: Failed to disable motor torque: {exc}")


def return_home_and_disable_torque(robot) -> None:
    move_home(robot, disable_torque=True)


def return_home_for_next_task(robot) -> None:
    move_home(robot, disable_torque=False)


def exact_act_task(task_text: str) -> str | None:
    candidate = task_text.strip()
    return candidate if candidate in ACT_ALLOWED_TASKS.values() else None


def task_goal_color(task: str) -> str:
    for color, canonical in ACT_ALLOWED_TASKS.items():
        if task == canonical:
            return color
    raise ValueError(f"No ACT goal color for task: {task!r}")


def add_goal_observation_if_needed(ds_features: dict, values: dict, task: str) -> dict:
    if GOAL_FEATURE not in ds_features:
        return values
    goal_names = ds_features[GOAL_FEATURE].get("names") or []
    goal = GOAL_ONE_HOT[task_goal_color(task)]
    if len(goal_names) != len(goal):
        raise ValueError(f"Unexpected {GOAL_FEATURE} names: {goal_names}")
    result = dict(values)
    result.update(dict(zip(goal_names, goal, strict=True)))
    return result


def allowed_task_message() -> str:
    return " | ".join(ACT_ALLOWED_TASKS.values())


def predict_action_with_chunk(
    *,
    observation_frame: dict,
    policy,
    device,
    preprocessor,
    postprocessor,
    use_amp: bool,
    task: str,
    robot_type: str,
):
    observation = dict(observation_frame)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else torch.no_grad(),
    ):
        observation = prepare_observation_for_inference(observation, device, task, robot_type)
        observation = preprocessor(observation)
        raw_chunk = policy.predict_action_chunk(observation)
        if policy.config.temporal_ensemble_coeff is not None:
            raw_action = policy.temporal_ensembler.update(raw_chunk)
        else:
            raw_action = raw_chunk[:, 0]
        action = postprocessor(raw_action)
        action_chunk = postprocessor(raw_chunk)
    return action, action_chunk


def current_joint_position(obs_processed: dict, ds_features: dict) -> np.ndarray | None:
    names = ds_features.get("observation.state", {}).get("names") or ds_features[ACTION].get("names") or []
    if not all(name in obs_processed for name in names):
        return None
    return np.asarray([obs_processed[name] for name in names], dtype=np.float32)


def zero_velocity_chunk_metrics(action_chunk, current_position: np.ndarray, horizon: int) -> tuple[float, float]:
    chunk = action_chunk.detach().to("cpu").numpy().squeeze(0)
    if chunk.ndim != 2 or current_position is None:
        return float("inf"), float("inf")
    chunk = chunk[: max(1, min(horizon, chunk.shape[0]))]
    delta = np.abs(chunk - current_position[None, :])
    return float(delta.max()), float(delta.mean())


def action_chunk_stability_metrics(action_chunk, previous_chunk: np.ndarray | None, horizon: int) -> tuple[float, float, np.ndarray]:
    chunk = action_chunk.detach().to("cpu").numpy().squeeze(0)
    if chunk.ndim != 2:
        return float("inf"), float("inf"), chunk
    if previous_chunk is None or previous_chunk.ndim != 2:
        return float("inf"), float("inf"), chunk

    # Align time: previous_chunk[1] and current_chunk[0] both predict the next executed step.
    aligned_horizon = max(1, min(horizon, chunk.shape[0], previous_chunk.shape[0] - 1))
    delta = np.abs(chunk[:aligned_horizon] - previous_chunk[1 : aligned_horizon + 1])
    return float(delta.max()), float(delta.mean()), chunk


def fetch_app_state(args: argparse.Namespace, timeout_s: float | None = None) -> dict | None:
    try:
        req = request.Request(args.user_command_endpoint, method="GET")
        timeout = args.user_command_timeout_s if timeout_s is None else timeout_s
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def fetch_app_task_event(args: argparse.Namespace) -> tuple[str | None, str | None]:
    payload = fetch_app_state(args)
    if not payload:
        return None, None
    instruction = payload.get("instruction") or {}
    text = instruction.get("text")
    ts = instruction.get("ts")
    if isinstance(text, str):
        text = text.strip()
    if isinstance(ts, str):
        ts = ts.strip()
    return text or None, ts or None


def fetch_app_stop_event(args: argparse.Namespace, timeout_s: float | None = None) -> str | None:
    payload = fetch_app_state(args, timeout_s=timeout_s)
    if not payload:
        return None
    stop = payload.get("stop") or {}
    if not stop.get("requested"):
        return None
    ts = stop.get("ts")
    return ts.strip() if isinstance(ts, str) else None


def notify_app_success(args: argparse.Namespace, task: str) -> None:
    if not args.use_app_command_task_bridge:
        return
    endpoint = args.user_command_endpoint.rsplit("/", 1)[0] + "/success"
    payload = json.dumps({"task": task}).encode("utf-8")
    try:
        req = request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=args.user_command_timeout_s):
            pass
        print(f"App success recorded: {task!r}")
    except (URLError, HTTPError, TimeoutError, ValueError) as exc:
        print(f"WARNING: Failed to record app success: {exc}")


def resolve_initial_task(args: argparse.Namespace) -> str:
    fallback = exact_act_task(args.task)
    if fallback:
        return fallback
    raise ValueError("Unsupported ACT task. Allowed tasks: " + allowed_task_message())


def wait_for_next_app_task(args: argparse.Namespace, last_seen_event_id: str | None) -> tuple[str, str | None]:
    print("Waiting for app command. Allowed ACT tasks:")
    for allowed in ACT_ALLOWED_TASKS.values():
        print(f"  - {allowed}")

    while True:
        raw_task, ts = fetch_app_task_event(args)
        event_id = ts or raw_task
        if raw_task and event_id != last_seen_event_id:
            task = exact_act_task(raw_task)
            if task:
                print(f"App command task applied: {task!r}")
                return task, event_id
            print(f"Unsupported app command ignored: {raw_task!r}")
            print(f"Allowed ACT tasks: {allowed_task_message()}")
            last_seen_event_id = event_id
        time.sleep(args.user_command_poll_s)


def run_policy_rollout(
    *,
    task: str,
    args: argparse.Namespace,
    dataset,
    policy,
    device,
    preprocessor,
    postprocessor,
    robot,
    robot_action_processor,
    robot_observation_processor,
    last_stop_event_id: str | None = None,
) -> str | None:
    policy.reset()
    start_t = time.perf_counter()
    step = 0
    zero_velocity_count = 0
    stable_chunk_count = 0
    previous_action_chunk = None
    last_stop_poll_t = 0.0
    print(f"Starting ACT rollout: {task!r}")
    while time.perf_counter() - start_t < args.duration_s:
        loop_t = time.perf_counter()
        stop_event_id = None
        if args.use_app_command_task_bridge and loop_t - last_stop_poll_t >= args.stop_command_poll_s:
            last_stop_poll_t = loop_t
            stop_event_id = fetch_app_stop_event(args, timeout_s=args.stop_command_timeout_s)
        if stop_event_id and stop_event_id != last_stop_event_id:
            print("App stop command received. Aborting current rollout and returning home.")
            return stop_event_id
        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        obs_processed = add_goal_observation_if_needed(dataset.features, obs_processed, task)
        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
        current_position = current_joint_position(obs_processed, dataset.features)
        action_tensor, action_chunk = predict_action_with_chunk(
            observation_frame=observation_frame,
            policy=policy,
            device=device,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=task,
            robot_type=robot.robot_type,
        )
        action_values = make_robot_action(action_tensor, dataset.features)
        robot_action_to_send = robot_action_processor((action_values, obs))
        robot.send_action(robot_action_to_send)

        zero_max_delta = float("inf")
        zero_mean_delta = float("inf")
        stable_max_delta = float("inf")
        stable_mean_delta = float("inf")
        elapsed_s = time.perf_counter() - start_t
        if args.auto_stop_on_zero_velocity and current_position is not None:
            zero_max_delta, zero_mean_delta = zero_velocity_chunk_metrics(
                action_chunk, current_position, args.zero_velocity_horizon
            )
            is_zero_velocity = (
                elapsed_s >= args.zero_velocity_min_duration_s
                and zero_max_delta <= args.zero_velocity_max_delta_deg
                and zero_mean_delta <= args.zero_velocity_mean_delta_deg
            )
            zero_velocity_count = zero_velocity_count + 1 if is_zero_velocity else 0
        else:
            zero_velocity_count = 0

        if args.auto_stop_on_chunk_stability:
            stable_max_delta, stable_mean_delta, current_action_chunk = action_chunk_stability_metrics(
                action_chunk, previous_action_chunk, args.chunk_stability_horizon
            )
            is_stable_chunk = (
                elapsed_s >= args.chunk_stability_min_duration_s
                and stable_max_delta <= args.chunk_stability_max_delta_deg
                and stable_mean_delta <= args.chunk_stability_mean_delta_deg
            )
            stable_chunk_count = stable_chunk_count + 1 if is_stable_chunk else 0
            previous_action_chunk = current_action_chunk
        else:
            stable_chunk_count = 0
            previous_action_chunk = action_chunk.detach().to("cpu").numpy().squeeze(0)

        if args.print_every > 0 and step % args.print_every == 0:
            preview = {k: round(float(v), 3) for k, v in list(robot_action_to_send.items())[:4]}
            if args.auto_stop_on_zero_velocity or args.auto_stop_on_chunk_stability:
                print(
                    f"[ACT {step:04d}] {preview} "
                    f"zero_vel max={zero_max_delta:.2f} mean={zero_mean_delta:.2f} "
                    f"count={zero_velocity_count}/{args.zero_velocity_consecutive_steps} "
                    f"stable max={stable_max_delta:.2f} mean={stable_mean_delta:.2f} "
                    f"count={stable_chunk_count}/{args.chunk_stability_consecutive_steps}"
                )
            else:
                print(f"[ACT {step:04d}] {preview}")

        if stable_chunk_count >= args.chunk_stability_consecutive_steps:
            print(
                "Auto stop: stable action chunk detected "
                f"for {stable_chunk_count} consecutive steps "
                f"(max_delta={stable_max_delta:.2f}, mean_delta={stable_mean_delta:.2f})."
            )
            notify_app_success(args, task)
            break

        if zero_velocity_count >= args.zero_velocity_consecutive_steps:
            print(
                "Auto stop: zero-velocity chunk detected "
                f"for {zero_velocity_count} consecutive steps "
                f"(max_delta={zero_max_delta:.2f}, mean_delta={zero_mean_delta:.2f})."
            )
            notify_app_success(args, task)
            break

        step += 1
        precise_sleep(max(1.0 / args.fps - (time.perf_counter() - loop_t), 0.0))
    print(f"ACT rollout finished: {task!r}")
    return last_stop_event_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy_path", default=os.getenv("ACT_POLICY_PATH"), required=os.getenv("ACT_POLICY_PATH") is None)
    parser.add_argument("--dataset_repo_id", default=os.getenv("ACT_DATASET_REPO_ID"), required=os.getenv("ACT_DATASET_REPO_ID") is None)
    parser.add_argument("--dataset_root", default=os.getenv("ACT_DATASET_ROOT"))
    parser.add_argument("--task", default=os.getenv("TASK_DESCRIPTION", ACT_ALLOWED_TASKS["green"]))
    parser.add_argument("--use_app_command_task_bridge", action=argparse.BooleanOptionalAction, default=env_bool("USE_USER_COMMAND_TASK_BRIDGE"))
    parser.add_argument("--user_command_endpoint", default=os.getenv("USER_COMMAND_ENDPOINT", "http://127.0.0.1:8000/command/latest"))
    parser.add_argument("--user_command_timeout_s", type=float, default=float(os.getenv("USER_COMMAND_TIMEOUT_S", "1.0")))
    parser.add_argument("--user_command_poll_s", type=float, default=float(os.getenv("USER_COMMAND_POLL_S", "0.5")))
    parser.add_argument("--stop_command_poll_s", type=float, default=float(os.getenv("ACT_STOP_COMMAND_POLL_S", "0.2")))
    parser.add_argument("--stop_command_timeout_s", type=float, default=float(os.getenv("ACT_STOP_COMMAND_TIMEOUT_S", "0.03")))
    parser.add_argument("--wait_for_app_command", action=argparse.BooleanOptionalAction, default=env_bool("WAIT_FOR_APP_COMMAND", "true"))
    parser.add_argument("--run_existing_app_command", action=argparse.BooleanOptionalAction, default=env_bool("ACT_RUN_EXISTING_APP_COMMAND"))
    parser.add_argument("--duration_s", type=float, default=float(os.getenv("ACT_DURATION_S", "30")))
    parser.add_argument("--fps", type=int, default=int(os.getenv("ACT_FPS", "10"))) # AI 추론 속도
    parser.add_argument("--camera_fps", type=int, default=30) # 수정됨: 카메라 하드웨어 속도 인자 추가
    parser.add_argument("--device", default=os.getenv("ACT_DEVICE", "cuda"))
    parser.add_argument("--robot_port", default=os.getenv("ROBOT_PORT", "/dev/ttyACM1"))
    parser.add_argument("--robot_id", default=os.getenv("ROBOT_ID", "my_follower"))
    parser.add_argument("--calibration_dir", default=os.getenv("LEROBOT_CALIBRATION_DIR"))
    parser.add_argument("--camera_name", default=os.getenv("ACT_CAMERA_NAME", "top"))
    parser.add_argument("--camera", default=os.getenv("ACT_CAMERA", "0"))
    parser.add_argument("--second_camera_name", default=os.getenv("ACT_SECOND_CAMERA_NAME", "wrist"))
    parser.add_argument("--second_camera", default=os.getenv("ACT_SECOND_CAMERA"), help="Optional second camera index/path, used only if the ACT dataset was trained with two views.")
    parser.add_argument("--width", type=int, default=int(os.getenv("ACT_CAMERA_WIDTH", "640")))
    parser.add_argument("--height", type=int, default=int(os.getenv("ACT_CAMERA_HEIGHT", "480")))
    parser.add_argument("--fourcc", default=os.getenv("ACT_CAMERA_FOURCC", "MJPG"))
    parser.add_argument("--max_relative_target", type=float, default=None)
    parser.add_argument("--temporal_ensemble_coeff", type=float, default=None)
    parser.add_argument("--auto_stop_on_zero_velocity", action=argparse.BooleanOptionalAction, default=env_bool("ACT_AUTO_STOP_ON_ZERO_VELOCITY", "true"))
    parser.add_argument("--zero_velocity_min_duration_s", type=float, default=float(os.getenv("ACT_ZERO_VELOCITY_MIN_DURATION_S", "12")))
    parser.add_argument("--zero_velocity_consecutive_steps", type=int, default=int(os.getenv("ACT_ZERO_VELOCITY_CONSECUTIVE_STEPS", "20")))
    parser.add_argument("--zero_velocity_horizon", type=int, default=int(os.getenv("ACT_ZERO_VELOCITY_HORIZON", "30")))
    parser.add_argument("--zero_velocity_max_delta_deg", type=float, default=float(os.getenv("ACT_ZERO_VELOCITY_MAX_DELTA_DEG", "3.0")))
    parser.add_argument("--zero_velocity_mean_delta_deg", type=float, default=float(os.getenv("ACT_ZERO_VELOCITY_MEAN_DELTA_DEG", "0.8")))
    parser.add_argument("--auto_stop_on_chunk_stability", action=argparse.BooleanOptionalAction, default=env_bool("ACT_AUTO_STOP_ON_CHUNK_STABILITY", "true"))
    parser.add_argument("--chunk_stability_min_duration_s", type=float, default=float(os.getenv("ACT_CHUNK_STABILITY_MIN_DURATION_S", "10")))
    parser.add_argument("--chunk_stability_consecutive_steps", type=int, default=int(os.getenv("ACT_CHUNK_STABILITY_CONSECUTIVE_STEPS", "20")))
    parser.add_argument("--chunk_stability_horizon", type=int, default=int(os.getenv("ACT_CHUNK_STABILITY_HORIZON", "30")))
    parser.add_argument("--chunk_stability_max_delta_deg", type=float, default=float(os.getenv("ACT_CHUNK_STABILITY_MAX_DELTA_DEG", "6.0")))
    parser.add_argument("--chunk_stability_mean_delta_deg", type=float, default=float(os.getenv("ACT_CHUNK_STABILITY_MEAN_DELTA_DEG", "1.5")))
    parser.add_argument("--return_home_after_each_task", action=argparse.BooleanOptionalAction, default=env_bool("ACT_RETURN_HOME_AFTER_EACH_TASK", "true"))
    parser.add_argument("--settle_after_home_s", type=float, default=float(os.getenv("ACT_SETTLE_AFTER_HOME_S", "1.0")))
    parser.add_argument("--disable_torque_on_disconnect", action="store_true")
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()

    init_logging()
    task = None if args.use_app_command_task_bridge else resolve_initial_task(args)
    dataset = LeRobotDataset(args.dataset_repo_id, root=args.dataset_root, download_videos=False)

    policy_cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cfg.pretrained_path = Path(args.policy_path)
    policy_cfg.device = args.device
    if args.temporal_ensemble_coeff is not None:
        policy_cfg.temporal_ensemble_coeff = args.temporal_ensemble_coeff
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    policy.reset()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.policy_path,
        dataset_stats=rename_stats(dataset.meta.stats, {}),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )

    _, robot_action_processor, robot_observation_processor = make_default_processors()
    robot = make_robot_from_config(build_robot_config(args))
    device = get_safe_torch_device(args.device)

    print(f"ACT policy: {args.policy_path}")
    print(f"dataset meta: {args.dataset_repo_id} root={dataset.root}")
    print(f"task: {task if task else '(waiting for app command)'}")
    print("NOTE: stop the ZMQ hardware server before using main_act.py; this script opens the robot port directly.")

    last_app_event_id = None
    app_video_stop_event = threading.Event()
    app_video_publisher = None
    app_preview_thread = None
    if args.use_app_command_task_bridge and not args.run_existing_app_command:
        raw_task, ts = fetch_app_task_event(args)
        last_app_event_id = ts or raw_task
        last_stop_event_id = fetch_app_stop_event(args)
        if last_app_event_id:
            print("Ignoring existing app command; waiting for a new app command.")
    try:
        robot.connect()
        if APP_VIDEO_SERVER_ENABLED:
            camera_count = max(1, len(robot.cameras))
            app_video_publisher = SharedVideoPublisher(
                (args.height, args.width * camera_count, 3),
                host=APP_VIDEO_SERVER_HOST,
                port=APP_VIDEO_SERVER_PORT,
                frame_period_s=APP_VIDEO_STREAM_SLEEP_S,
                jpeg_quality=APP_VIDEO_JPEG_QUALITY,
            )
            print(f"ACT video process: http://{APP_VIDEO_SERVER_HOST}:{APP_VIDEO_SERVER_PORT}/video_feed")
            app_preview_thread = threading.Thread(
                target=camera_preview_worker,
                args=(robot, app_video_publisher, app_video_stop_event),
                daemon=True,
            )
            app_preview_thread.start()
        if args.use_app_command_task_bridge:
            while True:
                task, last_app_event_id = wait_for_next_app_task(args, last_app_event_id)
                last_stop_event_id = run_policy_rollout(
                    task=task,
                    args=args,
                    dataset=dataset,
                    policy=policy,
                    device=device,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    robot=robot,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    last_stop_event_id=last_stop_event_id,
                )
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                if args.return_home_after_each_task:
                    return_home_for_next_task(robot)
                    precise_sleep(max(args.settle_after_home_s, 0.0))
                print("Ready for next app command.")
        else:
            run_policy_rollout(
                task=task,
                args=args,
                dataset=dataset,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                robot=robot,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
            )
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        app_video_stop_event.set()
        if app_preview_thread is not None:
            app_preview_thread.join(timeout=1.0)
        if app_video_publisher is not None:
            app_video_publisher.close()
        if robot.bus.is_connected:
            if robot.is_connected:
                return_home_and_disable_torque(robot)
            else:
                try:
                    robot.bus.disable_torque()
                    print("Motor torque disabled after partial connection.")
                except Exception as exc:
                    print(f"WARNING: Failed to disable motor torque: {exc}")
            try:
                robot.disconnect()
            except Exception as exc:
                print(f"WARNING: Failed to disconnect cleanly: {exc}")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()