#!/usr/bin/env python3
"""LeRobot recording with VP-VLA-style sidecar event metadata.

This script intentionally keeps the LeRobot dataset schema unchanged. It records
normal LeRobot episodes, while writing VP metadata to:

    <dataset_root>/meta/vp_events/episode_000000.json

The sidecar files are meant to drive later SAM/VLM preprocessing for visual
prompt overlays.
"""


import json
import logging
import queue
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch

try:
    import cv2
except ImportError:
    cv2 = None

LEROBOT_SRC = Path("/home/aivlab/lerobot/src")
if LEROBOT_SRC.exists() and str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from gripper_telemetry import read_gripper_telemetry
from lerobot.common.control_utils import is_headless, predict_action, sanity_check_dataset_name
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.policies import ActionInterpolator, make_policy, make_pre_post_processors, make_robot_action
from lerobot.processor import make_default_processors, rename_stats
from lerobot.robots import make_robot_from_config
from lerobot.scripts import lerobot_record as base_record
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


DEFAULT_KEYMAP = "1:pick_phase_start,2:place_phase_start,3:success,4:failure"
PHASE_EVENT_TYPES = {"pick_phase_start", "place_phase_start"}


@dataclass
class VPEventConfig:
    enabled: bool = True
    output_dir: str | None = None
    task_family: str = "pick_place"
    target_object: str = ""
    target_location: str = ""
    scene_id: str = ""
    operator_id: str = ""
    start_zone: str = ""
    target_zone: str = ""
    distractor_count: int = 0
    object_set: str = ""
    notes: str = ""
    keymap: str = DEFAULT_KEYMAP
    episode_plan_path: str | None = None
    interactive_metadata: bool = False
    require_terminal_event: bool = True
    auto_pick_phase_start: bool = True
    freeze_action_keys: str = ""
    freeze_action_reference: str = "episode_start"
    record_gripper_proprio: bool = True
    gripper_proprio_save_every_n_frames: int = 1
    visual_prompt_enabled: bool = False
    visual_prompt_output_dir: str | None = None
    visual_prompt_save_every_n_frames: int = 1
    visual_prompt_camera: str = ""
    visual_prompt_coord_format: str = "pixel"
    target_object_point: str = ""
    target_location_box: str = ""
    crosshair_line_length: int = 16
    crosshair_gap: int = 7
    visual_prompt_thickness: int = 2


@dataclass
class VPRecordConfig(base_record.RecordConfig):
    vp: VPEventConfig = field(default_factory=VPEventConfig)

    def __post_init__(self):
        super().__post_init__()

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


def _parse_keymap(spec: str) -> dict[str, str]:
    keymap: dict[str, str] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid VP keymap entry {part!r}; expected KEY:event_type")
        key, event_type = part.split(":", 1)
        keymap[key.strip().lower()] = event_type.strip()
    return keymap


def _load_episode_plan(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"VP episode plan not found: {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] == "[":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("VP episode plan JSON must be a list of objects")
        return [dict(x) for x in data]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            rows.append(dict(json.loads(line)))
    return rows


def _prompt_default(prompt: str, default: str) -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def _to_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)
    except Exception:
        return None


def _parse_float_list(value: Any, expected: int) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        for sep in [",", " "]:
            if sep in value:
                parts = [p for p in value.replace(",", " ").split() if p]
                break
        else:
            parts = [value]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        return None
    if len(parts) != expected:
        raise ValueError(f"Expected {expected} coordinate values, got {len(parts)} from {value!r}")
    return [float(x) for x in parts]


def _normalize_coord_format(value: Any) -> str:
    fmt = str(value or "pixel").strip().lower()
    aliases = {
        "px": "pixel",
        "pixels": "pixel",
        "norm": "normalized",
        "relative": "normalized",
        "rel": "normalized",
        "1000": "qwen1000",
        "qwen": "qwen1000",
        "qwen_1000": "qwen1000",
    }
    return aliases.get(fmt, fmt)


def _scale_xy(values: list[float], width: int, height: int, coord_format: str) -> list[float]:
    fmt = _normalize_coord_format(coord_format)
    scaled = list(values)
    if fmt == "normalized":
        scaled = [values[0] * width, values[1] * height]
        if len(values) == 4:
            scaled = [values[0] * width, values[1] * height, values[2] * width, values[3] * height]
    elif fmt == "qwen1000":
        scaled = [values[0] / 1000.0 * width, values[1] / 1000.0 * height]
        if len(values) == 4:
            scaled = [
                values[0] / 1000.0 * width,
                values[1] / 1000.0 * height,
                values[2] / 1000.0 * width,
                values[3] / 1000.0 * height,
            ]
    elif fmt != "pixel":
        raise ValueError("visual_prompt_coord_format must be one of: pixel, normalized, qwen1000")
    return scaled


def _clip_point(point: list[float], width: int, height: int) -> tuple[int, int]:
    x = int(round(min(max(point[0], 0), width - 1)))
    y = int(round(min(max(point[1], 0), height - 1)))
    return x, y


def _clip_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1 = int(round(min(max(box[0], 0), width - 1)))
    y1 = int(round(min(max(box[1], 0), height - 1)))
    x2 = int(round(min(max(box[2], 0), width - 1)))
    y2 = int(round(min(max(box[3], 0), height - 1)))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _image_to_uint8_hwc(image: Any) -> tuple[np.ndarray, dict[str, Any]]:
    meta = {"kind": "numpy", "dtype": None, "layout": "hwc", "pil_mode": None, "torch_device": None}
    if hasattr(image, "mode") and hasattr(image, "size"):
        meta["kind"] = "pil"
        meta["pil_mode"] = image.mode
        arr = np.asarray(image.convert("RGB"))
    elif isinstance(image, torch.Tensor):
        meta["kind"] = "torch"
        meta["dtype"] = image.dtype
        meta["torch_device"] = image.device
        arr = image.detach().cpu().numpy()
    elif isinstance(image, np.ndarray):
        meta["kind"] = "numpy"
        meta["dtype"] = image.dtype
        arr = image
    else:
        raise TypeError(f"Unsupported image type for VP overlay: {type(image)}")

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D image array, got shape {arr.shape}")
    if arr.shape[0] == 3 and arr.shape[-1] != 3:
        meta["layout"] = "chw"
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {arr.shape}")

    if arr.dtype == np.uint8:
        return arr.copy(), meta
    arr_float = arr.astype(np.float32)
    if arr_float.max(initial=0) <= 1.0:
        arr_float = arr_float * 255.0
    return np.clip(arr_float, 0, 255).astype(np.uint8), meta


def _restore_image_type(arr: np.ndarray, meta: dict[str, Any]) -> Any:
    out = arr
    if meta["layout"] == "chw":
        out = np.transpose(out, (2, 0, 1))
    if meta["kind"] == "pil":
        from PIL import Image

        return Image.fromarray(arr).convert(meta.get("pil_mode") or "RGB")
    if meta["kind"] == "torch":
        tensor = torch.from_numpy(out).to(meta["torch_device"])
        if meta["dtype"] is not None and meta["dtype"] != torch.uint8:
            tensor = tensor.to(torch.float32) / 255.0
            tensor = tensor.to(meta["dtype"])
        return tensor
    if meta["dtype"] is not None and meta["dtype"] != np.uint8:
        out = out.astype(np.float32) / 255.0
        return out.astype(meta["dtype"])
    return out


def _draw_visual_prompt(
    image: Any,
    point: list[float] | None,
    box: list[float] | None,
    coord_format: str,
    line_length: int,
    gap: int,
    thickness: int,
) -> Any:
    if cv2 is None:
        raise ImportError("cv2 is required for --vp.visual_prompt_enabled=true")
    arr, meta = _image_to_uint8_hwc(image)
    height, width = arr.shape[:2]
    result = arr.copy()
    if box is not None:
        scaled_box = _scale_xy(box, width, height, coord_format)
        clipped_box = _clip_box(scaled_box, width, height)
        if clipped_box is not None:
            cv2.rectangle(result, clipped_box[:2], clipped_box[2:], (255, 0, 0), thickness)
    if point is not None:
        scaled_point = _scale_xy(point, width, height, coord_format)
        x, y = _clip_point(scaled_point, width, height)
        cv2.circle(result, (x, y), max(2, thickness + 1), (255, 0, 0), -1)
        cv2.line(result, (x - gap - line_length, y), (x - gap, y), (0, 255, 0), thickness)
        cv2.line(result, (x + gap, y), (x + gap + line_length, y), (0, 255, 0), thickness)
        cv2.line(result, (x, y - gap - line_length), (x, y - gap), (0, 255, 0), thickness)
        cv2.line(result, (x, y + gap), (x, y + gap + line_length), (0, 255, 0), thickness)
    return _restore_image_type(result, meta)


class VPEventRecorder:
    def __init__(self, cfg: VPEventConfig, dataset: LeRobotDataset, dataset_repo_id: str):
        self.cfg = cfg
        self.dataset = dataset
        self.dataset_repo_id = dataset_repo_id
        self.keymap = _parse_keymap(cfg.keymap)
        self.pending: queue.Queue[tuple[float, str, str]] = queue.Queue()
        self.active = False
        self.session: dict[str, Any] | None = None
        self.frame_count = 0
        self.plan = _load_episode_plan(cfg.episode_plan_path)
        self._warned_visual_prompt_key = False
        self._warned_freeze_key = False
        self._warned_gripper_proprio = False
        self.frozen_action_values: dict[str, float] = {}
        self._last_gripper_sample: dict[str, Any] | None = None

        if cfg.visual_prompt_output_dir:
            self.visual_prompt_output_dir = Path(cfg.visual_prompt_output_dir).expanduser()
        else:
            self.visual_prompt_output_dir = Path(dataset.root) / "meta" / "vp_visual_prompts"
        if cfg.visual_prompt_enabled:
            self.visual_prompt_output_dir.mkdir(parents=True, exist_ok=True)

        if cfg.output_dir:
            self.output_dir = Path(cfg.output_dir).expanduser()
        else:
            self.output_dir = Path(dataset.root) / "meta" / "vp_events"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _freeze_action_keys(self) -> list[str]:
        return [k.strip() for k in str(self.cfg.freeze_action_keys or "").replace(",", " ").split() if k.strip()]

    def apply_action_freeze(
        self,
        action_values: dict[str, Any],
        robot_action_to_send: dict[str, Any],
        observation: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        keys = self._freeze_action_keys()
        if not keys:
            return action_values, robot_action_to_send

        for key in keys:
            if key not in self.frozen_action_values:
                if key in observation:
                    self.frozen_action_values[key] = float(observation[key])
                elif key in action_values:
                    self.frozen_action_values[key] = float(action_values[key])
                elif key in robot_action_to_send:
                    self.frozen_action_values[key] = float(robot_action_to_send[key])
                else:
                    if not self._warned_freeze_key:
                        print(f"[VP] WARNING: freeze action key not found in observation/action: {key!r}")
                        self._warned_freeze_key = True
                    continue
                print(f"[VP] locked {key}={self.frozen_action_values[key]:.6g}")

            frozen_value = self.frozen_action_values[key]
            if key in action_values:
                action_values[key] = frozen_value
            if key in robot_action_to_send:
                robot_action_to_send[key] = frozen_value

        if self.session is not None:
            self.session["action_freeze"] = {
                "enabled": True,
                "reference": self.cfg.freeze_action_reference,
                "keys": keys,
                "values": dict(self.frozen_action_values),
                "applied_to": ["robot_command", "dataset_action"],
            }
        return action_values, robot_action_to_send

    def _visual_prompt_values(self) -> dict[str, Any]:
        if self.session is None:
            return {}
        point = self.session.get("target_object_point")
        if point is None:
            point = self.session.get("target_object_xy")
        if point is None:
            point = self.session.get("target_object_center")
        box = self.session.get("target_location_box")
        if box is None:
            box = self.session.get("target_bbox")
        if box is None:
            box = self.session.get("place_box")
        camera = self.session.get("visual_prompt_camera") or self.cfg.visual_prompt_camera
        coord_format = self.session.get("visual_prompt_coord_format") or self.cfg.visual_prompt_coord_format
        return {
            "point": _parse_float_list(point, 2),
            "box": _parse_float_list(box, 4),
            "camera": str(camera or ""),
            "coord_format": str(coord_format or "pixel"),
        }

    def _resolve_visual_prompt_image_key(self, frame: dict[str, Any], features: dict[str, dict], camera: str) -> str | None:
        image_keys = [
            key
            for key, ft in features.items()
            if key in frame and key.startswith("observation.images.") and ft.get("dtype") in ["image", "video"]
        ]
        if not image_keys:
            return None
        if camera:
            candidates = [camera]
            if not camera.startswith("observation.images."):
                candidates.append(f"observation.images.{camera}")
            for key in candidates:
                if key in image_keys:
                    return key
            return None
        return image_keys[0]

    def record_gripper_proprio(
        self,
        robot,
        observation: dict[str, Any],
        action_values: dict[str, Any],
        frame_index: int,
        timestamp_s: float,
    ) -> None:
        if self.session is None or not self.active or not self.cfg.record_gripper_proprio:
            return
        stride = max(int(self.cfg.gripper_proprio_save_every_n_frames), 1)
        if int(frame_index) % stride != 0:
            return

        try:
            telemetry = read_gripper_telemetry(robot, observation)
        except Exception as e:
            telemetry = {"errors": {"read_gripper_telemetry": f"{type(e).__name__}: {e}"}}
            if not self._warned_gripper_proprio:
                print(f"[VP] WARNING: gripper proprio read failed: {telemetry['errors']['read_gripper_telemetry']}")
                self._warned_gripper_proprio = True

        pos = _to_float_or_none(telemetry.get("pos"))
        current = _to_float_or_none(telemetry.get("current"))
        load = _to_float_or_none(telemetry.get("load"))
        action_gripper = None
        for key in ("gripper.pos", "gripper"):
            if key in action_values:
                action_gripper = _to_float_or_none(action_values[key])
                break

        velocity = None
        if self._last_gripper_sample is not None and pos is not None:
            prev_pos = self._last_gripper_sample.get("pos")
            prev_t = self._last_gripper_sample.get("time_s")
            if prev_pos is not None and prev_t is not None and float(timestamp_s) > float(prev_t):
                velocity = (float(pos) - float(prev_pos)) / (float(timestamp_s) - float(prev_t))

        sample = {
            "frame": int(frame_index),
            "time_s": float(timestamp_s),
            "phase": self.session.get("current_phase"),
            "pos": pos,
            "current": current,
            "load": load,
            "velocity": velocity,
            "action_gripper": action_gripper,
        }
        errors = telemetry.get("errors") if isinstance(telemetry, dict) else None
        if errors:
            sample["errors"] = errors

        self.session.setdefault("gripper_proprio", []).append(sample)
        proprio_meta = self.session.setdefault(
            "proprio",
            {
                "enabled": True,
                "storage": "vp_events_json_sidecar",
                "gripper": {
                    "signals": ["pos", "current", "load", "velocity", "action_gripper"],
                    "save_every_n_frames": stride,
                },
                "num_gripper_samples": 0,
            },
        )
        proprio_meta["num_gripper_samples"] = len(self.session["gripper_proprio"])
        self._last_gripper_sample = sample

    def save_visual_prompt_copy(self, frame: dict[str, Any], features: dict[str, dict], frame_index: int) -> None:
        if not self.cfg.visual_prompt_enabled or self.session is None:
            return
        stride = max(int(self.cfg.visual_prompt_save_every_n_frames), 1)
        if int(frame_index) % stride != 0:
            return
        values = self._visual_prompt_values()
        if values["point"] is None and values["box"] is None:
            return
        image_key = self._resolve_visual_prompt_image_key(frame, features, values["camera"])
        if image_key is None:
            if not self._warned_visual_prompt_key:
                print(f"[VP] WARNING: visual prompt camera not found: {values['camera']!r}")
                self._warned_visual_prompt_key = True
            return

        overlay_image = _draw_visual_prompt(
            frame[image_key],
            point=values["point"],
            box=values["box"],
            coord_format=values["coord_format"],
            line_length=self.cfg.crosshair_line_length,
            gap=self.cfg.crosshair_gap,
            thickness=self.cfg.visual_prompt_thickness,
        )
        overlay_rgb, _ = _image_to_uint8_hwc(overlay_image)

        from PIL import Image

        episode_index = int(self.session["episode_index"])
        camera_name = image_key.removeprefix("observation.images.").replace("/", "_")
        out_dir = self.visual_prompt_output_dir / f"episode_{episode_index:06d}" / camera_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"frame_{int(frame_index):06d}.png"
        Image.fromarray(overlay_rgb).save(out_path)

        visual_prompt = self.session.setdefault(
            "visual_prompt",
            {
                "enabled": True,
                "storage": "sidecar_png",
                "output_dir": str(self.visual_prompt_output_dir),
                "image_key": image_key,
                "target_object_point": values["point"],
                "target_location_box": values["box"],
                "coord_format": values["coord_format"],
                "save_every_n_frames": stride,
                "saved_frames": 0,
            },
        )
        visual_prompt["saved_frames"] = int(visual_prompt.get("saved_frames", 0)) + 1
        visual_prompt["last_path"] = str(out_path)

    def on_key(self, key_name: str) -> bool:
        event_type = self.keymap.get(key_name.lower())
        if not event_type:
            return False
        if not self.active:
            print(f"[VP] ignored key {key_name!r} -> {event_type}; no active recording episode")
            return True
        self.pending.put((time.perf_counter(), key_name, event_type))
        print(f"[VP] queued {event_type}")
        return True

    def _base_metadata(self, episode_index: int, task: str) -> dict[str, Any]:
        meta = {
            "task_family": self.cfg.task_family,
            "language_instruction": task,
            "target_object": self.cfg.target_object,
            "target_location": self.cfg.target_location,
            "scene_id": self.cfg.scene_id,
            "operator_id": self.cfg.operator_id,
            "start_zone": self.cfg.start_zone,
            "target_zone": self.cfg.target_zone,
            "distractor_count": self.cfg.distractor_count,
            "object_set": self.cfg.object_set,
            "notes": self.cfg.notes,
            "visual_prompt_camera": self.cfg.visual_prompt_camera,
            "visual_prompt_coord_format": self.cfg.visual_prompt_coord_format,
            "target_object_point": self.cfg.target_object_point,
            "target_location_box": self.cfg.target_location_box,
        }
        if episode_index < len(self.plan):
            meta.update({k: v for k, v in self.plan[episode_index].items() if v is not None})
        if self.cfg.interactive_metadata:
            print("\n[VP] Episode metadata. Press Enter to keep defaults.")
            meta["task_family"] = _prompt_default("task_family", str(meta.get("task_family", "")))
            meta["language_instruction"] = _prompt_default(
                "language_instruction", str(meta.get("language_instruction", ""))
            )
            meta["target_object"] = _prompt_default("target_object", str(meta.get("target_object", "")))
            meta["target_location"] = _prompt_default("target_location", str(meta.get("target_location", "")))
            meta["start_zone"] = _prompt_default("start_zone", str(meta.get("start_zone", "")))
            meta["target_zone"] = _prompt_default("target_zone", str(meta.get("target_zone", "")))
            meta["scene_id"] = _prompt_default("scene_id", str(meta.get("scene_id", "")))
            if self.cfg.visual_prompt_enabled:
                meta["visual_prompt_camera"] = _prompt_default(
                    "visual_prompt_camera", str(meta.get("visual_prompt_camera", ""))
                )
                meta["visual_prompt_coord_format"] = _prompt_default(
                    "visual_prompt_coord_format", str(meta.get("visual_prompt_coord_format", "pixel"))
                )
                meta["target_object_point"] = _prompt_default(
                    "target_object_point x,y", str(meta.get("target_object_point", ""))
                )
                meta["target_location_box"] = _prompt_default(
                    "target_location_box x1,y1,x2,y2", str(meta.get("target_location_box", ""))
                )
            meta["notes"] = _prompt_default("notes", str(meta.get("notes", "")))
        return meta

    def _phase_targets_for_event(self, event_type: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        source = metadata if metadata is not None else (self.session or {})
        target_object = str(source.get("target_object", "") or "")
        target_location = str(source.get("target_location", "") or "")
        if event_type == "pick_phase_start":
            return {
                "phase": "pick",
                "target_object": target_object,
                "target_location": None,
                "overlay_target_object": "crosshair",
                "overlay_target_location": None,
                "sam_target_object_prompt": target_object or None,
                "sam_target_location_prompt": None,
            }
        if event_type == "place_phase_start":
            return {
                "phase": "place",
                "target_object": target_object,
                "target_location": target_location or None,
                "overlay_target_object": "crosshair" if target_object else None,
                "overlay_target_location": "box" if target_location else None,
                "sam_target_object_prompt": target_object or None,
                "sam_target_location_prompt": target_location or None,
            }
        return {}

    def _make_event(
        self,
        frame_index: int,
        timestamp_s: float,
        event_type: str,
        key_name: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = {
            "frame": int(frame_index),
            "time_s": float(timestamp_s),
            "type": event_type,
            "key": key_name,
        }
        if event_type in PHASE_EVENT_TYPES:
            event["active_visual_targets"] = self._phase_targets_for_event(event_type, metadata=metadata)
        return event

    def _apply_event_to_session(self, event: dict[str, Any]) -> None:
        if self.session is None:
            return
        event_type = event.get("type")
        if event_type in PHASE_EVENT_TYPES:
            targets = event.get("active_visual_targets") or self._phase_targets_for_event(str(event_type))
            self.session["current_phase"] = targets.get("phase")
            self.session["active_visual_targets"] = targets
        elif event_type == "success":
            self.session["success"] = True
        elif event_type == "failure":
            self.session["success"] = False
            self.session["failure_reason"] = self.session.get("failure_reason") or "marked_by_operator"

    def start_episode(self, episode_index: int, task: str, fps: int, robot_type: str) -> str:
        while not self.pending.empty():
            self.pending.get_nowait()
        metadata = self._base_metadata(episode_index, task)
        self.frame_count = 0
        self.frozen_action_values = {}
        self._last_gripper_sample = None
        initial_targets = (
            self._phase_targets_for_event("pick_phase_start", metadata=metadata)
            if self.cfg.auto_pick_phase_start
            else {}
        )
        initial_events = [self._make_event(0, 0.0, "episode_start", None, metadata=metadata)]
        if self.cfg.auto_pick_phase_start:
            initial_events.append(self._make_event(0, 0.0, "pick_phase_start", "auto", metadata=metadata))

        self.session = {
            "schema": "so101_vp_events_v2",
            "vp_phase_policy": "visual_target_phase_v1",
            "phase_definitions": {
                "pick": {
                    "starts_at": "episode_start",
                    "target_object_overlay": "crosshair",
                    "target_location_overlay": None,
                },
                "place": {
                    "starts_at": "place_phase_start",
                    "target_object_overlay": "crosshair",
                    "target_location_overlay": "box",
                },
            },
            "episode_index": episode_index,
            "dataset_repo_id": self.dataset_repo_id,
            "robot_type": robot_type,
            "fps": fps,
            **metadata,
            "current_phase": initial_targets.get("phase"),
            "active_visual_targets": initial_targets or None,
            "action_freeze": {
                "enabled": bool(self._freeze_action_keys()),
                "reference": self.cfg.freeze_action_reference,
                "keys": self._freeze_action_keys(),
                "values": {},
                "applied_to": ["robot_command", "dataset_action"],
            },
            "proprio": {
                "enabled": bool(self.cfg.record_gripper_proprio),
                "storage": "vp_events_json_sidecar",
                "gripper": {
                    "signals": ["pos", "current", "load", "velocity", "action_gripper"],
                    "save_every_n_frames": max(int(self.cfg.gripper_proprio_save_every_n_frames), 1),
                },
                "num_gripper_samples": 0,
            },
            "gripper_proprio": [],
            "success": None,
            "failure_reason": None,
            "events": initial_events,
            "keymap": self.keymap,
        }
        self.active = True
        self.print_legend(episode_index)
        return str(self.session["language_instruction"])

    def end_recording_window(self):
        self.active = False

    def discard_episode(self):
        self.session = None
        self.frame_count = 0
        while not self.pending.empty():
            self.pending.get_nowait()

    def drain_events(self, frame_index: int, timestamp_s: float):
        if self.session is None:
            return
        self.frame_count = max(self.frame_count, frame_index + 1)
        while not self.pending.empty():
            _, key_name, event_type = self.pending.get_nowait()
            event = self._make_event(frame_index, timestamp_s, event_type, key_name)
            if self._last_gripper_sample is not None:
                event["gripper_proprio"] = dict(self._last_gripper_sample)
            self.session["events"].append(event)
            self._apply_event_to_session(event)

    def save_episode(self):
        if self.session is None:
            return
        self.session["num_frames"] = int(self.frame_count)
        self.session["duration_s"] = float(self.frame_count / max(float(self.session.get("fps", 1)), 1.0))
        terminal = any(e["type"] in ("success", "failure") for e in self.session["events"])
        if self.cfg.require_terminal_event and not terminal:
            print("[VP] WARNING: no success/failure event was marked for this episode.")

        out = self.output_dir / f"episode_{int(self.session['episode_index']):06d}.json"
        out.write_text(json.dumps(self.session, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[VP] saved {out}")
        self.session = None

    def print_legend(self, episode_index: int):
        bindings = ", ".join(f"{k}={v}" for k, v in self.keymap.items())
        print(f"\n[VP] Recording metadata for episode {episode_index:06d}")
        print(f"[VP] Event keys: {bindings}")
        print("[VP] VP-VLA phase policy: pick=target object crosshair, place=target object crosshair + target location box.")
        print("[VP] Default: pick_phase_start is inserted at frame 0 automatically; press 2 when the object is securely grasped and transport to the place target begins.")
        print("[VP] LeRobot controls still apply: right-arrow=end episode, left-arrow=rerecord, esc=stop.\n")


def init_vp_keyboard_listener(vp_recorder: VPEventRecorder | None):
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    if is_headless():
        logging.warning("Headless environment detected. Keyboard event metadata will not be available.")
        return None, events

    from pynput import keyboard

    def key_name(key) -> str | None:
        try:
            if hasattr(key, "char") and key.char is not None:
                return str(key.char).lower()
            if key == keyboard.Key.right:
                return "right"
            if key == keyboard.Key.left:
                return "left"
            if key == keyboard.Key.esc:
                return "esc"
        except Exception:
            return None
        return None

    def on_press(key):
        try:
            name = key_name(key)
            if name is None:
                return
            if vp_recorder is not None and vp_recorder.on_key(name):
                return
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


@safe_stop_image_writer
def record_loop_vp(
    *,
    robot,
    events: dict,
    fps: int,
    teleop_action_processor,
    robot_action_processor,
    robot_observation_processor,
    dataset: LeRobotDataset | None = None,
    teleop=None,
    policy=None,
    preprocessor=None,
    postprocessor=None,
    control_time_s: int | float | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    interpolator: ActionInterpolator | None = None,
    display_compressed_images: bool = False,
    vp_recorder: VPEventRecorder | None = None,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        base_record.so_leader.SO100Leader
                        | base_record.so_leader.SO101Leader
                        | base_record.koch_leader.KochLeader
                        | base_record.omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )
        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. "
                "Currently only supported for LeKiwi robot."
            )

    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()
    if interpolator is not None:
        interpolator.reset()

    use_interpolation = interpolator is not None and interpolator.enabled and policy is not None
    control_interval = interpolator.get_control_interval(fps) if interpolator else 1 / fps
    action_keys = sorted(robot.action_features) if use_interpolation else []

    no_action_count = 0
    timestamp = 0.0
    start_episode_t = time.perf_counter()
    while timestamp < float(control_time_s):
        start_loop_t = time.perf_counter()
        if events["exit_early"]:
            events["exit_early"] = False
            break

        obs = robot.get_observation()
        obs_processed = robot_observation_processor(obs)
        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        is_record_frame = True
        if policy is not None and preprocessor is not None and postprocessor is not None:
            if use_interpolation:
                ran_inference = False
                if interpolator.needs_new_action():
                    action_values = predict_action(
                        observation=observation_frame,
                        policy=policy,
                        device=get_safe_torch_device(policy.config.device),
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        use_amp=policy.config.use_amp,
                        task=single_task,
                        robot_type=robot.robot_type,
                    )
                    act_processed_policy = make_robot_action(action_values, dataset.features)
                    robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                    action_tensor = torch.tensor([robot_action_to_send[k] for k in action_keys])
                    interpolator.add(action_tensor)
                    ran_inference = True

                interp_action = interpolator.get()
                if interp_action is not None:
                    robot_action_to_send = {k: interp_action[i].item() for i, k in enumerate(action_keys)}
                    action_values = robot_action_to_send
                else:
                    continue
                is_record_frame = ran_inference
            else:
                action_values = predict_action(
                    observation=observation_frame,
                    policy=policy,
                    device=get_safe_torch_device(policy.config.device),
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    use_amp=policy.config.use_amp,
                    task=single_task,
                    robot_type=robot.robot_type,
                )
                act_processed_policy = make_robot_action(action_values, dataset.features)
                robot_action_to_send = robot_action_processor((act_processed_policy, obs))
                action_values = robot_action_to_send
        elif policy is None and isinstance(teleop, base_record.Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        elif policy is None and isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning("No policy or teleoperator provided, skipping action generation.")
            continue

        if vp_recorder is not None:
            action_values, robot_action_to_send = vp_recorder.apply_action_freeze(
                action_values=action_values,
                robot_action_to_send=robot_action_to_send,
                observation=obs_processed,
            )

        robot.send_action(robot_action_to_send)

        if dataset is not None and is_record_frame:
            frame_index = int(dataset.writer.episode_buffer["size"])
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            if vp_recorder is not None:
                vp_recorder.save_visual_prompt_copy(frame, dataset.features, frame_index)
                vp_recorder.record_gripper_proprio(
                    robot=robot,
                    observation=obs_processed,
                    action_values=action_values,
                    frame_index=frame_index,
                    timestamp_s=frame_index / fps,
                )
            dataset.add_frame(frame)
            if vp_recorder is not None:
                vp_recorder.drain_events(frame_index=frame_index, timestamp_s=frame_index / fps)

        if display_data:
            log_rerun_data(
                observation=obs_processed,
                action=action_values,
                compress_images=display_compressed_images,
            )

        dt_s = time.perf_counter() - start_loop_t
        sleep_time_s = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than target FPS ({fps} Hz)."
            )
        precise_sleep(max(sleep_time_s, 0.0))
        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(cfg: VPRecordConfig) -> LeRobotDataset:
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording_vp", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None
    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None
    vp_recorder = None

    try:
        if cfg.resume:
            num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                vcodec=cfg.dataset.vcodec,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                encoder_threads=cfg.dataset.encoder_threads,
            )

        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        interpolator = None
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )
            if cfg.interpolation_multiplier > 1:
                interpolator = ActionInterpolator(multiplier=cfg.interpolation_multiplier)
                logging.info(f"Action interpolation enabled: {cfg.interpolation_multiplier}x control rate")

        if cfg.vp.enabled:
            vp_recorder = VPEventRecorder(cfg.vp, dataset, cfg.dataset.repo_id)

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_vp_keyboard_listener(vp_recorder)

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. Consider --dataset.streaming_encoding=true "
                "--dataset.encoder_threads=2 for faster episode saving."
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                episode_index = int(dataset.meta.total_episodes)
                log_say(f"Recording episode {episode_index}", cfg.play_sounds)
                episode_task = cfg.dataset.single_task
                if vp_recorder is not None:
                    episode_task = vp_recorder.start_episode(
                        episode_index=episode_index,
                        task=cfg.dataset.single_task,
                        fps=cfg.dataset.fps,
                        robot_type=robot.name,
                    )

                record_loop_vp(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=episode_task,
                    display_data=cfg.display_data,
                    interpolator=interpolator,
                    display_compressed_images=display_compressed_images,
                    vp_recorder=vp_recorder,
                )
                if vp_recorder is not None:
                    vp_recorder.end_recording_window()

                if not events["stop_recording"] and (
                    (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
                ):
                    log_say("Reset the environment", cfg.play_sounds)
                    record_loop_vp(
                        robot=robot,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        teleop=teleop,
                        control_time_s=cfg.dataset.reset_time_s,
                        single_task=episode_task,
                        display_data=cfg.display_data,
                        vp_recorder=vp_recorder,
                    )

                if events["rerecord_episode"]:
                    log_say("Re-record episode", cfg.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    if vp_recorder is not None:
                        vp_recorder.discard_episode()
                    continue

                dataset.save_episode()
                if vp_recorder is not None:
                    vp_recorder.save_episode()
                recorded_episodes += 1
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)
        if vp_recorder is not None:
            vp_recorder.end_recording_window()
        if dataset:
            dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()
        if not is_headless() and listener:
            listener.stop()
        if dataset and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
