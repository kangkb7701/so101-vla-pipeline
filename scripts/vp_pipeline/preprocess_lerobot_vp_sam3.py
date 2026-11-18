#!/usr/bin/env python3
"""Create a VP-SAM3 overlay copy of a LeRobot dataset.

The source dataset is left untouched. The script copies ``data/`` and ``meta/``
to a new root and rewrites one camera video stream with visual prompts.

Overlay policies:
* ``phase``: before ``place_phase_start`` draw target-object crosshair only,
  then switch to target-location box only.
* ``dual``: draw target-object crosshair and target-location box together on
  every frame.

Frames are sampled from the source videos using the 10 Hz LeRobot parquet rows,
so the output video is aligned with the policy data even if the original camera
captured at a higher FPS before LeRobot encoding.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets.sync.client


VP_UTILITY_DIR = (
    Path(__file__).resolve().parents[3]
    / "VP-VLA"
    / "examples"
    / "Robocasa_tabletop"
    / "visual_prompt_utility"
)
sys.path.insert(0, str(VP_UTILITY_DIR))

pd = None
packb = None
unpackb = None
decode_video_frames = None


def load_runtime_dependencies() -> None:
    global pd, packb, unpackb, decode_video_frames
    try:
        import pandas as _pd
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "pandas is required to read LeRobot parquet metadata. "
            "Install it in this environment, or run from the LeRobot environment "
            "after installing msgpack there."
        ) from exc

    try:
        from msgpack_utils import packb as _packb, unpackb as _unpackb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "msgpack is required for the SAM3 websocket protocol. "
            "Install msgpack in this environment, or run from the SAM3 environment "
            "after installing pandas/pyarrow there."
        ) from exc

    try:
        from lerobot.datasets.video_utils import decode_video_frames as _decode_video_frames
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "lerobot.datasets.video_utils.decode_video_frames is required to decode LeRobot AV1 videos. "
            "Run this script from the LeRobot environment."
        ) from exc

    pd = _pd
    packb = _packb
    unpackb = _unpackb
    decode_video_frames = _decode_video_frames


@dataclass(frozen=True)
class Detection:
    mask: np.ndarray | None
    box: list[int] | None
    score: float | None
    centroid: list[float] | None
    latency_s: float
    num_masks: int
    prompt: str
    fallback: bool = False


class SAM3TextClient:
    def __init__(self, host: str, port: int, timeout_s: float) -> None:
        self.uri = f"ws://{host}:{port}"
        self.ws = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
            ping_interval=None,
            ping_timeout=None,
        )
        self.metadata = unpackb(self.ws.recv())

    def close(self) -> None:
        self.ws.close()

    def segment(self, image_rgb: np.ndarray, prompt: str, threshold: float, mask_threshold: float) -> tuple[dict[str, Any], float]:
        request = {
            "type": "segment",
            "request_id": f"seg_{time.time()}",
            "image": image_rgb,
            "text_prompt": prompt,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }
        t0 = time.perf_counter()
        self.ws.send(packb(request))
        response = unpackb(self.ws.recv())
        elapsed = time.perf_counter() - t0
        if not response.get("ok", False):
            message = response.get("error", {}).get("message", "unknown SAM3 error")
            raise RuntimeError(f"SAM3 segmentation failed for {prompt!r}: {message}")
        return response.get("data", {}), elapsed


def mask_bbox(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid(mask: np.ndarray | None) -> list[float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask.astype(bool))
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def top_detection(result: dict[str, Any], latency_s: float, prompt: str) -> Detection:
    masks = np.asarray(result.get("masks", np.zeros((0,), dtype=np.uint8)))
    boxes = np.asarray(result.get("boxes", np.zeros((0, 4), dtype=np.float32)))
    scores = np.asarray(result.get("scores", np.zeros((0,), dtype=np.float32)))
    num_masks = int(result.get("num_masks", len(masks) if masks.ndim >= 3 else 0))
    if masks.ndim < 3 or len(masks) == 0:
        return Detection(None, None, None, None, latency_s, num_masks, prompt)

    idx = int(np.argmax(scores)) if len(scores) else 0
    mask = masks[idx].astype(bool)
    if boxes.ndim == 2 and len(boxes) > idx:
        box = [int(round(v)) for v in boxes[idx].tolist()]
    else:
        box = mask_bbox(mask)
    score = float(scores[idx]) if len(scores) > idx else None
    return Detection(mask, box, score, mask_centroid(mask), latency_s, num_masks, prompt)


def draw_crosshair(image_rgb: np.ndarray, center_xy: list[float], line_length: int, gap: int, thickness: int) -> None:
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(image_rgb, (x, y), max(2, thickness + 1), (255, 0, 0), -1)
    cv2.line(image_rgb, (x - gap - line_length, y), (x - gap, y), (0, 255, 0), thickness)
    cv2.line(image_rgb, (x + gap, y), (x + gap + line_length, y), (0, 255, 0), thickness)
    cv2.line(image_rgb, (x, y - gap - line_length), (x, y - gap), (0, 255, 0), thickness)
    cv2.line(image_rgb, (x, y + gap), (x, y + gap + line_length), (0, 255, 0), thickness)


def draw_box(image_rgb: np.ndarray, box: list[int], thickness: int) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (255, 0, 0), thickness)


def tint_mask(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image_rgb.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[:, :] = np.asarray(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = out[mask_bool] * (1.0 - alpha) + tint[mask_bool] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def segment_top(client: SAM3TextClient, image_rgb: np.ndarray, prompt: str, threshold: float, mask_threshold: float) -> Detection:
    result, latency = client.segment(image_rgb, prompt, threshold=threshold, mask_threshold=mask_threshold)
    return top_detection(result, latency, prompt)


def load_episode_metadata(root: Path) -> pd.DataFrame:
    parts = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path)
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta' / 'episodes'}")
    return pd.concat(parts, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


def load_data_rows(root: Path) -> pd.DataFrame:
    parts = []
    for path in sorted((root / "data").glob("chunk-*/*.parquet")):
        df = pd.read_parquet(path, columns=["episode_index", "frame_index", "timestamp", "index"])
        parts.append(df)
    if not parts:
        raise FileNotFoundError(f"No data parquet files found under {root / 'data'}")
    return pd.concat(parts, ignore_index=True).sort_values("index").reset_index(drop=True)


def load_vp_event(root: Path, episode_index: int) -> dict[str, Any]:
    path = root / "meta" / "vp_events" / f"episode_{episode_index:06d}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing VP event sidecar: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def place_phase_frame(vp_event: dict[str, Any], fallback_length: int) -> int:
    for event in vp_event.get("events", []):
        if event.get("type") == "place_phase_start":
            return int(event.get("frame", fallback_length))
    return fallback_length


def read_frame_at(video_path: Path, timestamp_s: float, tolerance_s: float, backend: str) -> np.ndarray:
    frames = decode_video_frames(video_path, [timestamp_s], tolerance_s=tolerance_s, backend=backend)
    frame = frames[0].detach().cpu()
    if frame.ndim != 3:
        raise RuntimeError(f"Unexpected decoded frame shape: {tuple(frame.shape)}")
    # LeRobot returns C,H,W float32 in [0, 1].
    if frame.shape[0] in (1, 3):
        frame = frame.permute(1, 2, 0)
    arr = frame.numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return arr


def ensure_clean_output_root(input_root: Path, output_root: Path, overwrite: bool) -> None:
    if input_root.resolve() == output_root.resolve():
        raise ValueError("output_root must be different from dataset_root")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists. Use --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def copy_dataset_shell(input_root: Path, output_root: Path, video_key: str, copy_other_videos: bool) -> None:
    shutil.copytree(input_root / "data", output_root / "data")
    shutil.copytree(input_root / "meta", output_root / "meta")

    src_videos = input_root / "videos"
    dst_videos = output_root / "videos"
    dst_videos.mkdir(parents=True, exist_ok=True)
    selected_dir_name = video_key

    for src_camera_dir in sorted(src_videos.glob("*")):
        if not src_camera_dir.is_dir():
            continue
        dst_camera_dir = dst_videos / src_camera_dir.name
        if src_camera_dir.name == selected_dir_name:
            dst_camera_dir.mkdir(parents=True, exist_ok=True)
            continue
        if copy_other_videos:
            shutil.copytree(src_camera_dir, dst_camera_dir)
        else:
            for src_video in sorted(src_camera_dir.glob("chunk-*/*.mp4")):
                rel = src_video.relative_to(src_camera_dir)
                dst_video = dst_camera_dir / rel
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                dst_video.symlink_to(src_video.resolve())


def update_output_info(output_root: Path, video_key: str, fps: int, codec: str) -> None:
    info_path = output_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(video_key)
    if feature and "info" in feature:
        feature["info"]["video.fps"] = fps
        feature["info"]["video.codec"] = codec
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_episode_indices(value: str | None, available: list[int]) -> list[int]:
    if not value:
        return available
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            selected.update(range(int(start_s), int(end_s) + 1))
        else:
            selected.add(int(part))
    available_set = set(available)
    missing = sorted(selected - available_set)
    if missing:
        raise ValueError(f"Requested episodes not found: {missing}")
    return [ep for ep in available if ep in selected]


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, help="Source LeRobot dataset root")
    parser.add_argument("--output_root", required=True, help="New dataset root with overlaid video stream")
    parser.add_argument("--camera", default="observation.images.top", help="Video feature to overlay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--timeout_s", type=float, default=150.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--target_fps", type=int, default=10)
    parser.add_argument("--video_backend", default="torchcodec", choices=["torchcodec", "pyav", "video_reader"])
    parser.add_argument("--decode_tolerance_s", type=float, default=0.06)
    parser.add_argument("--output_codec", default="mp4v")
    parser.add_argument("--place_mode", choices=["first_frame", "place_frame", "every_frame"], default="first_frame")
    parser.add_argument("--overlay_mode", choices=["phase", "dual"], default="phase")
    parser.add_argument("--episode_indices", default=None, help='Comma/range list, e.g. "0,1,12-23". Default: all')
    parser.add_argument("--max_frames_per_episode", type=int, default=None, help="Debug limit")
    parser.add_argument("--fallback_last_target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fallback_last_place", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tint_masks", action="store_true")
    parser.add_argument("--crosshair_line_length", type=int, default=16)
    parser.add_argument("--crosshair_gap", type=int, default=7)
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--copy_other_videos", action="store_true", help="Copy non-overlaid cameras instead of symlinking")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    load_runtime_dependencies()

    input_root = Path(args.dataset_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    video_key = args.camera

    episodes = load_episode_metadata(input_root)
    data = load_data_rows(input_root)
    available_eps = [int(x) for x in episodes["episode_index"].tolist()]
    selected_eps = parse_episode_indices(args.episode_indices, available_eps)
    selected_set = set(selected_eps)

    ensure_clean_output_root(input_root, output_root, args.overwrite)
    copy_dataset_shell(input_root, output_root, video_key, args.copy_other_videos)
    update_output_info(output_root, video_key, args.target_fps, args.output_codec)

    client = SAM3TextClient(args.host, args.port, args.timeout_s)
    print(f"[SAM3] connected to {client.uri}: {client.metadata}")

    overlay_meta_dir = output_root / "meta" / "vp_sam3_overlay"
    overlay_meta_dir.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*args.output_codec)
    episode_summaries: list[dict[str, Any]] = []

    try:
        chunk_col = f"videos/{video_key}/chunk_index"
        file_col = f"videos/{video_key}/file_index"
        from_col = f"videos/{video_key}/from_timestamp"

        for (chunk_idx, file_idx), file_eps_df in episodes.groupby([chunk_col, file_col], sort=True):
            file_eps_df = file_eps_df[file_eps_df["episode_index"].isin(selected_set)]
            if file_eps_df.empty:
                continue

            src_video = input_root / "videos" / video_key / f"chunk-{int(chunk_idx):03d}" / f"file-{int(file_idx):03d}.mp4"
            dst_video = output_root / "videos" / video_key / f"chunk-{int(chunk_idx):03d}" / f"file-{int(file_idx):03d}.mp4"
            dst_video.parent.mkdir(parents=True, exist_ok=True)

            first_ep_row = file_eps_df.sort_values("episode_index").iloc[0]
            first_rgb_for_shape = read_frame_at(src_video, float(first_ep_row[from_col]), args.decode_tolerance_s, args.video_backend)
            height, width = first_rgb_for_shape.shape[:2]
            writer = cv2.VideoWriter(str(dst_video), fourcc, args.target_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open output writer: {dst_video}")

            print(f"[video] {src_video} -> {dst_video} backend={args.video_backend}")

            for _, ep_row in file_eps_df.sort_values("episode_index").iterrows():
                ep_idx = int(ep_row["episode_index"])
                ep_data = data[data["episode_index"] == ep_idx].sort_values("frame_index")
                if args.max_frames_per_episode is not None:
                    ep_data = ep_data.head(args.max_frames_per_episode)
                if ep_data.empty:
                    raise RuntimeError(f"No data rows for episode {ep_idx}")

                vp_event = load_vp_event(input_root, ep_idx)
                target_prompt = vp_event.get("target_object") or "orange cube"
                place_prompt = vp_event.get("target_location")
                if not place_prompt:
                    raise RuntimeError(f"Episode {ep_idx} has no target_location in VP event sidecar")

                phase_switch = place_phase_frame(vp_event, fallback_length=int(ep_row["length"]))
                ep_video_from = float(ep_row[from_col])

                last_target: Detection | None = None
                last_place: Detection | None = None
                records: list[dict[str, Any]] = []

                if args.place_mode == "first_frame":
                    first_rgb = read_frame_at(src_video, ep_video_from, args.decode_tolerance_s, args.video_backend)
                    last_place = segment_top(client, first_rgb, place_prompt, args.threshold, args.mask_threshold)

                print(
                    f"[episode {ep_idx:06d}] target={target_prompt!r} place={place_prompt!r} "
                    f"switch_frame={phase_switch} frames={len(ep_data)}"
                )

                for _, frame_row in ep_data.iterrows():
                    local_frame = int(frame_row["frame_index"])
                    timestamp_s = ep_video_from + (local_frame / float(args.target_fps))
                    image_rgb = read_frame_at(src_video, timestamp_s, args.decode_tolerance_s, args.video_backend)
                    overlay_rgb = image_rgb.copy()

                    phase = "pick" if local_frame < phase_switch else "place"
                    active_det: Detection | None = None
                    target_det: Detection | None = None
                    place_det: Detection | None = None

                    target_det = segment_top(client, image_rgb, target_prompt, args.threshold, args.mask_threshold)
                    if target_det.centroid is None and args.fallback_last_target and last_target is not None:
                        target_det = Detection(
                            None,
                            last_target.box,
                            last_target.score,
                            last_target.centroid,
                            target_det.latency_s,
                            target_det.num_masks,
                            target_prompt,
                            fallback=True,
                        )
                    elif target_det.centroid is not None:
                        last_target = target_det

                    if args.place_mode == "every_frame":
                        last_place = segment_top(client, image_rgb, place_prompt, args.threshold, args.mask_threshold)
                    elif args.place_mode == "place_frame" and phase == "place" and last_place is None:
                        last_place = segment_top(client, image_rgb, place_prompt, args.threshold, args.mask_threshold)

                    place_det = last_place
                    if (place_det is None or place_det.box is None) and args.fallback_last_place and last_place is not None:
                        place_det = Detection(
                            None,
                            last_place.box,
                            last_place.score,
                            last_place.centroid,
                            0.0,
                            0,
                            place_prompt,
                            fallback=True,
                        )

                    draw_target = args.overlay_mode == "dual" or phase == "pick"
                    draw_place = args.overlay_mode == "dual" or phase == "place"

                    if draw_target and target_det is not None and target_det.centroid is not None:
                        if args.tint_masks and target_det.mask is not None:
                            overlay_rgb = tint_mask(overlay_rgb, target_det.mask, (255, 128, 0), 0.25)
                        draw_crosshair(
                            overlay_rgb,
                            target_det.centroid,
                            line_length=args.crosshair_line_length,
                            gap=args.crosshair_gap,
                            thickness=args.thickness,
                        )

                    if draw_place and place_det is not None and place_det.box is not None:
                        if args.tint_masks and place_det.mask is not None:
                            overlay_rgb = tint_mask(overlay_rgb, place_det.mask, (0, 128, 255), 0.18)
                        draw_box(overlay_rgb, place_det.box, thickness=args.thickness)

                    active_det = target_det if phase == "pick" else place_det

                    writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                    records.append(
                        {
                            "frame_index": local_frame,
                            "timestamp_s": float(frame_row["timestamp"]),
                            "video_timestamp_s": timestamp_s,
                            "phase": phase,
                            "overlay_mode": args.overlay_mode,
                            "prompt": active_det.prompt if active_det else None,
                            "num_masks": active_det.num_masks if active_det else 0,
                            "score": active_det.score if active_det else None,
                            "box": active_det.box if active_det else None,
                            "centroid": active_det.centroid if active_det else None,
                            "latency_s": active_det.latency_s if active_det else 0.0,
                            "fallback": active_det.fallback if active_det else False,
                            "target_prompt": target_prompt,
                            "target_box": target_det.box if target_det else None,
                            "target_centroid": target_det.centroid if target_det else None,
                            "target_score": target_det.score if target_det else None,
                            "target_latency_s": target_det.latency_s if target_det else 0.0,
                            "target_fallback": target_det.fallback if target_det else False,
                            "place_prompt": place_prompt,
                            "place_box": place_det.box if place_det else None,
                            "place_centroid": place_det.centroid if place_det else None,
                            "place_score": place_det.score if place_det else None,
                            "place_latency_s": place_det.latency_s if place_det else 0.0,
                            "place_fallback": place_det.fallback if place_det else False,
                        }
                    )

                ep_summary = {
                    "episode_index": ep_idx,
                    "video_key": video_key,
                    "source_video": str(src_video),
                    "output_video": str(dst_video),
                    "target_object": target_prompt,
                    "target_location": place_prompt,
                    "place_phase_start_frame": phase_switch,
                    "num_output_frames": len(records),
                    "place_mode": args.place_mode,
                    "overlay_mode": args.overlay_mode,
                    "records": records,
                }
                (overlay_meta_dir / f"episode_{ep_idx:06d}.json").write_text(
                    json.dumps(ep_summary, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                episode_summaries.append({k: v for k, v in ep_summary.items() if k != "records"})

            writer.release()

    finally:
        client.close()

    summary = {
        "source_root": str(input_root),
        "output_root": str(output_root),
        "video_key": video_key,
        "target_fps": args.target_fps,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "place_mode": args.place_mode,
        "overlay_mode": args.overlay_mode,
        "selected_episodes": selected_eps,
        "episodes": episode_summaries,
    }
    (overlay_meta_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[done] wrote overlay dataset to {output_root}")
    print(f"[done] metadata: {overlay_meta_dir}")


if __name__ == "__main__":
    run()
