from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from vp_runtime_overlay import SAM3TextClient, draw_box, draw_crosshair, top_detection


def load_episode_meta(dataset_root: Path, episode_index: int) -> dict:
    meta_path = dataset_root / "meta" / "vp_events" / f"episode_{episode_index:06d}.json"
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_video_info(dataset_root: Path, episode_index: int) -> tuple[Path, float]:
    import pyarrow.parquet as pq

    table = pq.read_table(dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = table.slice(episode_index, 1).to_pydict()
    chunk_index = int(row["videos/observation.images.top/chunk_index"][0])
    file_index = int(row["videos/observation.images.top/file_index"][0])
    from_ts = float(row["videos/observation.images.top/from_timestamp"][0])
    video_path = (
        dataset_root
        / "videos"
        / "observation.images.top"
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4"
    )
    return video_path, from_ts


def extract_frame(video_path: Path, timestamp_s: float) -> tuple[bool, any]:
    cmd = [
        "/usr/bin/ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "-",
    ]
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return False, None
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame_bgr is not None, frame_bgr


def find_event(meta: dict, event_type: str) -> dict:
    for event in meta.get("events", []):
        if event.get("type") == event_type:
            return event
    raise ValueError(f"event {event_type!r} not found")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sam3-host", default="127.0.0.1")
    parser.add_argument("--sam3-port", type=int, default=10094)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = load_episode_meta(dataset_root, args.episode_index)
    video_path, episode_from_ts = episode_video_info(dataset_root, args.episode_index)

    pick_event = find_event(meta, "pick_phase_start")
    place_event = find_event(meta, "place_phase_start")

    pick_ts = episode_from_ts + float(pick_event["time_s"])
    place_ts = episode_from_ts + float(place_event["time_s"])

    ok_pick, pick_bgr = extract_frame(video_path, pick_ts)
    ok_place, place_bgr = extract_frame(video_path, place_ts)
    if not ok_pick:
        raise RuntimeError(f"failed to extract pick frame at {pick_ts:.3f}s from {video_path}")
    if not ok_place:
        raise RuntimeError(f"failed to extract place frame at {place_ts:.3f}s from {video_path}")

    pick_rgb = cv2.cvtColor(pick_bgr, cv2.COLOR_BGR2RGB)
    place_rgb = cv2.cvtColor(place_bgr, cv2.COLOR_BGR2RGB)

    target_object = meta["target_object"]
    target_location = meta["target_location"]

    client = SAM3TextClient(args.sam3_host, args.sam3_port, timeout_s=5.0)
    try:
        pick_result, pick_latency = client.segment(
            pick_rgb,
            target_object,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )
        place_obj_result, place_obj_latency = client.segment(
            place_rgb,
            target_object,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )
        place_box_result, place_box_latency = client.segment(
            place_rgb,
            target_location,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )
    finally:
        client.close()

    pick_det = top_detection(pick_result, pick_latency, target_object)
    place_obj_det = top_detection(place_obj_result, place_obj_latency, target_object)
    place_box_det = top_detection(place_box_result, place_box_latency, target_location)

    if pick_det.centroid is None:
        raise RuntimeError(f"failed to detect target object {target_object!r} on pick frame")
    if place_box_det.box is None:
        raise RuntimeError(f"failed to detect target location {target_location!r} on place frame")

    pick_overlay = pick_rgb.copy()
    draw_crosshair(pick_overlay, pick_det.centroid)

    place_overlay = place_rgb.copy()
    if place_obj_det.centroid is not None:
        draw_crosshair(place_overlay, place_obj_det.centroid)
    draw_box(place_overlay, place_box_det.box)

    pick_out = output_dir / f"episode_{args.episode_index:06d}_pick_crosshair.png"
    place_out = output_dir / f"episode_{args.episode_index:06d}_place_box.png"

    cv2.imwrite(str(pick_out), cv2.cvtColor(pick_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(place_out), cv2.cvtColor(place_overlay, cv2.COLOR_RGB2BGR))

    print(f"saved: {pick_out}")
    print(f"saved: {place_out}")


if __name__ == "__main__":
    main()
