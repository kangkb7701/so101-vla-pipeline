#!/usr/bin/env python3
"""Convert completed SAM3 overlay metadata into VP-VLA visual-prompt NPZ files.

No SAM3 inference is performed here. Target centroids become small masks used
for crosshairs, and place boxes become valid filled masks plus boxes. Original
top and wrist videos remain the VP-VLA image sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_episode_indices(value: str | None, available: list[int]) -> list[int]:
    if not value:
        return available
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    missing = sorted(selected - set(available))
    if missing:
        raise ValueError(f"Requested episodes not found: {missing}")
    return [episode for episode in available if episode in selected]


def load_episode_lengths(dataset_root: Path) -> dict[int, int]:
    paths = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No episode metadata under {dataset_root}")
    episodes = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    return {int(row["episode_index"]): int(row["length"]) for _, row in episodes.iterrows()}


def load_vp_event(dataset_root: Path, episode_index: int) -> dict[str, Any]:
    path = dataset_root / "meta" / "vp_events" / f"episode_{episode_index:06d}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def draw_disk(mask: np.ndarray, centroid: Any, radius: int) -> bool:
    if centroid is None or len(centroid) != 2:
        return False
    x, y = (int(round(float(value))) for value in centroid)
    height, width = mask.shape
    if not (0 <= x < width and 0 <= y < height):
        return False
    yy, xx = np.ogrid[:height, :width]
    mask[(xx - x) ** 2 + (yy - y) ** 2 <= radius**2] = 1
    return True


def fill_box(mask: np.ndarray, box: Any) -> tuple[bool, np.ndarray]:
    output = np.zeros(4, dtype=np.float32)
    if box is None or len(box) != 4:
        return False, output
    height, width = mask.shape
    x1, y1, x2, y2 = (int(round(float(value))) for value in box)
    x1, x2 = sorted((max(0, min(width - 1, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height - 1, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return False, output
    mask[y1:y2, x1:x2] = 1
    output[:] = [x1, y1, x2, y2]
    return True, output


def process_episode(
    dataset_root: Path, overlay_root: Path, output_path: Path, episode_index: int,
    num_frames: int, image_height: int, image_width: int, fps: float,
    target_mask_radius: int, min_coverage: float, overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        return {"episode_index": episode_index, "status": "skipped", "path": str(output_path)}

    overlay_path = overlay_root / "meta" / "vp_sam3_overlay" / f"episode_{episode_index:06d}.json"
    if not overlay_path.exists():
        raise FileNotFoundError(overlay_path)
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    records = {int(record["frame_index"]): record for record in overlay.get("records", [])}
    vp_event = load_vp_event(dataset_root, episode_index)

    target_masks = np.zeros((num_frames, image_height, image_width), dtype=np.uint8)
    target_boxes = np.zeros((num_frames, 4), dtype=np.float32)
    target_scores = np.zeros(num_frames, dtype=np.float32)
    location_masks = np.zeros_like(target_masks)
    location_boxes = np.zeros_like(target_boxes)
    location_scores = np.zeros_like(target_scores)

    target_valid = 0
    place_valid = 0
    last_target: dict[str, Any] | None = None
    last_place: dict[str, Any] | None = None
    for frame_index in range(num_frames):
        record = records.get(frame_index, {})
        if record.get("target_centroid") is not None:
            last_target = record
        if record.get("place_box") is not None:
            last_place = record
        if last_target is not None and draw_disk(
            target_masks[frame_index], last_target.get("target_centroid"), target_mask_radius
        ):
            target_valid += 1
            target_boxes[frame_index] = np.asarray(
                last_target.get("target_box") or [0, 0, 0, 0], dtype=np.float32
            )
            target_scores[frame_index] = float(last_target.get("target_score") or 0.0)
        if last_place is not None:
            valid, clipped_box = fill_box(location_masks[frame_index], last_place.get("place_box"))
            if valid:
                place_valid += 1
                location_boxes[frame_index] = clipped_box
                location_scores[frame_index] = float(last_place.get("place_score") or 0.0)

    target_coverage = target_valid / num_frames
    place_coverage = place_valid / num_frames
    if target_coverage < min_coverage or place_coverage < min_coverage:
        raise RuntimeError(
            f"Episode {episode_index}: insufficient prompt coverage "
            f"target={target_coverage:.3f}, place={place_coverage:.3f}"
        )

    task_description = str(vp_event.get("language_instruction") or "").strip()
    target_object = str(vp_event.get("target_object") or overlay.get("target_object") or "").strip()
    target_location = str(vp_event.get("target_location") or overlay.get("target_location") or "").strip()
    if not task_description or not target_object or not target_location:
        raise ValueError(f"Episode {episode_index}: missing task/object/location labels")
    vlm_results = [{
        "frame": 0, "subtask": task_description, "subtask_idx": 0, "decision": "continue",
        "target_object": target_object, "target_location": target_location,
        "reasoning": "Dataset labels; target crosshair and place box active together.",
        "gripper_state_changed": False,
    }]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        episode_index=np.array(episode_index), task_description=np.array(task_description),
        num_frames=np.array(num_frames), fps=np.array(fps),
        target_object=np.array(target_object), target_location=np.array(target_location),
        target_masks=target_masks, target_boxes=target_boxes, target_scores=target_scores,
        location_masks=location_masks, location_boxes=location_boxes, location_scores=location_scores,
        vlm_results=np.array(json.dumps(vlm_results)),
    )
    return {
        "episode_index": episode_index, "status": "success", "path": str(output_path),
        "num_frames": num_frames, "target_coverage": target_coverage,
        "place_coverage": place_coverage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True, help="Prepared VP-VLA LeRobot dataset")
    parser.add_argument("--overlay_root", required=True, help="Completed SAM3 overlay dataset")
    parser.add_argument("--output_dir", required=True, help="Root containing <dataset_name>/episode_*.npz")
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--episode_indices", default=None, help='Comma/range list, e.g. "0-179"')
    parser.add_argument("--target_mask_radius", type=int, default=3)
    parser.add_argument("--min_coverage", type=float, default=0.98)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    overlay_root = Path(args.overlay_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    dataset_name = args.dataset_name or dataset_root.name
    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    image_height, image_width = info["features"]["observation.images.top"]["shape"][:2]
    fps = float(info["fps"])
    episode_lengths = load_episode_lengths(dataset_root)
    selected = parse_episode_indices(args.episode_indices, sorted(episode_lengths))

    results = []
    dataset_output = output_dir / dataset_name
    for episode_index in selected:
        output_path = dataset_output / f"episode_{episode_index:06d}.npz"
        result = process_episode(
            dataset_root, overlay_root, output_path, episode_index, episode_lengths[episode_index],
            int(image_height), int(image_width), fps, args.target_mask_radius,
            args.min_coverage, args.overwrite,
        )
        results.append(result)
        print(f"[episode {episode_index:06d}] {result}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset_root": str(dataset_root), "overlay_root": str(overlay_root),
        "dataset_name": dataset_name, "output_dir": str(dataset_output),
        "primary_view_only": True, "simultaneous_target_and_place": True,
        "target_mask_radius": args.target_mask_radius, "min_coverage": args.min_coverage,
        "selected_episodes": selected, "results": results,
    }
    (output_dir / f"{dataset_name}_summary.json").write_text(
        json.dumps(summary, indent=4) + "\n", encoding="utf-8"
    )
    print(f"[done] visual prompt NPZ root: {dataset_output}")


if __name__ == "__main__":
    main()
