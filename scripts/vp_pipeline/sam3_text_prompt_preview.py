#!/usr/bin/env python3
"""Preview VP-VLA-style text-prompt SAM3 overlays.

This script does not use Qwen or pixel coordinates. It sends text prompts
directly to a running VP-VLA SAM3 websocket server:

    image + "orange cube" -> SAM3 mask -> crosshair overlay
    image + "white basket" -> SAM3 mask/box -> bbox overlay
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import websockets.sync.client
from PIL import Image


VP_UTILITY_DIR = (
    Path(__file__).resolve().parents[2]
    / "VP-VLA"
    / "examples"
    / "Robocasa_tabletop"
    / "visual_prompt_utility"
)
sys.path.insert(0, str(VP_UTILITY_DIR))
from msgpack_utils import packb, unpackb  # noqa: E402


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


def draw_crosshair(image: np.ndarray, center_xy: list[float]) -> None:
    x, y = int(round(center_xy[0])), int(round(center_xy[1]))
    cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
    gap, length = 7, 16
    cv2.line(image, (x - gap - length, y), (x - gap, y), (0, 255, 0), 2)
    cv2.line(image, (x + gap, y), (x + gap + length, y), (0, 255, 0), 2)
    cv2.line(image, (x, y - gap - length), (x, y - gap), (0, 255, 0), 2)
    cv2.line(image, (x, y + gap), (x, y + gap + length), (0, 255, 0), 2)


def tint_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> np.ndarray:
    out = image.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[:, :] = np.array(color, dtype=np.float32)
    mask_bool = mask.astype(bool)
    out[mask_bool] = out[mask_bool] * (1.0 - alpha) + tint[mask_bool] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


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

    def segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        threshold: float,
        mask_threshold: float,
    ) -> tuple[dict[str, Any], float]:
        request = {
            "type": "segment",
            "request_id": f"seg_{time.time()}",
            "image": image,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }
        t0 = time.perf_counter()
        self.ws.send(packb(request))
        response = unpackb(self.ws.recv())
        elapsed = time.perf_counter() - t0
        if not response.get("ok", False):
            message = response.get("error", {}).get("message", "unknown SAM3 error")
            raise RuntimeError(f"SAM3 segmentation failed for {text_prompt!r}: {message}")
        return response.get("data", {}), elapsed


def top_instance(result: dict[str, Any]) -> tuple[np.ndarray | None, list[int] | None, float | None]:
    masks = np.asarray(result.get("masks", np.zeros((0,), dtype=np.uint8)))
    boxes = np.asarray(result.get("boxes", np.zeros((0, 4), dtype=np.float32)))
    scores = np.asarray(result.get("scores", np.zeros((0,), dtype=np.float32)))
    if masks.ndim < 3 or len(masks) == 0:
        return None, None, None
    idx = int(np.argmax(scores)) if len(scores) else 0
    mask = masks[idx].astype(bool)
    if boxes.ndim == 2 and len(boxes) > idx:
        box = [int(round(v)) for v in boxes[idx].tolist()]
    else:
        box = mask_bbox(mask)
    score = float(scores[idx]) if len(scores) > idx else None
    return mask, box, score


def save_mask(path: Path, mask: np.ndarray | None) -> str | None:
    if mask is None:
        return None
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return str(path)


def slug(text: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9가-힣_-]+", "_", text.strip()).strip("_") or "target"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target_object", required=True, help='Example: "orange cube"')
    parser.add_argument("--target_location", default=None, help='Example: "white basket"')
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10094)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask_threshold", type=float, default=0.5)
    parser.add_argument("--overlay_alpha", type=float, default=0.35)
    parser.add_argument("--timeout_s", type=float, default=150.0)
    parser.add_argument("--out_dir", default="outputs/sam3_text_prompt_preview")
    args = parser.parse_args()

    image_path = Path(args.image).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    overlay = image.copy()

    client = SAM3TextClient(args.host, args.port, args.timeout_s)
    print(f"[SAM3] connected to {client.uri}: {client.metadata}")

    meta: dict[str, Any] = {
        "image": str(image_path),
        "sam3_uri": client.uri,
        "sam3_metadata": client.metadata,
        "threshold": args.threshold,
        "mask_threshold": args.mask_threshold,
        "target_object": args.target_object,
        "target_location": args.target_location,
    }

    try:
        target_result, target_latency = client.segment(
            image,
            args.target_object,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
        )
        target_mask, target_box, target_score = top_instance(target_result)
        if target_mask is not None:
            overlay = tint_mask(overlay, target_mask, (255, 128, 0), args.overlay_alpha)
            centroid = mask_centroid(target_mask)
            if centroid is not None:
                draw_crosshair(overlay, centroid)
        else:
            centroid = None

        target_mask_path = out_dir / f"{image_path.stem}_{slug(args.target_object)}_target_mask.png"
        meta.update(
            {
                "target_latency_s": target_latency,
                "target_num_masks": int(target_result.get("num_masks", 0)),
                "target_score": target_score,
                "target_box": target_box,
                "target_mask_bbox": mask_bbox(target_mask),
                "target_mask_centroid": centroid,
                "target_mask_path": save_mask(target_mask_path, target_mask),
            }
        )

        place_mask = None
        if args.target_location:
            place_result, place_latency = client.segment(
                image,
                args.target_location,
                threshold=args.threshold,
                mask_threshold=args.mask_threshold,
            )
            place_mask, place_box, place_score = top_instance(place_result)
            if place_mask is not None:
                overlay = tint_mask(overlay, place_mask, (0, 128, 255), 0.22)
            if place_box is not None:
                x1, y1, x2, y2 = place_box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

            place_mask_path = out_dir / f"{image_path.stem}_{slug(args.target_location)}_place_mask.png"
            meta.update(
                {
                    "place_latency_s": place_latency,
                    "place_num_masks": int(place_result.get("num_masks", 0)),
                    "place_score": place_score,
                    "place_box": place_box,
                    "place_mask_bbox": mask_bbox(place_mask),
                    "place_mask_centroid": mask_centroid(place_mask),
                    "place_mask_path": save_mask(place_mask_path, place_mask),
                }
            )
    finally:
        client.close()

    suffix = slug(args.target_object)
    if args.target_location:
        suffix += f"_{slug(args.target_location)}"
    overlay_path = out_dir / f"{image_path.stem}_{suffix}_sam3_text_overlay.png"
    meta_path = out_dir / f"{image_path.stem}_{suffix}_sam3_text_meta.json"

    Image.fromarray(overlay).save(overlay_path)
    meta["overlay_path"] = str(overlay_path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[saved] {overlay_path}")
    print(f"[saved] {meta_path}")


if __name__ == "__main__":
    main()
