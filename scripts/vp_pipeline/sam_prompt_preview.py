#!/usr/bin/env python3
"""Run SAM on one image with point and/or box prompts and save VP preview artifacts."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


def parse_floats(value, expected):
    if value is None or value == "":
        return None
    vals = [float(x) for x in str(value).replace(",", " ").split() if x]
    if len(vals) != expected:
        raise ValueError(f"Expected {expected} values, got {len(vals)} from {value!r}")
    return np.array(vals, dtype=np.float32)


def mask_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def auto_orange_point(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([5, 45, 50]), np.array([35, 255, 255]))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 10:
            candidates.append((area, float(cents[i][0]), float(cents[i][1])))
    if not candidates:
        return None
    _, x, y = max(candidates)
    return np.array([x, y], dtype=np.float32)


def auto_white_box(image_rgb):
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 145]), np.array([179, 90, 255]))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    candidates = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x, y, w, h = [int(stats[i, k]) for k in range(4)]
        if area >= 20:
            candidates.append((area, x, y, x + w - 1, y + h - 1))
    if not candidates:
        return None
    _, x1, y1, x2, y2 = max(candidates)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def draw_crosshair(image, point):
    out = image.copy()
    x, y = int(round(point[0])), int(round(point[1]))
    cv2.circle(out, (x, y), 3, (255, 0, 0), -1)
    gap, length = 7, 16
    cv2.line(out, (x - gap - length, y), (x - gap, y), (0, 255, 0), 2)
    cv2.line(out, (x + gap, y), (x + gap + length, y), (0, 255, 0), 2)
    cv2.line(out, (x, y - gap - length), (x, y - gap), (0, 255, 0), 2)
    cv2.line(out, (x, y + gap), (x, y + gap + length), (0, 255, 0), 2)
    return out


def apply_mask_tint(image, mask, color, alpha):
    out = image.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[:, :] = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + tint[mask] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="models/sam/sam_vit_b_01ec64.pth")
    ap.add_argument("--model_type", default="vit_b")
    ap.add_argument("--out_dir", default="outputs/vp_visual_prompt_preview/sam_preview")
    ap.add_argument("--target_point", default="auto_orange")
    ap.add_argument("--place_box", default="auto_white")
    args = ap.parse_args()

    image_path = Path(args.image).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    image = np.asarray(Image.open(image_path).convert("RGB"))
    target_point = auto_orange_point(image) if args.target_point == "auto_orange" else parse_floats(args.target_point, 2)
    place_box = auto_white_box(image) if args.place_box == "auto_white" else parse_floats(args.place_box, 4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image)

    target_mask = None
    target_score = None
    if target_point is not None:
        masks, scores, _ = predictor.predict(
            point_coords=target_point.reshape(1, 2),
            point_labels=np.array([1], dtype=np.int32),
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        target_mask = masks[best].astype(bool)
        target_score = float(scores[best])

    place_mask = None
    place_score = None
    if place_box is not None:
        masks, scores, _ = predictor.predict(box=place_box, multimask_output=True)
        best = int(np.argmax(scores))
        place_mask = masks[best].astype(bool)
        place_score = float(scores[best])

    overlay = image.copy()
    if place_mask is not None:
        overlay = apply_mask_tint(overlay, place_mask, (0, 128, 255), 0.28)
    if target_mask is not None:
        overlay = apply_mask_tint(overlay, target_mask, (255, 128, 0), 0.35)
    if place_box is not None:
        x1, y1, x2, y2 = [int(round(v)) for v in place_box]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    if target_point is not None:
        overlay = draw_crosshair(overlay, target_point)

    stem = image_path.stem
    overlay_path = out_dir / f"{stem}_sam_vp_overlay.png"
    target_mask_path = out_dir / f"{stem}_target_mask.png"
    place_mask_path = out_dir / f"{stem}_place_mask.png"
    meta_path = out_dir / f"{stem}_sam_vp_meta.json"

    Image.fromarray(overlay).save(overlay_path)
    if target_mask is not None:
        Image.fromarray((target_mask.astype(np.uint8) * 255)).save(target_mask_path)
    if place_mask is not None:
        Image.fromarray((place_mask.astype(np.uint8) * 255)).save(place_mask_path)

    meta = {
        "image": str(image_path),
        "checkpoint": args.checkpoint,
        "device": device,
        "target": "orange cube",
        "place": "white basket",
        "target_point_xy": None if target_point is None else [float(target_point[0]), float(target_point[1])],
        "target_score": target_score,
        "target_mask_bbox": None if target_mask is None else mask_bbox(target_mask),
        "place_box_xyxy": None if place_box is None else [float(x) for x in place_box],
        "place_score": place_score,
        "place_mask_bbox": None if place_mask is None else mask_bbox(place_mask),
        "overlay_path": str(overlay_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
