#!/usr/bin/env python3
"""Interactive point-click labeling and SAM preview generation for VP-VLA prompts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


@dataclass
class LabelState:
    target_point: tuple[int, int] | None = None
    place_point: tuple[int, int] | None = None
    place_box: tuple[int, int, int, int] | None = None
    drag_start: tuple[int, int] | None = None
    drag_current: tuple[int, int] | None = None
    mode: str = "target"


def collect_images(image: str | None, image_dir: str | None, pattern: str) -> list[Path]:
    if image:
        paths = [Path(image).expanduser()]
    elif image_dir:
        paths = sorted(Path(image_dir).expanduser().glob(pattern))
    else:
        raise ValueError("Provide --image or --image_dir")
    paths = [p for p in paths if p.exists() and p.is_file()]
    if not paths:
        raise FileNotFoundError("No images found")
    return paths


def make_dirs(out_dir: Path) -> dict[str, Path]:
    dirs = {
        "labels": out_dir / "labels",
        "overlays": out_dir / "overlays",
        "masks": out_dir / "masks",
        "meta": out_dir / "meta",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def clamp_xy(x: int, y: int, width: int, height: int) -> tuple[int, int]:
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


def normalize_box(box: tuple[int, int, int, int] | None) -> tuple[int, int, int, int] | None:
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def mask_bbox(mask: np.ndarray | None) -> list[int] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_centroid(mask: np.ndarray | None) -> list[float] | None:
    if mask is None:
        return None
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.mean()), float(ys.mean())]


def draw_crosshair(image: np.ndarray, point: tuple[int, int], scale: float = 1.0) -> None:
    x, y = point
    thickness = max(1, int(round(2 * scale)))
    radius = max(2, int(round(3 * scale)))
    gap = max(4, int(round(7 * scale)))
    length = max(10, int(round(16 * scale)))
    cv2.circle(image, (x, y), radius, (255, 0, 0), -1)
    cv2.line(image, (x - gap - length, y), (x - gap, y), (0, 255, 0), thickness)
    cv2.line(image, (x + gap, y), (x + gap + length, y), (0, 255, 0), thickness)
    cv2.line(image, (x, y - gap - length), (x, y - gap), (0, 255, 0), thickness)
    cv2.line(image, (x, y + gap), (x, y + gap + length), (0, 255, 0), thickness)


def draw_point_marker(
    image: np.ndarray,
    point: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 1.0,
) -> None:
    x, y = point
    radius = max(3, int(round(5 * scale)))
    thickness = max(1, int(round(2 * scale)))
    cv2.circle(image, (x, y), radius, color, thickness)
    cv2.line(image, (x - radius - 3, y), (x + radius + 3, y), color, thickness)
    cv2.line(image, (x, y - radius - 3), (x, y + radius + 3), color, thickness)


def tint_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = image.copy().astype(np.float32)
    tint = np.zeros_like(out)
    tint[:, :] = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + tint[mask] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_display(image: np.ndarray, state: LabelState, display_scale: float, filename: str) -> np.ndarray:
    display = cv2.resize(image, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
    if state.target_point is not None:
        px = int(round(state.target_point[0] * display_scale))
        py = int(round(state.target_point[1] * display_scale))
        draw_crosshair(display, (px, py), scale=display_scale)
    if state.place_point is not None:
        px = int(round(state.place_point[0] * display_scale))
        py = int(round(state.place_point[1] * display_scale))
        draw_point_marker(display, (px, py), color=(0, 128, 255), scale=display_scale)
    box = normalize_box(state.place_box)
    if box is not None:
        x1, y1, x2, y2 = [int(round(v * display_scale)) for v in box]
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), max(1, int(round(2 * display_scale))))
    if state.drag_start is not None and state.drag_current is not None:
        x1, y1 = state.drag_start
        x2, y2 = state.drag_current
        x1, y1, x2, y2 = [int(round(v * display_scale)) for v in (x1, y1, x2, y2)]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), max(1, int(round(1 * display_scale))))

    status = (
        f"{filename} | mode={state.mode} | "
        "t:target point, l:place point, b:optional box, enter:SAM+save, r:reset, n/p:next/prev, q:quit"
    )
    cv2.rectangle(display, (0, 0), (display.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(display, status, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return display


def save_label(label_path: Path, image_path: Path, state: LabelState, args: argparse.Namespace) -> dict:
    box = normalize_box(state.place_box)
    label = {
        "image": str(image_path),
        "target_object": args.target_object,
        "target_point": None if state.target_point is None else [int(state.target_point[0]), int(state.target_point[1])],
        "place": args.place,
        "place_point": None if state.place_point is None else [int(state.place_point[0]), int(state.place_point[1])],
        "place_box_prompt": None if box is None else [int(v) for v in box],
        "coord_format": "pixel",
    }
    label_path.write_text(json.dumps(label, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return label


def load_existing_label(label_path: Path, state: LabelState) -> None:
    if not label_path.exists():
        return
    label = json.loads(label_path.read_text(encoding="utf-8"))
    if label.get("target_point") is not None:
        state.target_point = tuple(map(int, label["target_point"]))
    if label.get("place_point") is not None:
        state.place_point = tuple(map(int, label["place_point"]))
    if label.get("place_box_prompt") is not None:
        state.place_box = tuple(map(int, label["place_box_prompt"]))
    elif label.get("place_box") is not None:
        state.place_box = tuple(map(int, label["place_box"]))


def predict_from_point(predictor: SamPredictor, point_xy: tuple[int, int]) -> tuple[np.ndarray, float]:
    point = np.array(point_xy, dtype=np.float32).reshape(1, 2)
    masks, scores, _ = predictor.predict(
        point_coords=point,
        point_labels=np.array([1], dtype=np.int32),
        multimask_output=True,
    )
    best = int(np.argmax(scores))
    return masks[best].astype(bool), float(scores[best])


def run_sam(
    predictor: SamPredictor,
    image: np.ndarray,
    image_path: Path,
    state: LabelState,
    args: argparse.Namespace,
    dirs: dict[str, Path],
) -> dict:
    predictor.set_image(image)
    target_mask = None
    target_score = None
    if state.target_point is not None:
        target_mask, target_score = predict_from_point(predictor, state.target_point)

    place_mask = None
    place_score = None
    place_prompt_type = None
    box_prompt = normalize_box(state.place_box)
    if state.place_point is not None:
        place_mask, place_score = predict_from_point(predictor, state.place_point)
        place_prompt_type = "point"
    elif box_prompt is not None:
        masks, scores, _ = predictor.predict(box=np.array(box_prompt, dtype=np.float32), multimask_output=True)
        best = int(np.argmax(scores))
        place_mask = masks[best].astype(bool)
        place_score = float(scores[best])
        place_prompt_type = "box"

    target_box = mask_bbox(target_mask)
    generated_place_box = mask_bbox(place_mask)

    overlay = image.copy()
    if place_mask is not None:
        overlay = tint_mask(overlay, place_mask, (0, 128, 255), 0.28)
    if target_mask is not None:
        overlay = tint_mask(overlay, target_mask, (255, 128, 0), 0.35)
    if generated_place_box is not None:
        x1, y1, x2, y2 = generated_place_box
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)
    elif box_prompt is not None:
        cv2.rectangle(overlay, box_prompt[:2], box_prompt[2:], (255, 0, 0), 2)
    if state.target_point is not None:
        draw_crosshair(overlay, state.target_point)
    if state.place_point is not None:
        draw_point_marker(overlay, state.place_point, color=(0, 128, 255))

    stem = image_path.stem
    overlay_path = dirs["overlays"] / f"{stem}_sam_vp_overlay.png"
    meta_path = dirs["meta"] / f"{stem}_sam_vp_meta.json"
    target_mask_path = dirs["masks"] / f"{stem}_target_mask.png"
    place_mask_path = dirs["masks"] / f"{stem}_place_mask.png"

    Image.fromarray(overlay).save(overlay_path)
    if target_mask is not None:
        Image.fromarray((target_mask.astype(np.uint8) * 255)).save(target_mask_path)
    if place_mask is not None:
        Image.fromarray((place_mask.astype(np.uint8) * 255)).save(place_mask_path)

    meta = {
        "image": str(image_path),
        "target_object": args.target_object,
        "target_prompt_type": "point" if state.target_point is not None else None,
        "target_point": None if state.target_point is None else list(state.target_point),
        "target_score": target_score,
        "target_mask_bbox": target_box,
        "target_mask_centroid": mask_centroid(target_mask),
        "place": args.place,
        "place_prompt_type": place_prompt_type,
        "place_point": None if state.place_point is None else list(state.place_point),
        "place_box_prompt": None if box_prompt is None else list(box_prompt),
        "place_score": place_score,
        "place_mask_bbox": generated_place_box,
        "place_mask_centroid": mask_centroid(place_mask),
        "generated_place_box": generated_place_box,
        "overlay_path": str(overlay_path),
        "target_mask_path": None if target_mask is None else str(target_mask_path),
        "place_mask_path": None if place_mask is None else str(place_mask_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None)
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--pattern", default="*.png")
    parser.add_argument("--out_dir", default="outputs/vp_sam_labels")
    parser.add_argument("--checkpoint", default="models/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", default="vit_b")
    parser.add_argument("--target_object", default="orange cube")
    parser.add_argument("--place", default="white basket")
    parser.add_argument("--max_display_size", type=int, default=900)
    parser.add_argument("--no_sam", action="store_true", help="Only save click labels; do not load/run SAM.")
    args = parser.parse_args()

    image_paths = collect_images(args.image, args.image_dir, args.pattern)
    out_dir = Path(args.out_dir).expanduser()
    dirs = make_dirs(out_dir)

    predictor = None
    if not args.no_sam:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        print(f"[VP-SAM] loaded {args.model_type} on {device}")

    window = "vp_sam_click_labeler"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    index = 0
    state = LabelState()

    def image_to_display_scale(image: np.ndarray) -> float:
        h, w = image.shape[:2]
        return min(1.0, float(args.max_display_size) / max(h, w))

    current_image = np.zeros((1, 1, 3), dtype=np.uint8)
    display_scale = 1.0

    def to_image_xy(x: int, y: int) -> tuple[int, int]:
        h, w = current_image.shape[:2]
        return clamp_xy(int(round(x / display_scale)), int(round(y / display_scale)), w, h)

    def on_mouse(event, x, y, flags, param):
        nonlocal state
        ix, iy = to_image_xy(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            if state.mode == "target":
                state.target_point = (ix, iy)
                print(f"[VP-SAM] target_point={state.target_point}")
            elif state.mode == "place":
                state.place_point = (ix, iy)
                print(f"[VP-SAM] place_point={state.place_point}")
            else:
                state.drag_start = (ix, iy)
                state.drag_current = (ix, iy)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start is not None:
            state.drag_current = (ix, iy)
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start is not None:
            state.drag_current = (ix, iy)
            state.place_box = normalize_box((*state.drag_start, *state.drag_current))
            state.drag_start = None
            state.drag_current = None
            print(f"[VP-SAM] optional place_box_prompt={state.place_box}")
        elif event == cv2.EVENT_RBUTTONDOWN:
            state.drag_start = (ix, iy)
            state.drag_current = (ix, iy)
        elif event == cv2.EVENT_RBUTTONUP and state.drag_start is not None:
            state.drag_current = (ix, iy)
            state.place_box = normalize_box((*state.drag_start, *state.drag_current))
            state.drag_start = None
            state.drag_current = None
            print(f"[VP-SAM] optional place_box_prompt={state.place_box}")

    cv2.setMouseCallback(window, on_mouse)

    while 0 <= index < len(image_paths):
        image_path = image_paths[index]
        current_image = np.asarray(Image.open(image_path).convert("RGB"))
        display_scale = image_to_display_scale(current_image)
        state = LabelState()
        label_path = dirs["labels"] / f"{image_path.stem}.json"
        load_existing_label(label_path, state)

        print(f"\n[VP-SAM] {index + 1}/{len(image_paths)} {image_path}")
        print("[VP-SAM] t=target point, l=place point, b=optional box, left click/drag, enter=SAM+save, s=save labels, r=reset, n/p=next/prev, q=quit")

        while True:
            display = render_display(current_image, state, display_scale, image_path.name)
            cv2.imshow(window, cv2.cvtColor(display, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                return
            if key == ord("t"):
                state.mode = "target"
                print("[VP-SAM] mode=target point")
            elif key == ord("l"):
                state.mode = "place"
                print("[VP-SAM] mode=place point")
            elif key == ord("b"):
                state.mode = "box"
                print("[VP-SAM] mode=optional place box")
            elif key == ord("r"):
                state = LabelState(mode=state.mode)
                print("[VP-SAM] reset labels")
            elif key == ord("s"):
                label = save_label(label_path, image_path, state, args)
                print(f"[VP-SAM] saved label {label_path}: {label}")
            elif key in (13, 10):
                label = save_label(label_path, image_path, state, args)
                print(f"[VP-SAM] saved label {label_path}: {label}")
                if predictor is not None:
                    meta = run_sam(predictor, current_image, image_path, state, args, dirs)
                    print(f"[VP-SAM] saved overlay {meta['overlay_path']}")
                    print(f"[VP-SAM] generated target_bbox={meta['target_mask_bbox']} place_bbox={meta['generated_place_box']}")
            elif key == ord("n"):
                index += 1
                break
            elif key == ord("p"):
                index = max(index - 1, 0)
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
