#!/usr/bin/env python3
"""Create a quick VP-VLA visual prompt preview image.

This is only a visualization helper. It does not touch a LeRobot dataset.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.scripts.lerobot_record_vp import _draw_visual_prompt, _parse_float_list


def make_synthetic_scene(width: int = 640, height: int = 480) -> np.ndarray:
    image = Image.new("RGB", (width, height), (194, 187, 172))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, int(height * 0.18)], fill=(154, 161, 157))
    draw.rectangle([0, int(height * 0.18), width, height], fill=(211, 202, 184))
    draw.rectangle([75, 255, 185, 365], fill=(185, 52, 47), outline=(105, 32, 30), width=3)
    draw.rectangle([103, 232, 157, 255], fill=(210, 75, 70), outline=(105, 32, 30), width=2)
    draw.ellipse([430, 230, 570, 370], fill=(45, 45, 48), outline=(20, 20, 22), width=4)
    draw.ellipse([455, 255, 545, 345], fill=(68, 68, 72))
    draw.rectangle([255, 285, 330, 355], fill=(55, 95, 178), outline=(34, 62, 116), width=3)
    draw.line([0, 380, width, 405], fill=(170, 161, 145), width=3)
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="", help="Optional input RGB image path.")
    parser.add_argument("--out_dir", type=str, default="/home/aivlab/kkb_capstone/outputs/vp_visual_prompt_preview")
    parser.add_argument("--coord_format", type=str, default="pixel", choices=["pixel", "normalized", "qwen1000"])
    parser.add_argument("--target_object_point", type=str, default="130,300")
    parser.add_argument("--target_location_box", type=str, default="420,220,580,380")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        clean = np.asarray(Image.open(args.image).convert("RGB"))
    else:
        clean = make_synthetic_scene()

    point = _parse_float_list(args.target_object_point, 2)
    box = _parse_float_list(args.target_location_box, 4)
    overlay = _draw_visual_prompt(
        clean,
        point=point,
        box=box,
        coord_format=args.coord_format,
        line_length=22,
        gap=9,
        thickness=3,
    )

    clean_path = out_dir / "clean_example.png"
    overlay_path = out_dir / "vp_overlay_example.png"
    Image.fromarray(clean).save(clean_path)
    Image.fromarray(overlay).save(overlay_path)
    print(clean_path)
    print(overlay_path)


if __name__ == "__main__":
    main()
