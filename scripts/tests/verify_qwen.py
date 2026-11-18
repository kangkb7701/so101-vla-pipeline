"""
Quick verification: load Qwen2-VL-2B-Instruct, run multi-image generate, measure latency.

First run downloads the model (~4.4GB) to HF cache.
"""
import time
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEBUG_FRAMES = REPO_ROOT / "debug_frames"

print(f"📦 Loading model: {MODEL_ID}")
print(f"   (first run will download ~4.4GB to HF cache)")
t0 = time.monotonic()
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="cuda",
)
processor = AutoProcessor.from_pretrained(MODEL_ID)
load_time = time.monotonic() - t0
print(f"✅ Loaded in {load_time:.1f}s")
print(f"   model dtype: {next(model.parameters()).dtype}")
print(f"   GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")

top_path  = DEBUG_FRAMES / "20260515_170815_step01_top_raw.png"
wrist_path = DEBUG_FRAMES / "20260515_170815_step01_front_raw.png"
top_img   = Image.open(top_path).convert("RGB")
wrist_img = Image.open(wrist_path).convert("RGB")
print(f"\n📷 Loaded images: top {top_img.size}, wrist {wrist_img.size}")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": top_img},
            {"type": "image", "image": wrist_img},
            {"type": "text",
             "text": "Image 1 is a top-down view, image 2 is a wrist-mounted view "
                     "of a robot arm working on a tabletop pickup task. "
                     "Briefly describe what you see in each image and what the robot "
                     "appears to be doing. Keep your answer under 4 sentences."},
        ],
    }
]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")
print(f"\n🛰️  Running generate (3 trials: cold + warm + warm)...")
for trial in range(3):
    t1 = time.monotonic()
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
        )
    elapsed_ms = (time.monotonic() - t1) * 1000.0
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    tag = "[COLD]" if trial == 0 else "[WARM]"
    print(f"\n--- Trial {trial+1} {tag}: {elapsed_ms:.1f}ms ---")
    print(output_text)

print(f"\n📊 Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved")
print("\n✅ Qwen2-VL verification passed.")
