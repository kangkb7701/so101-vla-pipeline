from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None


@dataclass
class VPEventDecision:
    switch_to_place: bool
    confidence: float
    reason: str
    latency_ms: float
    raw_response: str


class QwenVPEventTrigger:
    """Qwen-VL event detector for VP-VLA-style phase changes.

    This is intentionally not a planner and not an instruction generator. Octo keeps
    receiving the full pick-and-place task; Qwen only decides whether the visual
    prompt should switch from target-object crosshair to placement-location box.
    """

    def __init__(
        self,
        task_description: str,
        target_object: str,
        target_location: str,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 80,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.task_description = task_description
        self.target_object = target_object
        self.target_location = target_location
        self.max_new_tokens = max_new_tokens
        self.device = device

        print(f"📦 [QwenVPEventTrigger] Loading {model_id} on {device}...")
        t0 = time.monotonic()
        model_cls = Qwen2VLForConditionalGeneration
        if "Qwen2.5-VL" in model_id:
            if Qwen2_5_VLForConditionalGeneration is None:
                raise ImportError("Qwen2.5-VL requires a transformers build with Qwen2_5_VLForConditionalGeneration")
            model_cls = Qwen2_5_VLForConditionalGeneration
        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_id)
        load_time = time.monotonic() - t0
        vram_gb = torch.cuda.memory_allocated(device) / 1e9
        print(f"✅ [QwenVPEventTrigger] Loaded in {load_time:.1f}s, VRAM={vram_gb:.2f}GB")

        self._system_prompt = (
            "You are an event detector for a VP-VLA robot policy.\n"
            "Your only job is deciding whether the visual prompt should switch from "
            "PICK phase to PLACE phase.\n\n"
            f"Task: {task_description}\n"
            f"Target object: {target_object}\n"
            f"Placement location: {target_location}\n\n"
            "Images:\n"
            "- Image 1: top view of the full tabletop scene.\n"
            "- Image 2: wrist-mounted close-up near the gripper.\n\n"
            "Return switch_to_place=true ONLY when the target object is already physically "
            "secured by the gripper or clearly lifted/carried by the robot. Do NOT switch "
            "just because the gripper is near the object, aligned with it, or ready to close. "
            "If the object is still resting ungrasped on the table, return false. "
            "If the wrist view is ambiguous, return false.\n\n"
            "Use confidence >= 0.85 only for an obvious completed pick. "
            "Respond with ONLY JSON like this:\n"
            '{"switch_to_place": false, "confidence": 0.0, "reason": "one short sentence"}'
        )

    @staticmethod
    def _to_pil(img) -> Image.Image:
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

    def _build_messages(self, top_frame_rgb, wrist_frame_rgb, elapsed_s: float) -> list:
        top_pil = self._to_pil(top_frame_rgb)
        wrist_pil = self._to_pil(wrist_frame_rgb)
        user_context = (
            f"Elapsed control time: {elapsed_s:.1f}s\n"
            "Decide whether the robot has completed the pick and should now attend to "
            "the placement location. Be conservative; false is safer when uncertain.\n"
            "Respond with ONLY the JSON object."
        )
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": self._system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Image 1 (top view):"},
                    {"type": "image", "image": top_pil},
                    {"type": "text", "text": "Image 2 (wrist view):"},
                    {"type": "image", "image": wrist_pil},
                    {"type": "text", "text": user_context},
                ],
            },
        ]

    @torch.no_grad()
    def decide(self, top_frame_rgb, wrist_frame_rgb, elapsed_s: float) -> VPEventDecision:
        t0 = time.monotonic()
        try:
            messages = self._build_messages(top_frame_rgb, wrist_frame_rgb, elapsed_s)
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            latency_ms = (time.monotonic() - t0) * 1000.0

            data = self._parse_json(output_text)
            if data is None:
                return self._fail_safe(latency_ms, "JSON parse failed", output_text)

            switch_to_place = bool(data.get("switch_to_place", False))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(data.get("reason", ""))[:180]
            return VPEventDecision(
                switch_to_place=switch_to_place,
                confidence=confidence,
                reason=reason,
                latency_ms=latency_ms,
                raw_response=output_text,
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._fail_safe(latency_ms, f"{type(exc).__name__}: {exc}", "")

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fence_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        return None

    @staticmethod
    def _fail_safe(latency_ms: float, msg: str, raw: str) -> VPEventDecision:
        print(f"⚠️ [QwenVPEventTrigger] fail-safe: {msg}")
        return VPEventDecision(
            switch_to_place=False,
            confidence=0.0,
            reason=f"fail-safe: {msg}",
            latency_ms=latency_ms,
            raw_response=raw,
        )
