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
class GraspTriggerDecision:
    close_gripper: bool
    confidence: float
    reason: str
    latency_ms: float
    raw_response: str


class QwenGraspTrigger:
    """Qwen VLM used only as a gripper-close trigger, not as a planner."""

    def __init__(
        self,
        task_description: str,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 80,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        self.task_description = task_description
        self.max_new_tokens = max_new_tokens
        self.device = device

        print(f"📦 [QwenGraspTrigger] Loading {model_id} on {device}...")
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
        print(f"✅ [QwenGraspTrigger] Loaded in {load_time:.1f}s, VRAM={vram_gb:.2f}GB")

        self._system_prompt = (
            "You are only deciding whether a robot gripper should close now.\n"
            f"Task: {task_description}\n\n"
            "You see two images:\n"
            "- Image 1: top view of the scene. Use it only for coarse context.\n"
            "- Image 2: wrist view near the gripper. This is the primary evidence.\n\n"
            "Be conservative. Return close_gripper=true ONLY when the wrist view clearly "
            "shows the orange cube inside the closing region between the two gripper fingers, "
            "and the gripper is already at grasp height. The cube must not merely be near, "
            "in front of, beside, below, or partially outside the fingers.\n"
            "Never return true from top-view proximity alone: the top view cannot prove "
            "height or finger enclosure. Return false if the cube only appears close in "
            "the top view, if the gripper is hovering above it, or if the wrist view does "
            "not show the cube centered in the finger gap.\n"
            "Return false if the wrist view is ambiguous, the cube is not visibly between "
            "the fingers, the gripper is still approaching, the gripper is above the cube, "
            "or closing now might miss the cube. When uncertain, always return false.\n\n"
            "Confidence must be calibrated: use confidence >= 0.85 only for an obvious, "
            "ready-to-close grasp pose. Use <= 0.60 for ambiguous cases.\n\n"
            "Respond with ONLY JSON like this:\n"
            '{"close_gripper": false, "confidence": 0.0, "reason": "one short sentence"}'
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
            "Use the wrist image as the decisive view. Answer true only if the cube is "
            "clearly centered between the gripper fingers and at grasp height right now. "
            "Ignore top-view closeness unless the wrist view also proves finger alignment "
            "and low height. If there is any doubt, answer false.\n"
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
    def decide(self, top_frame_rgb, wrist_frame_rgb, elapsed_s: float) -> GraspTriggerDecision:
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

            close_gripper = bool(data.get("close_gripper", False))
            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))
            reason = str(data.get("reason", ""))[:160]
            return GraspTriggerDecision(
                close_gripper=close_gripper,
                confidence=confidence,
                reason=reason,
                latency_ms=latency_ms,
                raw_response=output_text,
            )
        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._fail_safe(latency_ms, f"{type(e).__name__}: {e}", "")

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
    def _fail_safe(latency_ms: float, msg: str, raw: str) -> GraspTriggerDecision:
        print(f"⚠️ [QwenGraspTrigger] fail-safe: {msg}")
        return GraspTriggerDecision(
            close_gripper=False,
            confidence=0.0,
            reason=f"fail-safe: {msg}",
            latency_ms=latency_ms,
            raw_response=raw,
        )
