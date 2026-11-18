"""
QwenSupervisor — Local Qwen2-VL-2B-Instruct supervisor for the VLA pickup pipeline.

Same `monitor()` interface as GeminiSupervisor — main_real2 can swap between the
two via config without touching integration code. Shared dataclasses/helpers are
imported from supervisor_agent.py (single source of truth).

Install:
    pip install --upgrade transformers qwen-vl-utils
    (model auto-downloads from HuggingFace Hub on first __init__ call, ~4.4GB)

Memory footprint (fp16): ~4.5GB VRAM. Runs comfortably on RTX 3090 24GB
alongside an octo_server process. Enable MPS for best concurrent throughput.

Design notes:
- Model + processor loaded ONCE at __init__ (slow: ~10s warm cache, ~50s first download).
- monitor() is the only hot path. Build messages → tokenize → generate → parse → return.
- Qwen2-VL does NOT have schema-enforced structured output (unlike Gemini). We rely on
  prompt-level JSON request + robust regex parsing + fail-safe fallback.
- max_new_tokens=150 caps response length (our JSON is ~80-120 tokens).
"""

from __future__ import annotations

import json
import re
import time

import numpy as np
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
from qwen_vl_utils import process_vision_info

from agents.supervisor_types import (
    STATUS_VALUES,
    StateSnapshot, ObjectiveSignals, InstructionUseRecord, SupervisorDecision,
)


# ============================================================
# QwenSupervisor
# ============================================================

class QwenSupervisor:
    """Local Qwen2-VL-2B-Instruct supervisor with the same monitor() contract
    as GeminiSupervisor. monitor() never raises — failures return a fail-safe
    decision with status='in_progress' and current_instruction unchanged.
    """

    def __init__(
        self,
        task_description: str,
        candidate_instructions: list,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        instruction_mode: str = "strict",  # 'strict' | 'warn' | 'flexible'
        motion_stall_deg_per_s: float = 1.0,
        ee_stall_cm_over_5s: float = 2.0,
        max_new_tokens: int = 150,
        device: str = "cuda",
        torch_dtype=torch.float16,
    ):
        if not candidate_instructions:
            raise ValueError("candidate_instructions must be non-empty")
        if instruction_mode not in ('strict', 'warn', 'flexible'):
            raise ValueError(f"invalid instruction_mode: {instruction_mode!r}")

        self.candidate_instructions = list(candidate_instructions)
        self.candidate_set = set(self.candidate_instructions)
        self.instruction_mode = instruction_mode
        self.max_new_tokens = max_new_tokens
        self.device = device

        print(f"📦 [QwenSupervisor] Loading {model_id} on {device}...")
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
        print(f"✅ [QwenSupervisor] Loaded in {load_time:.1f}s, VRAM={vram_gb:.2f}GB")

        self._system_prompt = self._build_system_prompt(
            task_description,
            candidate_instructions,
            instruction_mode,
            motion_stall_deg_per_s,
            ee_stall_cm_over_5s,
        )

    # ----------- prompt builders -----------
    @staticmethod
    def _build_system_prompt(task_description: str, candidate_instructions: list,
                             instruction_mode: str,
                             motion_stall_deg_per_s: float,
                             ee_stall_cm_over_5s: float) -> str:
        """Compressed system prompt — Qwen2-VL-2B can't reliably follow Gemini's
        800-token policy doc. Keep it short, explicit, with a concrete JSON example.
        """
        cands_numbered = "\n".join(
            f"{i+1}. {c!r}" for i, c in enumerate(candidate_instructions)
        )
        example_instruction = candidate_instructions[0]
        if instruction_mode == 'strict':
            instruction_policy = (
                "Pick ONE instruction (the EXACT string from this fixed list):\n"
                f"{cands_numbered}\n\n"
            )
        else:
            instruction_policy = (
                "You may issue any short natural-language instruction for the VLA.\n"
                "Use simple robot-action language that a tabletop pickup VLA is likely "
                "to understand: move/approach/align/open/close/grasp/lift/place. "
                "Name the visible target object using plain attributes such as color, "
                "shape, and object type. Avoid long plans, coordinates, JSON inside "
                "the instruction, or multiple commands joined together.\n\n"
                "Good instruction style examples (not a closed set):\n"
                f"{cands_numbered}\n\n"
            )
        return (
            "You are a supervisor monitoring a robot arm executing this task:\n"
            f"  {task_description}\n\n"
            "You see two images:\n"
            "  - Image 1: Top-down view (overall scene)\n"
            "  - Image 2: Wrist-mounted view (close-up of gripper)\n\n"
            f"{instruction_policy}"
            "Pick ONE phase for the current/next pickup phase:\n"
            "  - approach: gripper is far from the target object; move toward it\n"
            "  - align: target is near/visible; align gripper with the target\n"
            "  - pre_grasp: gripper is aligned and open; lower or position around target\n"
            "  - grasp: target is between/under fingers; grasp or close on target\n"
            "  - lift: target appears secured; lift the target object\n"
            "  - complete: target is grasped and lifted; final goal achieved\n"
            "  - recover: correction/recovery is needed before continuing\n\n"
            "Pick ONE status:\n"
            '  - "in_progress": current sub-goal is improving goal progress; keep current instruction\n'
            '  - "subgoal_complete": current sub-goal clearly done; move to next phase\n'
            '  - "corrective_action": robot is moving, but alignment/distance is getting worse\n'
            '  - "stuck": no progress, need different approach\n'
            '  - "failed": unrecoverable error\n'
            '  - "task_complete": entire pickup task is done\n\n'
            "RULES (read carefully):\n"
            "  1. DEFAULT to status='in_progress' with the CURRENT instruction. Do NOT\n"
            "     switch unless you have STRONG evidence.\n"
            "  2. Instruction changes are applied only when status is 'subgoal_complete',\n"
            "     'corrective_action', or 'stuck'. For 'task_complete', 'failed', and\n"
            "     'in_progress', keep the current instruction.\n"
            f"  3. 'stuck' requires ALL: motion_magnitude < {motion_stall_deg_per_s} deg/s,\n"
            f"     EE position change < {ee_stall_cm_over_5s} cm over 5s, and no visual\n"
            "     change. Slow but purposeful motion is NOT stuck.\n"
            "  4. Use 'corrective_action' when the robot is still moving but the current\n"
            "     command is no longer helping: the gripper moves farther from the target,\n"
            "     overshoots the object, rotates away, drifts sideways, or hovers near the\n"
            "     object without improving alignment. Issue one short corrective command.\n"
            "  5. Do not keep a broad approach command for a long time. If the same\n"
            "     instruction has been active for more than 8 seconds and the gripper is\n"
            "     not clearly better aligned with the target, use 'corrective_action' or\n"
            "     'subgoal_complete' instead of 'in_progress'.\n"
            "  6. When uncertain (occluded, short hold <5s, ambiguous), return\n"
            "     status='in_progress' with current instruction.\n"
            "  7. Use the objective signals (telemetry) as ground truth; vision is\n"
            "     supplementary.\n"
            "  8. Phase transitions for any target object, not just cubes: approach when\n"
            "     far, align when near, pre_grasp when aligned and open, grasp when the\n"
            "     object is between/under the fingers, lift when grasped, complete only\n"
            "     after the target is lifted.\n"
            "  9. Do NOT treat top-view 2D proximity as grasp readiness. The top view\n"
            "     cannot prove height or that the object is between the fingers. If the\n"
            "     wrist view does not clearly show the object centered between/under the\n"
            "     fingers at low grasp height, keep pre_grasp/lower/align instead of grasp.\n"
            "  10. When the gripper is already in true grasp range, do not keep issuing\n"
            "      broad approach commands. Switch to phase='grasp' with an instruction\n"
            "      like 'grasp the target object' or 'grasp the orange cube'.\n"
            "  11. Return status='task_complete' only when the final goal is visibly and\n"
            "      physically achieved, such as the target object being grasped and lifted.\n\n"
            "Respond with ONLY a JSON object (no markdown, no commentary). Do not copy this example:\n"
            '{"status": "in_progress", "phase": "approach", "instruction": '
            f'"{example_instruction}", "reason": "the gripper is still moving toward the target"}}'
        )

    @staticmethod
    def _format_signals(signals: ObjectiveSignals, history: list) -> str:
        last3 = history[-3:] if len(history) >= 3 else history
        ee_pose_last_3 = [tuple(round(v, 4) for v in s.ee_pos) for s in last3]
        return (
            "Objective signals (sensor telemetry):\n"
            f"  gripper_closed_recent: {signals.gripper_closed_recent}\n"
            f"  ee_z_delta_recent:     {signals.ee_z_delta_recent:+.4f} m  "
            f"(positive = arm lifting)\n"
            f"  motion_magnitude:      {signals.motion_magnitude:.2f} deg/s\n"
            f"  ee_pose_last_3:        {ee_pose_last_3}"
        )

    @staticmethod
    def _format_episode_history(history: list,
                                current_instruction: str,
                                held_for_s: float) -> str:
        lines = ["Instructions used so far:"]
        for i, rec in enumerate(history, start=1):
            lines.append(f'  {i}. "{rec.instruction}"  ({rec.duration_s:.1f}s)')
        cur_idx = len(history) + 1
        lines.append(
            f'  {cur_idx}. "{current_instruction}"  '
            f'(currently active, {held_for_s:.1f}s)'
        )
        return "\n".join(lines)

    @staticmethod
    def _format_recent_decisions(decisions: list, poll_s: float = 1.5) -> str:
        if not decisions:
            return "Your recent decisions: (none yet — this is the first call)"
        lines = ["Your recent decisions:"]
        n = len(decisions)
        for i, d in enumerate(decisions):
            t_offset = -poll_s * (n - i)
            phase = f", phase={d.phase}" if getattr(d, 'phase', '') else ""
            lines.append(
                f'  {t_offset:+.1f}s: status={d.status}{phase}, reason="{d.reason}"'
            )
        lines.append("  now:    (this call)")
        return "\n".join(lines)

    @staticmethod
    def _to_pil(img) -> Image.Image:
        if isinstance(img, np.ndarray):
            return Image.fromarray(img)
        return img

    def _build_messages(
        self,
        top_frame_rgb, wrist_frame_rgb,
        current_instruction: str, held_for_s: float,
        state_history: list,
        signals: ObjectiveSignals,
        episode_history: list,
        recent_decisions: list,
    ) -> list:
        top_pil   = self._to_pil(top_frame_rgb)
        wrist_pil = self._to_pil(wrist_frame_rgb)
        user_context = (
            f'Current instruction (active for {held_for_s:.1f}s): "{current_instruction}"\n\n'
            f"{self._format_signals(signals, state_history)}\n\n"
            f"{self._format_episode_history(episode_history, current_instruction, held_for_s)}\n\n"
            f"{self._format_recent_decisions(recent_decisions)}\n\n"
            "Respond with ONLY the JSON object — no markdown fences, no extra text."
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

    # ----------- main entry point -----------
    @torch.no_grad()
    def monitor(
        self,
        top_frame_rgb,
        wrist_frame_rgb,
        current_instruction: str,
        held_for_s: float,
        state_history: list,
        signals: ObjectiveSignals,
        episode_history: list,
        recent_decisions: list,
    ) -> SupervisorDecision:
        """Single Qwen forward pass. Never raises."""
        t0 = time.monotonic()
        try:
            messages = self._build_messages(
                top_frame_rgb, wrist_frame_rgb,
                current_instruction, held_for_s,
                state_history, signals,
                episode_history, recent_decisions,
            )
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
                return self._fail_safe(
                    current_instruction, latency_ms,
                    f"JSON parse failed", output_text,
                )

            status = data.get('status')
            if status not in STATUS_VALUES:
                return self._fail_safe(
                    current_instruction, latency_ms,
                    f"invalid status {status!r}", output_text,
                )

            instruction = data.get('instruction', current_instruction)
            if status in ('in_progress', 'failed', 'task_complete'):
                instruction = current_instruction

            if self.instruction_mode == 'strict':
                if instruction not in self.candidate_set:
                    return self._fail_safe(
                        current_instruction, latency_ms,
                        f"out-of-set instruction {instruction!r} in strict mode",
                        output_text,
                    )
            elif self.instruction_mode == 'warn':
                if instruction not in self.candidate_set:
                    print(
                        f"⚠️ [QwenSupervisor] out-of-set instruction accepted "
                        f"(warn mode): {instruction!r}"
                    )
            # 'flexible' silently accepts anything.

            phase = data.get('phase', '')
            if status == 'task_complete':
                phase = phase or 'complete'
            elif status == 'corrective_action':
                phase = phase or 'recover'

            reason = data.get('reason', '')
            return SupervisorDecision(
                status=status,
                instruction=instruction,
                reason=reason,
                latency_ms=latency_ms,
                raw_response=output_text,
                phase=phase,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._fail_safe(
                current_instruction, latency_ms,
                f"{type(e).__name__}: {e}", "",
            )

    @staticmethod
    def _parse_json(text: str) -> dict | None:
        """Robust JSON extraction. Tries direct parse, then strips markdown fences,
        then regex-extracts the first {...} block. Returns None if all fail.
        """
        text = text.strip()
        # 1. direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        # 2. strip ```json ... ``` markdown fences (Qwen sometimes adds them)
        fence_match = re.search(
            r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL,
        )
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass
        # 3. find first balanced {...} block
        brace_match = re.search(r'\{.*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass
        return None

    def _fail_safe(self, current_instruction: str, latency_ms: float,
                   msg: str, raw: str) -> SupervisorDecision:
        print(f"⚠️ [QwenSupervisor] fail-safe: {msg}")
        return SupervisorDecision(
            status='in_progress',
            instruction=current_instruction,
            reason=f'fail-safe: {msg}',
            latency_ms=latency_ms,
            raw_response=raw,
        )
