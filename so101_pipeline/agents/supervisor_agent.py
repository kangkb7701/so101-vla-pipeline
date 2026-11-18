"""
GeminiSupervisor — High-level LLM supervisor for the VLA pickup pipeline.

Watches top + wrist camera frames and decides whether to swap the natural-language
instruction passed to the low-level VLA (Octo). Stateless API: every call rebuilds
the full operational context from telemetry + episode history + recent decisions.

Install:
    pip install google-genai

Environment:
    GEMINI_API_KEY must be set before constructing GeminiSupervisor.

Design notes:
- This module imports `google-genai`. main_real2.py uses **lazy import** gated on
  USE_SUPERVISOR=True so the baseline run does NOT require this dependency.
- All durations use time.monotonic() (NTP-immune).
- monitor() never raises — any failure (network, parse, schema) → fail-safe
  decision with status='in_progress' and current_instruction unchanged.
"""

from __future__ import annotations

import io
import json
import time

import numpy as np
from PIL import Image

from google import genai
from google.genai import types

# Shared types + helpers live in supervisor_types so QwenSupervisor and main_real2
# can use them without pulling in google-genai.
from agents.supervisor_types import (  # noqa: F401  (re-export for back-compat)
    STATUS_VALUES,
    ALLOW_SWITCH_STATUSES,
    StateSnapshot,
    ObjectiveSignals,
    InstructionUseRecord,
    SupervisorDecision,
    rot_mat_to_rpy_deg,
    compute_signals,
)


def _img_to_jpeg_part(img, quality: int = 70) -> types.Part:
    """Convert RGB ndarray or PIL.Image → Gemini inline JPEG Part."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg')


# ============================================================
# GeminiSupervisor
# ============================================================

class GeminiSupervisor:
    """Stateless Gemini-Flash supervisor that proposes instruction swaps.

    Construction is cheap; the system prompt and response schema are built once
    and reused across calls. The expensive work is monitor() which makes a single
    Gemini API call per invocation.
    """

    def __init__(
        self,
        api_key: str,
        task_description: str,
        candidate_instructions: list,
        model_name: str = 'gemini-2.5-flash',
        request_timeout_s: float = 10.0,   # Gemini API minimum allowed deadline is 10s
        instruction_mode: str = 'strict',  # 'strict' | 'warn' | 'flexible'
        motion_stall_deg_per_s: float = 1.0,
        ee_stall_cm_over_5s: float = 2.0,
    ):
        if not candidate_instructions:
            raise ValueError("candidate_instructions must be non-empty")
        if instruction_mode not in ('strict', 'warn', 'flexible'):
            raise ValueError(f"invalid instruction_mode: {instruction_mode!r}")

        self.model_name = model_name
        self.candidate_instructions = list(candidate_instructions)
        self.candidate_set = set(self.candidate_instructions)
        self.instruction_mode = instruction_mode

        # Per-call timeout enforced at the HTTP layer (ms).
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=int(request_timeout_s * 1000)),
        )

        # Built once, reused per call.
        self._system_prompt = self._build_system_prompt(
            task_description,
            candidate_instructions,
            motion_stall_deg_per_s,
            ee_stall_cm_over_5s,
        )
        self._response_schema = self._build_response_schema()

    # ----------- prompt builders -----------
    @staticmethod
    def _build_system_prompt(task_description: str, candidate_instructions: list,
                             motion_stall_deg_per_s: float,
                             ee_stall_cm_over_5s: float) -> str:
        cands = "\n".join(f"  - {c!r}" for c in candidate_instructions)
        return (
            "You are a high-level supervisor for an SO-ARM100 6-DOF robot arm "
            "executing a pickup task.\n\n"
            "Robot capability: 6 joints + 1-DOF gripper, controlled by a low-level "
            "VLA (Octo) that takes a single natural-language instruction and "
            "outputs joint deltas at 10Hz.\n\n"
            "Cameras: two views.\n"
            "  - Top view   — overall scene layout, target position on table, gross\n"
            "                 arm pose. May be occluded by arm/gripper.\n"
            "  - Wrist view — close-up of gripper and what is directly in front of\n"
            "                 it. Best for verifying alignment, grasp completion,\n"
            "                 and fine approach.\n"
            "Use both. Top alone is insufficient once the arm is near the object.\n\n"
            f"Task: {task_description}\n\n"
            "You may issue ONLY instructions from this fixed set "
            "(any other text will be rejected):\n"
            f"{cands}\n\n"
            "═══════════════════════════════════════════════════════════════════════\n"
            "STATUS TAXONOMY (pick exactly one per call):\n"
            "═══════════════════════════════════════════════════════════════════════\n"
            "- in_progress      : current sub-goal is going well; you are confident\n"
            "                     it is still the right instruction.\n"
            "- subgoal_complete : current sub-goal is CLEARLY done; the next\n"
            "                     instruction in the natural progression should run.\n"
            "- stuck            : no observable progress AND multi-signal evidence\n"
            "                     (see below). Re-plan to a different instruction.\n"
            "- failed           : a fault state has occurred that no instruction can\n"
            "                     recover (e.g., the pen has been knocked away).\n"
            "- task_complete    : the WHOLE pickup task is done (pen lifted and held).\n\n"
            "The system will ONLY act on your proposed `instruction` when status is\n"
            "`subgoal_complete` or `stuck`. For all other statuses, set `instruction`\n"
            "to the current instruction (it will be ignored anyway).\n\n"
            "═══════════════════════════════════════════════════════════════════════\n"
            "CRITICAL SWITCHING POLICY — read carefully, this is non-negotiable:\n"
            "═══════════════════════════════════════════════════════════════════════\n"
            "1. The current instruction is the DEFAULT. Do not switch unless you have\n"
            "   strong, concrete evidence.\n\n"
            "2. `subgoal_complete` requires: a clearly visible state change matching\n"
            "   the intent of the current instruction (e.g., gripper now ABOVE the\n"
            "   pen for 'approach pen from above'; gripper now CLOSED around pen for\n"
            "   'grasp pen').\n\n"
            "3. `stuck` requires ALL THREE simultaneously, not just one:\n"
            f"   (a) motion_magnitude near zero (< {motion_stall_deg_per_s} deg/s),\n"
            f"   (b) EE pose nearly unchanged (||Δ EE_pos|| < {ee_stall_cm_over_5s} cm over last 5 s),\n"
            "   (c) no meaningful visual scene change you can detect.\n"
            "   Slow but PURPOSEFUL motion (e.g., fine alignment, careful descent) is\n"
            "   NOT stuck. Low motion alone is not stuck.\n\n"
            "4. Long hold duration alone does NOT imply stagnation or failure. A\n"
            "   robot may legitimately spend 10+ seconds on a delicate sub-goal.\n"
            "   Look at the signals and image, not the clock.\n\n"
            "5. When uncertain: return status='in_progress' with the CURRENT\n"
            "   instruction unchanged. 'Uncertain' includes: occluded views,\n"
            "   ambiguous gripper state, short hold times (<5 s), conflicting signals.\n\n"
            "6. The cost of a WRONG SWITCH is much higher than a LATE SWITCH. A\n"
            "   wrong switch can derail the entire task; a late switch just costs\n"
            "   seconds.\n\n"
            "Use the objective signals (computed from joint/EE telemetry) as ground\n"
            "truth. Vision is supplementary, not primary, for state estimation.\n"
            "═══════════════════════════════════════════════════════════════════════\n\n"
            "Output JSON matching the response schema. Be concise in 'reason'\n"
            "(≤1 sentence)."
        )

    def _build_response_schema(self) -> dict:
        instruction_prop = {"type": "string"}
        if self.instruction_mode == 'strict':
            instruction_prop["enum"] = self.candidate_instructions
        return {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": list(STATUS_VALUES)},
                "phase": {
                    "type": "string",
                    "enum": [
                        "approach", "align", "pre_grasp", "grasp",
                        "lift", "complete", "recover",
                    ],
                },
                "instruction": instruction_prop,
                "reason": {"type": "string"},
            },
            "required": ["status", "phase", "instruction", "reason"],
        }

    # ----------- user-turn builders -----------
    @staticmethod
    def _format_signals(signals: ObjectiveSignals, history: list) -> str:
        last3 = history[-3:] if len(history) >= 3 else history
        ee_pose_last_3 = [tuple(round(v, 4) for v in s.ee_pos) for s in last3]
        return (
            "Objective signals (machine-truth telemetry):\n"
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
        lines = ["Instructions used so far in this episode:"]
        for i, rec in enumerate(history, start=1):
            lines.append(f'  {i}. "{rec.instruction}"  ({rec.duration_s:.1f}s)')
        cur_idx = len(history) + 1
        lines.append(
            f'  {cur_idx}. "{current_instruction}"  '
            f'(currently active, {held_for_s:.1f}s)'
        )
        return "\n".join(lines)

    @staticmethod
    def _format_recent_decisions(decisions: list, poll_s: float = 3.0) -> str:
        if not decisions:
            return "Your recent decisions: (none yet — this is the first call)"
        lines = ["Your recent decisions:"]
        n = len(decisions)
        for i, d in enumerate(decisions):
            t_offset = -poll_s * (n - i)
            lines.append(
                f'  {t_offset:+.1f}s: status={d.status}, reason="{d.reason}"'
            )
        lines.append("  now:    (this call)")
        return "\n".join(lines)

    def _build_user_contents(
        self,
        top_frame_rgb, wrist_frame_rgb,
        current_instruction: str, held_for_s: float,
        state_history: list,
        signals: ObjectiveSignals,
        episode_history: list,
        recent_decisions: list,
    ) -> list:
        """Build the user-turn parts list. Order matters."""
        parts = [
            "Top view:",
            _img_to_jpeg_part(top_frame_rgb),
            "Wrist view:",
            _img_to_jpeg_part(wrist_frame_rgb),
            f'Current instruction (active for {held_for_s:.1f}s): "{current_instruction}"',
            self._format_signals(signals, state_history),
            self._format_episode_history(episode_history, current_instruction, held_for_s),
            self._format_recent_decisions(recent_decisions),
            "Respond with JSON matching the schema. Be concise in 'reason' (≤1 sentence).",
        ]
        return parts

    # ----------- main entry point -----------
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
        """Single Gemini call. Never raises — always returns a SupervisorDecision."""
        t0 = time.monotonic()
        try:
            user_parts = self._build_user_contents(
                top_frame_rgb, wrist_frame_rgb,
                current_instruction, held_for_s,
                state_history, signals,
                episode_history, recent_decisions,
            )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_parts,
                config=types.GenerateContentConfig(
                    system_instruction=self._system_prompt,
                    response_mime_type='application/json',
                    response_schema=self._response_schema,
                    temperature=0.2,
                ),
            )
            latency_ms = (time.monotonic() - t0) * 1000.0

            raw = response.text or ""
            data = json.loads(raw)

            status = data.get('status')
            if status not in STATUS_VALUES:
                return self._fail_safe(
                    current_instruction, latency_ms,
                    f"invalid status {status!r}", raw,
                )

            instruction = data.get('instruction', current_instruction)
            if self.instruction_mode == 'strict':
                if instruction not in self.candidate_set:
                    return self._fail_safe(
                        current_instruction, latency_ms,
                        f"out-of-set instruction {instruction!r} in strict mode", raw,
                    )
            elif self.instruction_mode == 'warn':
                if instruction not in self.candidate_set:
                    print(
                        f"⚠️ [Supervisor] out-of-set instruction accepted (warn mode): "
                        f"{instruction!r}"
                    )
            # 'flexible' silently accepts anything.

            if status in ('in_progress', 'failed', 'task_complete'):
                instruction = current_instruction

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
                raw_response=raw,
                phase=phase,
            )

        except Exception as e:
            latency_ms = (time.monotonic() - t0) * 1000.0
            return self._fail_safe(
                current_instruction, latency_ms,
                f"{type(e).__name__}: {e}", "",
            )

    def _fail_safe(self, current_instruction: str, latency_ms: float,
                   msg: str, raw: str) -> SupervisorDecision:
        print(f"⚠️ [Supervisor] fail-safe: {msg}")
        return SupervisorDecision(
            status='in_progress',
            instruction=current_instruction,
            reason=f'fail-safe: {msg}',
            latency_ms=latency_ms,
            raw_response=raw,
        )
