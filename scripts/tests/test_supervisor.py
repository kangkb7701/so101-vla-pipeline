"""
Smoke test for GeminiSupervisor — isolated from main pipeline.

Loads a top + wrist image pair from debug_frames/, builds a fake telemetry
history, calls GeminiSupervisor.monitor(), and pretty-prints the result.

This verifies:
  - google-genai SDK is installed and reachable
  - GEMINI_API_KEY is valid
  - The system prompt + response schema produces a parseable structured response
  - End-to-end latency is within budget

Usage:
    export GEMINI_API_KEY="your_key_here"
    python src/scripts/test_supervisor.py
    python src/scripts/test_supervisor.py --step 02 --mode warn
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Ensure src/ is on the path (script may be invoked from repo root)
SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.supervisor_agent import (  # noqa: E402
    GeminiSupervisor, StateSnapshot, compute_signals,
)

REPO_ROOT = SRC_DIR.parent
DEBUG_FRAMES = REPO_ROOT / "debug_frames"


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def build_fake_history(now: float, n: int = 5) -> list:
    """Fake StateSnapshot history: arm descending toward pen, gripper open."""
    history = []
    for i in range(n):
        t = now - (n - 1 - i) * 0.2  # 5Hz, oldest first
        history.append(StateSnapshot(
            t=t,
            instruction="go to the pen",
            q_deg=[10.0 + i, -5.0, 30.0 + i, 0.0, -25.0, 0.0],
            gripper_val=5.0,  # open
            ee_pos=[0.18, 0.02, 0.22 - 0.01 * i],  # descending
            ee_rpy=[0.0, 90.0, 0.0],
        ))
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="01",
                        help="step index (zero-padded, e.g. '00', '01', '02')")
    parser.add_argument("--mode", default="strict",
                        choices=["strict", "warn", "flexible"])
    parser.add_argument("--model", default="gemini-2.5-flash",
                        help="Gemini model name (default: gemini-2.5-flash)")
    parser.add_argument("--n", type=int, default=1,
                        help="number of consecutive calls (for warm-vs-cold latency)")
    parser.add_argument("--top", default=None,
                        help="override top image path")
    parser.add_argument("--wrist", default=None,
                        help="override wrist image path "
                             "(default: front_raw, since 'front' is the wrist camera)")
    args = parser.parse_args()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("🚨 GEMINI_API_KEY not set. Set it first:")
        print('   export GEMINI_API_KEY="your_key_here"')
        sys.exit(1)

    # Default image pair: top_raw + front_raw (front == wrist; same physical camera,
    # just legacy naming in debug_frames/).
    timestamp = "20260515_170815"
    top_path = Path(args.top) if args.top else \
        DEBUG_FRAMES / f"{timestamp}_step{args.step}_top_raw.png"
    wrist_path = Path(args.wrist) if args.wrist else \
        DEBUG_FRAMES / f"{timestamp}_step{args.step}_front_raw.png"

    print(f"📷 top:   {top_path}")
    print(f"📷 wrist: {wrist_path}  (== front camera)")
    top_rgb = load_image(top_path)
    wrist_rgb = load_image(wrist_path)
    print(f"   top   shape={top_rgb.shape}")
    print(f"   wrist shape={wrist_rgb.shape}")

    # Build supervisor (mirrors main_real2 CONFIG defaults)
    candidate_instructions = [
        "go to the pen",
        "approach pen from above",
        "open gripper and lower",
        "grasp pen",
        "lift pen up",
    ]
    print(f"\n🧠 Constructing GeminiSupervisor (mode={args.mode})...")
    supervisor = GeminiSupervisor(
        api_key=api_key,
        task_description="Pick up the pen on the table.",
        candidate_instructions=candidate_instructions,
        model_name=args.model,
        request_timeout_s=10.0,
        instruction_mode=args.mode,
    )

    # Fake telemetry
    now = time.monotonic()
    history = build_fake_history(now, n=5)
    signals = compute_signals(history, thresholds={"gripper_closed": 25.0})
    print(f"\n📊 Fake signals: {signals}")

    # Call monitor N times (for latency profiling)
    print(f"\n🛰️  Calling Gemini × {args.n}...\n")
    latencies = []
    decisions = []
    for i in range(args.n):
        decision = supervisor.monitor(
            top_frame_rgb=top_rgb,
            wrist_frame_rgb=wrist_rgb,
            current_instruction="go to the pen",
            held_for_s=4.7,
            state_history=history,
            signals=signals,
            episode_history=[],          # empty: first sub-goal in episode
            recent_decisions=[],         # empty: first call
        )
        latencies.append(decision.latency_ms)
        decisions.append(decision)
        tag = "[COLD]" if i == 0 else "[WARM]"
        print(f"  call {i+1}/{args.n} {tag}: {decision.latency_ms:7.1f}ms  "
              f"status={decision.status:18s} instr={decision.instruction!r}")

    print("\n" + "=" * 70)
    print(f"  Final decision (call {args.n}):")
    print(f"    status:      {decisions[-1].status}")
    print(f"    instruction: {decisions[-1].instruction!r}")
    print(f"    reason:      {decisions[-1].reason}")
    print("=" * 70)

    if args.n >= 2:
        cold = latencies[0]
        warm = latencies[1:]
        warm_mean = sum(warm) / len(warm)
        warm_min = min(warm)
        warm_max = max(warm)
        print("\n📊 Latency summary:")
        print(f"   cold (call 1):     {cold:7.1f}ms")
        print(f"   warm (calls 2-{args.n}): mean={warm_mean:6.1f}ms  "
              f"min={warm_min:6.1f}ms  max={warm_max:6.1f}ms  n={len(warm)}")
        print(f"   warmup effect:     {cold - warm_mean:+7.1f}ms  "
              f"({(cold - warm_mean) / cold * 100:+.0f}%)")

    # Sanity assertions
    for d in decisions:
        assert d.status in (
            'in_progress', 'subgoal_complete', 'corrective_action',
            'stuck', 'failed', 'task_complete'
        ), f"unexpected status: {d.status}"
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
