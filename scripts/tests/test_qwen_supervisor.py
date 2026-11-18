"""
Smoke test for QwenSupervisor — end-to-end on real images.

Loads Qwen2-VL-2B-Instruct, calls monitor() N times, reports latency + decisions.
Use --step to pick image, --n for cold-vs-warm comparison.

Usage:
    python src/scripts/test_qwen_supervisor.py
    python src/scripts/test_qwen_supervisor.py --step 02 --n 5
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.qwen_supervisor import QwenSupervisor                # noqa: E402
from agents.supervisor_agent import StateSnapshot, compute_signals  # noqa: E402

REPO_ROOT = SRC_DIR.parent
DEBUG_FRAMES = REPO_ROOT / "debug_frames"


def load_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    return np.array(Image.open(path).convert("RGB"))


def build_fake_history(now: float, n: int = 5) -> list:
    history = []
    for i in range(n):
        t = now - (n - 1 - i) * 0.2
        history.append(StateSnapshot(
            t=t,
            instruction="go to the pen",
            q_deg=[10.0 + i, -5.0, 30.0 + i, 0.0, -25.0, 0.0],
            gripper_val=5.0,
            ee_pos=[0.18, 0.02, 0.22 - 0.01 * i],
            ee_rpy=[0.0, 90.0, 0.0],
        ))
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="01")
    parser.add_argument("--mode", default="strict",
                        choices=["strict", "warn", "flexible"])
    parser.add_argument("--n", type=int, default=3,
                        help="number of consecutive calls (warm-vs-cold)")
    parser.add_argument("--top", default=None)
    parser.add_argument("--wrist", default=None,
                        help="(default: front_raw, same physical camera)")
    args = parser.parse_args()

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

    candidate_instructions = [
        "go to the pen",
        "approach pen from above",
        "open gripper and lower",
        "grasp pen",
        "lift pen up",
    ]
    print(f"\n🧠 Constructing QwenSupervisor (mode={args.mode})...")
    supervisor = QwenSupervisor(
        task_description="Pick up the pen on the table.",
        candidate_instructions=candidate_instructions,
        instruction_mode=args.mode,
    )

    now = time.monotonic()
    history = build_fake_history(now, n=5)
    signals = compute_signals(history, thresholds={"gripper_closed": 25.0})
    print(f"\n📊 Fake signals: {signals}")

    print(f"\n🛰️  Calling QwenSupervisor × {args.n}...\n")
    latencies = []
    decisions = []
    for i in range(args.n):
        d = supervisor.monitor(
            top_frame_rgb=top_rgb,
            wrist_frame_rgb=wrist_rgb,
            current_instruction="go to the pen",
            held_for_s=4.7,
            state_history=history,
            signals=signals,
            episode_history=[],
            recent_decisions=[],
        )
        latencies.append(d.latency_ms)
        decisions.append(d)
        tag = "[COLD]" if i == 0 else "[WARM]"
        print(f"  call {i+1}/{args.n} {tag}: {d.latency_ms:7.1f}ms  "
              f"status={d.status:18s} instr={d.instruction!r}")
        print(f"           reason: {d.reason}")
        print(f"           raw:    {d.raw_response!r}")
        print()

    print("=" * 70)
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

    for d in decisions:
        assert d.status in (
            'in_progress', 'subgoal_complete', 'corrective_action',
            'stuck', 'failed', 'task_complete'
        ), f"unexpected status: {d.status}"
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
