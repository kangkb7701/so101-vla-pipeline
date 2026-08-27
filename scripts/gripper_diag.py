#!/usr/bin/env python3
r"""Gripper health check for the SO-101 follower.

Why this exists: the gripper is the one motor configure() puts protections on
(Max_Torque_Limit=500, Protection_Current=250, Overload_Torque=25). When the
overload protection trips mid-grasp, the servo drops to 25% torque and the
gripper sits closed, ignoring open commands until the protection clears - which
looks exactly like "policy won't open the gripper". This script reads the
protection state directly so that can be confirmed or ruled out.

Run with the edge agent STOPPED (only one process can own the serial port):

    conda activate lerobot312
    set LEROBOT_CALIBRATION_DIR=<repo>\calibration
    python scripts/gripper_diag.py                 # read-only status dump
    python scripts/gripper_diag.py --exercise      # + slow open/close sweep

--exercise moves ONLY the gripper fingers (open 80 -> close 20 -> back), arm
joints stay torque-off. Keep fingers clear anyway.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

STATUS_BITS = {  # Feetech STS series Status register (addr 65)
    0: "voltage out of range",
    1: "sensor error",
    2: "over temperature",
    3: "over current",
    5: "overload protection ACTIVE",
}


def read(bus, reg: str) -> int:
    return int(bus.read(reg, "gripper", normalize=False))


def dump_status(bus, label: str) -> int:
    pos = float(bus.read("Present_Position", "gripper"))  # normalized 0-100
    raw = {r: read(bus, r) for r in (
        "Present_Load", "Present_Current", "Present_Temperature",
        "Present_Voltage", "Status", "Moving", "Torque_Enable",
    )}
    load = raw["Present_Load"] & 0x3FF  # bit10 = direction
    cur = raw["Present_Current"] * 6.5  # mA per LSB on STS3215
    print(f"\n[{label}]")
    print(f"  position   : {pos:6.1f} / 100   (0=closed)")
    print(f"  load       : {load}  (protection kicks in around sustained high load)")
    print(f"  current    : {cur:.0f} mA  (Protection_Current is set to 250 LSB = ~1625 mA)")
    print(f"  temperature: {raw['Present_Temperature']} C  (protection typically at 70+ C)")
    print(f"  voltage    : {raw['Present_Voltage'] / 10:.1f} V")
    print(f"  torque on  : {raw['Torque_Enable']}   moving: {raw['Moving']}")
    status = raw["Status"]
    if status:
        flags = [name for bit, name in STATUS_BITS.items() if status & (1 << bit)]
        print(f"  STATUS     : 0x{status:02X}  <<< {', '.join(flags) or f'unknown bits set'}")
    else:
        print("  STATUS     : 0x00 (no protection flags)")
    return status


def exercise(bus) -> None:
    print("\n--- exercise: open 80 -> close 20 -> open 50 (gripper only) ---")
    bus.write("Torque_Enable", "gripper", 1, normalize=False)
    try:
        for target in (80.0, 20.0, 50.0):
            bus.write("Goal_Position", "gripper", target)
            worst_err, worst_load = 0.0, 0
            for _ in range(20):  # 2s per leg
                time.sleep(0.1)
                pos = float(bus.read("Present_Position", "gripper"))
                load = read(bus, "Present_Load") & 0x3FF
                worst_err = max(worst_err, abs(target - pos))
                worst_load = max(worst_load, load)
            final = float(bus.read("Present_Position", "gripper"))
            ok = abs(target - final) < 5.0
            print(f"  goal {target:5.1f} -> reached {final:5.1f}  "
                  f"{'OK ' if ok else 'FAIL'}  peak load {worst_load}")
            if not ok:
                print("         ^ not tracking: protection state, mechanical bind, or torque too low")
    finally:
        bus.write("Torque_Enable", "gripper", 0, normalize=False)
        print("  torque released")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot_port", default=os.getenv("ROBOT_PORT", "COM3"))
    parser.add_argument("--robot_id", default=os.getenv("ROBOT_ID", "my_follower"))
    parser.add_argument("--calibration_dir", default=os.getenv("LEROBOT_CALIBRATION_DIR"))
    parser.add_argument("--exercise", action="store_true", help="Also run a slow open/close sweep (moves the fingers).")
    args = parser.parse_args()

    robot = make_robot_from_config(
        SOFollowerRobotConfig(
            id=args.robot_id,
            port=args.robot_port,
            calibration_dir=Path(args.calibration_dir) if args.calibration_dir else None,
            disable_torque_on_disconnect=True,
            use_degrees=True,
            cameras={},
        )
    )
    robot.bus.connect()
    try:
        status = dump_status(robot.bus, "as found")
        if args.exercise:
            exercise(robot.bus)
            dump_status(robot.bus, "after exercise")
        if status & (1 << 5) or status & (1 << 3):
            print("\n=> overload/current protection is (or was) active. The servo held only "
                  "25% torque (Overload_Torque). Power-cycle the arm PSU to clear it, then "
                  "consider whether the policy squeezes harder than the object needs.")
    finally:
        robot.bus.disconnect()


if __name__ == "__main__":
    main()
