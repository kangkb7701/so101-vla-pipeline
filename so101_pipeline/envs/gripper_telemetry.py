from __future__ import annotations


def _read_gripper_register(robot, register_name: str):
    try:
        values = robot.bus.sync_read(register_name, "gripper")
        if isinstance(values, dict) and "gripper" in values:
            return float(values["gripper"]), None
        return None, f"missing gripper in {register_name} response"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def read_gripper_telemetry(robot, obs=None) -> dict:
    telemetry = {}
    errors = {}

    if obs is not None and "gripper.pos" in obs:
        try:
            telemetry["pos"] = float(obs["gripper.pos"])
        except Exception as e:
            errors["pos"] = f"{type(e).__name__}: {e}"

    for register_name, key in (
        ("Present_Current", "current"),
        ("Present_Load", "load"),
    ):
        value, err = _read_gripper_register(robot, register_name)
        telemetry[key] = value
        if err is not None:
            errors[key] = err

    if errors:
        telemetry["errors"] = errors
    return telemetry
