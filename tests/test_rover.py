"""Headless checks for the rover model, controller and terrains.

    MUJOCO_GL=egl python3 -m pytest tests/        (or: python3 tests/test_rover.py)
"""
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import terrains  # noqa: E402
from control import Command, RoverController  # noqa: E402

SCENE = os.path.join(os.path.dirname(HERE), "model", "scene.xml")


def load(terrain="flat"):
    model = mujoco.MjModel.from_xml_path(SCENE)
    t = terrains.make(terrain)
    terrains.apply(model, t)
    data = mujoco.MjData(model)
    data.qpos[2] = t.height_at(0, 0) + 0.17
    return model, data, t


def run(model, data, ctrl, cmd, seconds):
    for _ in range(int(seconds / model.opt.timestep)):
        ctrl.apply(data, cmd)
        mujoco.mj_step(model, data)


def yaw_deg(data, bid):
    q = data.xquat[bid]
    return np.degrees(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def test_rests_on_six_wheels():
    model, data, _ = load()
    ctrl = RoverController(model)
    run(model, data, ctrl, Command(), 2.0)
    assert abs(data.xpos[1][2] - 0.1587) < 0.004
    touching = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
                for c in data.contact[:data.ncon]}
    assert {f"tire_{w}" for w in ("FL", "FR", "ML", "MR", "RL", "RR")} <= touching


def test_drives_straight_at_commanded_speed():
    model, data, _ = load()
    ctrl = RoverController(model)
    run(model, data, ctrl, Command(), 1.0)
    run(model, data, ctrl, Command(throttle=0.5), 4.0)
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    vel = data.sensordata[model.sensor_adr[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "velocity")]:][:3]
    expected = 0.5 * 32.0 * ctrl.wheel_radius
    assert abs(vel[0] - expected) < 0.15 * expected
    assert abs(data.xpos[chassis][1]) < 0.05
    assert abs(yaw_deg(data, chassis)) < 3.0


def test_ackermann_turn_left_and_right():
    for steer in (1.0, -1.0):
        model, data, _ = load()
        ctrl = RoverController(model)
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        run(model, data, ctrl, Command(), 1.0)
        dist, total_yaw = 0.0, 0.0
        prev = data.xpos[chassis][:2].copy()
        yaw_prev = np.radians(yaw_deg(data, chassis))
        for _ in range(int(4.0 / model.opt.timestep)):
            ctrl.apply(data, Command(throttle=0.5, steer=steer))
            mujoco.mj_step(model, data)
            p = data.xpos[chassis][:2].copy()
            dist += np.linalg.norm(p - prev)
            prev = p
            yaw = np.radians(yaw_deg(data, chassis))
            total_yaw += (yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi
            yaw_prev = yaw
        assert np.sign(total_yaw) == np.sign(steer)
        assert dist > 2.5, dist                                   # kept rolling, no stall
        radius = dist / abs(total_yaw)
        commanded = 1.0 / ctrl.max_curvature
        assert abs(radius - commanded) < 0.1 * commanded, (radius, commanded)


def test_spin_in_place():
    model, data, _ = load()
    ctrl = RoverController(model)
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    run(model, data, ctrl, Command(), 1.0)
    total_yaw = 0.0
    yaw_prev = np.radians(yaw_deg(data, chassis))
    for _ in range(int(3.0 / model.opt.timestep)):
        ctrl.apply(data, Command(spin=1.0))
        mujoco.mj_step(model, data)
        yaw = np.radians(yaw_deg(data, chassis))
        total_yaw += (yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = yaw
    assert np.degrees(total_yaw) > 180, np.degrees(total_yaw)
    assert np.linalg.norm(data.xpos[chassis][:2]) < 0.15


def test_climbs_rock_garden_and_steps():
    for terrain in ("rocks", "obstacles"):
        model, data, t = load(terrain)
        ctrl = RoverController(model)
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        run(model, data, ctrl, Command(), 1.0)
        run(model, data, ctrl, Command(throttle=0.5), 4.0)
        x = data.xpos[chassis][0]
        assert x > 2.0, (terrain, x)
        up = data.xmat[chassis].reshape(3, 3)[2, 2]
        assert up > 0.7, (terrain, "rolled over")


def test_terrains_build():
    for name in terrains.TERRAIN_ORDER:
        t = terrains.make(name)
        assert t.heights.shape == (terrains.NROW, terrains.NCOL)
        assert t.heights.min() >= 0.0 and t.heights.max() <= terrains.ELEVATION
        # the spawn area is flat
        h0 = t.height_at(0, 0)
        for dx, dy in ((0.4, 0), (-0.4, 0), (0, 0.4), (0, -0.4), (0.3, 0.3)):
            assert abs(t.height_at(dx, dy) - h0) < 1e-6, name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
