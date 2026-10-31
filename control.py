"""Drive/steer mixing for the six-wheel, four-corner-steer Papaya Pathfinder.

The rover has six driven wheels (GA25 gear motors) and four steering servos on the corner
wheels; the middle wheels are fixed. All geometry is read from the MuJoCo model, so the
controller stays correct if tools/build_rover.py changes the wheel layout.

Commands are normalised, like the gamepad axes the real firmware consumes:
    throttle in [-1, 1]   forward speed of the chassis
    steer    in [-1, 1]   left (+) / right (-) turn, Ackermann geometry about the mid axle
    spin     in [-1, 1]   point turn in place (counter-clockwise +), wheels on a circle

Unlike the firmware (which writes the same servo offset to every corner), the sim solves the
true Ackermann geometry: every wheel is tangent to the circle around the instantaneous
centre of rotation and rolls at the matching speed, so nothing scrubs.
Frames: chassis frame, x forward, y left, metres, radians.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

WHEELS = ("FL", "FR", "ML", "MR", "RL", "RR")
STEERED = ("FL", "FR", "RL", "RR")

MAX_WHEEL_SPEED = 32.0     # rad/s, ~1.7 m/s on 54 mm tires (GA25 20:1 on 3S, no load)
MAX_STEER_ANGLE = 0.61     # rad (35 deg), SERVO_MAX_DELTA in the upstream firmware
MAX_SPIN_RATE = 2.5        # rad/s chassis yaw rate at full spin command


@dataclass
class Command:
    throttle: float = 0.0
    steer: float = 0.0
    spin: float = 0.0


class RoverController:
    """Maps a Command to actuator controls (wheel rad/s and steering angles)."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.wheel_radius = float(model.geom_size[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tire_FL")][0])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        origin = data.xpos[chassis]
        self.pos = {}
        for w in WHEELS:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{w}")
            self.pos[w] = (data.xpos[bid] - origin)[:2].copy()   # (dx, dy) in chassis frame
        self.x_mid = 0.5 * (self.pos["ML"][0] + self.pos["MR"][0])
        # the wheel centre sits beside the steering axis (knuckle offset), so steering moves
        # the contact patch: pivot position + offset in the knuckle frame, both chassis-frame
        # at zero steer
        self.pivot, self.offset = {}, {}
        for w in STEERED:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"knuckle_{w}")
            wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{w}")
            self.pivot[w] = (data.xpos[kid] - origin)[:2].copy()
            self.offset[w] = model.body_pos[wid][:2].copy()
        self.drive_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{w}")
                          for w in WHEELS]
        self.steer_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"steer_{w}")
                          for w in STEERED]
        # curvature at |steer| = 1 is the one that puts the inner front wheel at
        # MAX_STEER_ANGLE; the ICR always lies on the (non-steered) middle axle line
        dx = abs(self.pos["FL"][0] - self.x_mid)
        dy = abs(self.pos["FL"][1])
        self.max_curvature = 1.0 / (dx / np.tan(MAX_STEER_ANGLE) + dy)

    def wheel_targets(self, cmd: Command) -> tuple[dict, dict]:
        """Return (wheel angular velocity rad/s, steering angle rad) per wheel."""
        throttle = float(np.clip(cmd.throttle, -1.0, 1.0))
        steer = float(np.clip(cmd.steer, -1.0, 1.0))
        spin = float(np.clip(cmd.spin, -1.0, 1.0))
        speeds, angles = {}, {}
        if abs(spin) > 1e-3 and abs(throttle) < 1e-3:
            # point turn: ICR at the chassis centre on the mid axle
            omega = spin * MAX_SPIN_RATE
            for w in WHEELS:
                delta, (dx, dy) = self._steer(w, lambda dx, dy: np.arctan2(dx, -dy))
                v = omega * (-dy * np.cos(delta) + dx * np.sin(delta))
                speeds[w], angles[w] = v / self.wheel_radius, delta
        else:
            v = throttle * MAX_WHEEL_SPEED * self.wheel_radius
            kappa = steer * self.max_curvature
            for w in WHEELS:
                delta, (dx, dy) = self._steer(
                    w, lambda dx, dy: np.arctan2(kappa * dx, 1.0 - kappa * dy))
                vi = v * ((1.0 - kappa * dy) * np.cos(delta) + kappa * dx * np.sin(delta))
                speeds[w], angles[w] = vi / self.wheel_radius, delta
        # servos keep their steering pose while stopped so a turn can start from rest
        if abs(throttle) < 1e-3 and abs(spin) < 1e-3:
            kappa = steer * self.max_curvature
            for w in STEERED:
                angles[w], _ = self._steer(w, lambda dx, dy: np.arctan2(kappa * dx, 1.0 - kappa * dy))
        return speeds, angles

    def _wheel_xy(self, w: str, delta: float) -> np.ndarray:
        """Wheel centre in the chassis frame for steering angle delta (offset knuckle)."""
        if w not in STEERED:
            return self.pos[w]
        c, s = np.cos(delta), np.sin(delta)
        ox, oy = self.offset[w]
        return self.pivot[w] + np.array([c * ox - s * oy, s * ox + c * oy])

    def _steer(self, w: str, law) -> tuple[float, tuple[float, float]]:
        """Steering angle from law(dx, dy), solved at the position the wheel actually
        takes once steered (fixed-point iteration, the offset is small vs the turn radius)."""
        delta = 0.0
        for _ in range(8 if w in STEERED else 1):
            x, y = self._wheel_xy(w, delta)
            dx, dy = x - self.x_mid, y
            if w not in STEERED:
                return 0.0, (dx, dy)
            delta = (law(dx, dy) + np.pi / 2) % np.pi - np.pi / 2       # keep within +-90 deg
        return delta, (dx, dy)

    def apply(self, data: mujoco.MjData, cmd: Command) -> None:
        speeds, angles = self.wheel_targets(cmd)
        for aid, w in zip(self.drive_ids, WHEELS):
            data.ctrl[aid] = speeds[w]
        for aid, w in zip(self.steer_ids, STEERED):
            data.ctrl[aid] = angles[w]
