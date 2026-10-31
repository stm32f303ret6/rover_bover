#!/usr/bin/env python3
"""Drive the Papaya Pathfinder rover in MuJoCo with the keyboard.

    python3 drive.py                    # mixed terrain (hills, rocks, ramp, steps)
    python3 drive.py --terrain rocks    # flat | hills | rocks | slope | obstacles | mixed
    python3 drive.py --list-terrains

Keys (hold to drive, release to coast):
    W / S        throttle forward / reverse          Up / Down     same
    A / D        steer left / right (Ackermann)      Left / Right  same
    Q / E        spin in place (point turn)
    Space        brake (zero every command)
    Shift        turbo (full speed instead of cruise)
    1 .. 6       switch terrain                       R             reset rover at spawn
    C            cycle camera: track / chase / onboard / free
    H            toggle help overlay                  G             toggle collision geoms
    P            pause                                Esc           quit
Mouse: left-drag rotate, right-drag pan, scroll zoom (free and track cameras).

The window is a plain GLFW loop around the MuJoCo renderer (instead of mujoco.viewer) so
key press *and* release events are available, which hold-to-drive needs.
"""
from __future__ import annotations

import argparse
import os
import time

import glfw
import mujoco
import numpy as np

import terrains
from control import Command, RoverController

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "model", "scene.xml")

CRUISE = 0.55            # fraction of full speed without Shift
RAMP_UP = 3.0            # command units per second while a key is held
RAMP_DOWN = 6.0          # command units per second after release
SPAWN_CLEARANCE = 0.17   # chassis origin height above ground at spawn (rests at 0.1587)
CAMERA_MODES = ("track", "chase", "onboard", "free")


class KeyState:
    """Tracks which keys are held; GLFW gives press/release events."""

    def __init__(self):
        self.down: set[int] = set()
        self.pressed: list[int] = []

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.down.add(key)
            self.pressed.append(key)
        elif action == glfw.RELEASE:
            self.down.discard(key)

    def held(self, *keys) -> bool:
        return any(k in self.down for k in keys)

    def take_presses(self) -> list[int]:
        p, self.pressed = self.pressed, []
        return p


class RoverSim:
    def __init__(self, terrain_name: str, width=1400, height=900):
        self.model = mujoco.MjModel.from_xml_path(SCENE)
        self.data = mujoco.MjData(self.model)
        self.controller = RoverController(self.model)
        self.chassis_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.cmd = Command()
        self.paused = False
        self.show_help = True
        self.camera_mode = 0
        self.terrain: terrains.Terrain | None = None

        if not glfw.init():
            raise RuntimeError("could not initialise GLFW")
        glfw.window_hint(glfw.SAMPLES, 4)
        self.window = glfw.create_window(width, height, "Papaya Pathfinder - MuJoCo", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("could not create a GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.keys = KeyState()
        glfw.set_key_callback(self.window, self.keys.on_key)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_move)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_scroll_callback(self.window, self._on_scroll)
        self._last_mouse = (0.0, 0.0)
        self._buttons = {"left": False, "middle": False, "right": False}

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[3] = 0          # collision primitives hidden by default
        self.scene = mujoco.MjvScene(self.model, maxgeom=20000)
        self.ctx = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.set_terrain(terrain_name)
        self._apply_camera_mode()

    # ------------------------------------------------------------------ terrain / reset
    def set_terrain(self, name: str) -> None:
        self.terrain = terrains.make(name)
        terrains.apply(self.model, self.terrain)
        hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        mujoco.mjr_uploadHField(self.model, self.ctx, hid)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (0.0, 0.0, self.terrain.height_at(0.0, 0.0) + SPAWN_CLEARANCE)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.cmd = Command()
        self.controller.apply(self.data, self.cmd)
        mujoco.mj_forward(self.model, self.data)
        self.wall_start = time.perf_counter() - self.data.time

    # ------------------------------------------------------------------ camera
    def _apply_camera_mode(self) -> None:
        mode = CAMERA_MODES[self.camera_mode]
        if mode == "track":
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = self.chassis_id
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 1.8, 135.0, -22.0
        elif mode in ("chase", "onboard"):
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, mode)
        else:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.cam.lookat[:] = self.data.xpos[self.chassis_id]
            self.cam.distance, self.cam.azimuth, self.cam.elevation = 2.5, 135.0, -25.0

    def _on_mouse_button(self, window, button, action, mods):
        name = {glfw.MOUSE_BUTTON_LEFT: "left", glfw.MOUSE_BUTTON_MIDDLE: "middle",
                glfw.MOUSE_BUTTON_RIGHT: "right"}.get(button)
        if name:
            self._buttons[name] = action == glfw.PRESS
        self._last_mouse = glfw.get_cursor_pos(window)

    def _on_mouse_move(self, window, x, y):
        dx, dy = x - self._last_mouse[0], y - self._last_mouse[1]
        self._last_mouse = (x, y)
        if not any(self._buttons.values()) or self.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            return
        w, h = glfw.get_window_size(window)
        shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        if self._buttons["right"]:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif self._buttons["left"]:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            act = mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.model, act, dx / h, dy / h, self.scene, self.cam)

    def _on_scroll(self, window, xoff, yoff):
        if self.cam.type != mujoco.mjtCamera.mjCAMERA_FIXED:
            mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0, -0.05 * yoff,
                                  self.scene, self.cam)

    # ------------------------------------------------------------------ input
    def _handle_presses(self) -> bool:
        for key in self.keys.take_presses():
            if key == glfw.KEY_ESCAPE:
                return False
            if key == glfw.KEY_R:
                self.reset()
            elif key == glfw.KEY_P:
                self.paused = not self.paused
                self.wall_start = time.perf_counter() - self.data.time
            elif key == glfw.KEY_C:
                self.camera_mode = (self.camera_mode + 1) % len(CAMERA_MODES)
                self._apply_camera_mode()
            elif key == glfw.KEY_H:
                self.show_help = not self.show_help
            elif key == glfw.KEY_G:
                self.opt.geomgroup[3] = not self.opt.geomgroup[3]
            elif glfw.KEY_1 <= key <= glfw.KEY_9:
                idx = key - glfw.KEY_1
                if idx < len(terrains.TERRAIN_ORDER):
                    self.set_terrain(terrains.TERRAIN_ORDER[idx])
        return True

    def _update_command(self, dt: float) -> None:
        k = self.keys
        limit = 1.0 if k.held(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT) else CRUISE
        if k.held(glfw.KEY_SPACE):
            self.cmd = Command()
            return
        axes = {
            "throttle": (k.held(glfw.KEY_W, glfw.KEY_UP), k.held(glfw.KEY_S, glfw.KEY_DOWN), limit),
            "steer": (k.held(glfw.KEY_A, glfw.KEY_LEFT), k.held(glfw.KEY_D, glfw.KEY_RIGHT), 1.0),
            "spin": (k.held(glfw.KEY_Q), k.held(glfw.KEY_E), limit),
        }
        for name, (pos, neg, lim) in axes.items():
            v = getattr(self.cmd, name)
            target = lim if pos and not neg else -lim if neg and not pos else 0.0
            rate = RAMP_UP if target != 0.0 else RAMP_DOWN
            if v < target:
                v = min(v + rate * dt, target)
            elif v > target:
                v = max(v - rate * dt, target)
            setattr(self.cmd, name, v)

    # ------------------------------------------------------------------ main loop
    def run(self) -> None:
        last = time.perf_counter()
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if not self._handle_presses():
                break
            now = time.perf_counter()
            frame_dt, last = now - last, now
            self._update_command(min(frame_dt, 0.1))

            if not self.paused:
                # keep the simulation clock locked to wall time, at most 0.1 s per frame
                target = now - self.wall_start
                steps = 0
                while self.data.time < target and steps < int(0.1 / self.model.opt.timestep):
                    self.controller.apply(self.data, self.cmd)
                    mujoco.mj_step(self.model, self.data)
                    steps += 1
                if self.data.time < target:
                    self.wall_start = now - self.data.time
            self._render()
            glfw.swap_buffers(self.window)
        glfw.terminate()

    def _render(self) -> None:
        w, h = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                               mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        mujoco.mjr_render(viewport, self.scene, self.ctx)
        self._overlay(viewport)

    def _overlay(self, viewport) -> None:
        d = self.data
        vel = d.sensordata[self.model.sensor_adr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "velocity")]:][:3]
        q = d.xquat[self.chassis_id]
        roll = np.degrees(np.arctan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
        pitch = np.degrees(np.arcsin(np.clip(2 * (q[0] * q[2] - q[3] * q[1]), -1, 1)))
        mode = "SPIN" if abs(self.cmd.spin) > 1e-3 and abs(self.cmd.throttle) < 1e-3 else "DRIVE"
        left = "\n".join([
            f"terrain [{terrains.TERRAIN_ORDER.index(self.terrain.name) + 1}] {self.terrain.name}",
            self.terrain.description,
            f"mode {mode}{'  (paused)' if self.paused else ''}",
            f"throttle {self.cmd.throttle:+.2f}   steer {self.cmd.steer:+.2f}   spin {self.cmd.spin:+.2f}",
            f"speed {np.linalg.norm(vel[:2]):.2f} m/s   roll {roll:+.0f} deg   pitch {pitch:+.0f} deg",
            f"pos ({d.xpos[self.chassis_id][0]:+.1f}, {d.xpos[self.chassis_id][1]:+.1f})   "
            f"camera {CAMERA_MODES[self.camera_mode]}   t = {d.time:.1f} s",
        ])
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                           viewport, left, "", self.ctx)
        if self.show_help:
            help_l = "W/S  A/D  Q/E\nShift\nSpace\n1-6\nR  C  G\nP  H  Esc"
            help_r = ("throttle, steer, spin\nturbo\nbrake\nterrain\n"
                      "reset, camera, collision geoms\npause, help, quit")
            mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                               viewport, help_l, help_r, self.ctx)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--terrain", default="mixed", choices=terrains.TERRAIN_ORDER)
    parser.add_argument("--list-terrains", action="store_true")
    args = parser.parse_args()
    if args.list_terrains:
        for i, name in enumerate(terrains.TERRAIN_ORDER, 1):
            print(f"  [{i}] {name:10s} {terrains.make(name).description}")
        return
    RoverSim(args.terrain).run()


if __name__ == "__main__":
    main()
