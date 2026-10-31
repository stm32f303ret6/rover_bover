"""Watch servo steering and skid steering side by side in one MuJoCo scene.

Two copies of the rover share the terrain: the orange one is the real rover (four steering
servos, control.py), the red one has its steering joints welded straight and turns only by a
left/right wheel speed difference. Both receive the same demanded chassis speed and yaw rate.

    python3 tools/show_steering.py                      # live viewer, cycles every scenario
    python3 tools/show_steering.py --scenario point_turn --mu 0.8
    MUJOCO_GL=egl python3 tools/show_steering.py --video DIR   # one mp4 per scenario
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import terrains  # noqa: E402
from control import Command, RoverController, WHEELS, MAX_WHEEL_SPEED, MAX_SPIN_RATE  # noqa: E402
from compare_steering import spawn_pose  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "model")
GAP = 0.8               # m, each rover this far either side of the scenario centre
SETTLE = 1.0

SCENARIOS = {
    "point_turn":      dict(terrain="flat", centre=(0, 0), v=0.0, omega=MAX_SPIN_RATE, seconds=5, mu=1.2,
                            title="point turn on flat ground, tire friction 1.2"),
    "point_turn_mu08": dict(terrain="flat", centre=(0, 0), v=0.0, omega=MAX_SPIN_RATE, seconds=5, mu=0.8,
                            title="point turn on flat ground, tire friction 0.8"),
    "arc":             dict(terrain="flat", centre=(0, 0), v=0.6, omega=0.6 / 0.42, seconds=6, mu=1.2,
                            title="0.42 m radius arc at 0.6 m/s (tightest Ackermann radius)"),
    "arc_rocks":       dict(terrain="rocks", centre=(0, 1.5), v=0.6, omega=0.6 / 1.0, seconds=6, mu=1.2,
                            title="1.0 m radius arc through the rock garden"),
    "climb":           dict(terrain="obstacles", centre=(0, 0), v=0.86, omega=0.0, seconds=4, mu=1.2,
                            title="straight over the steps (identical by design)"),
}


# ----------------------------------------------------------------------------- scene
def two_rover_xml() -> str:
    rover = open(os.path.join(MODEL_DIR, "rover.xml")).read()
    rover = rover.replace('meshdir="assets"', f'meshdir="{os.path.join(MODEL_DIR, "assets")}"')
    head = rover.split('<mujoco model="papaya_pathfinder">', 1)[1].split("<worldbody>", 1)[0]
    head = head.replace("</asset>", '    <material name="frame_skid" rgba="0.80 0.15 0.15 1" '
                                     'specular="0.3" shininess="0.3"/>\n  </asset>')
    parts = {tag: re.search(rf"<{tag}>(.*?)</{tag}>", rover, re.S).group(1)
             for tag in ("worldbody", "equality", "contact", "actuator", "sensor")}

    def variant(text: str, skid: bool) -> str:
        if skid:
            text = re.sub(r'\n[ \t]*<joint name="steer_[A-Z]+" class="steer"/>', "", text)
            text = re.sub(r'\n[ \t]*<position name="steer_[A-Z]+" joint="steer_[A-Z]+" class="servo"/>', "", text)
            text = re.sub(r'\b(name|joint|body1|body2|site|objname)="([^"]+)"',
                          lambda m: f'{m.group(1)}="s_{m.group(2)}"', text)
            text = text.replace('material="frame"', 'material="frame_skid"')
        return text

    body = head
    for tag, txt in parts.items():
        body += f"<{tag}>{variant(txt, False)}{variant(txt, True)}</{tag}>\n"
    scene = open(os.path.join(MODEL_DIR, "scene.xml")).read()
    return scene.replace('<include file="rover.xml"/>', body)


class SkidDrive:
    def __init__(self, model, prefix="s_"):
        self.r = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{prefix}tire_FL")][0])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        c = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}chassis")]
        ys = [(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}wheel_{w}")] - c)[1]
              for w in WHEELS]
        self.half_track = float(np.mean(np.abs(ys)))
        self.ids = {w: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{prefix}drive_{w}") for w in WHEELS}

    def apply(self, data, v, omega):
        d = omega * self.half_track / self.r
        for w, aid in self.ids.items():
            data.ctrl[aid] = np.clip(v / self.r + (-d if w.endswith("L") else d), -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)


class ServoDrive:
    def __init__(self, model):
        self.c = RoverController(model)
        self.r = self.c.wheel_radius

    def apply(self, data, v, omega):
        if abs(v) < 1e-6:
            cmd = Command(spin=np.clip(omega / MAX_SPIN_RATE, -1, 1))
        else:
            cmd = Command(throttle=v / (MAX_WHEEL_SPEED * self.r),
                          steer=np.clip((omega / v) / self.c.max_curvature, -1, 1))
        self.c.apply(data, cmd)


class Scene:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_string(two_rover_xml())
        self.data = mujoco.MjData(self.model)
        self.servo, self.skid = ServoDrive(self.model), SkidDrive(self.model)
        self.chassis = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("chassis", "s_chassis")]
        self.qadr = [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
                     for n in ("chassis", "s_chassis")]
        self.tires = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{p}tire_{w}")
                      for p in ("", "s_") for w in WHEELS]
        self.hid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_HFIELD, "terrain")
        self.terrain = None

    def setup(self, sc: dict):
        self.terrain = terrains.make(sc["terrain"])
        terrains.apply(self.model, self.terrain)
        self.model.geom_friction[self.tires, 0] = sc["mu"]
        mujoco.mj_resetData(self.model, self.data)
        cx, cy = sc["centre"]
        for adr, y in zip(self.qadr, (cy + GAP, cy - GAP)):
            pos, quat = spawn_pose(self.terrain, cx, y)
            self.data.qpos[adr:adr + 3], self.data.qpos[adr + 3:adr + 7] = pos, quat
        mujoco.mj_forward(self.model, self.data)
        self.yaw0 = [self.yaw(i) for i in range(2)]
        self.yaw_acc = [0.0, 0.0]
        self.yaw_prev = list(self.yaw0)

    def yaw(self, i):
        q = self.data.xquat[self.chassis[i]]
        return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def step(self, v, omega):
        self.servo.apply(self.data, v, omega)
        self.skid.apply(self.data, v, omega)
        mujoco.mj_step(self.model, self.data)
        for i in range(2):
            y = self.yaw(i)
            self.yaw_acc[i] += (y - self.yaw_prev[i] + np.pi) % (2 * np.pi) - np.pi
            self.yaw_prev[i] = y

    def midpoint(self):
        return 0.5 * (self.data.xpos[self.chassis[0]] + self.data.xpos[self.chassis[1]])

    def frame_camera(self, cam):
        """Free camera looking at the midpoint, backing off as the rovers separate."""
        p0, p1 = self.data.xpos[self.chassis[0]], self.data.xpos[self.chassis[1]]
        cam.lookat[:] = 0.5 * (p0 + p1) + np.array([0, 0, 0.05])
        cam.distance = float(np.clip(1.1 * np.linalg.norm(p0 - p1) + 1.0, 2.4, 7.0))


# ----------------------------------------------------------------------------- video
def render_video(scene: Scene, name: str, sc: dict, path: str, width=1280, height=720, fps=30):
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=26)
    small = ImageFont.load_default(size=22)
    renderer = mujoco.Renderer(scene.model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 3.6, 150, -32
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                           "-pix_fmt", "yuv420p", "-crf", "20", path], stdin=subprocess.PIPE)
    scene.setup(sc)
    dt = scene.model.opt.timestep
    steps_per_frame = int(round(1.0 / (fps * dt)))
    total = int((SETTLE + sc["seconds"]) / dt)
    for k in range(total):
        t = k * dt
        v, om = (0.0, 0.0) if t < SETTLE else (sc["v"], sc["omega"])
        scene.step(v, om)
        if k % steps_per_frame == 0:
            scene.frame_camera(cam)
            renderer.update_scene(scene.data, cam, opt)
            img = Image.fromarray(renderer.render())
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, width, 92], fill=(0, 0, 0, 160))
            d.text((16, 8), sc["title"], font=font, fill=(255, 255, 255))
            d.text((16, 44), f"orange = servo steering   yaw {np.degrees(scene.yaw_acc[0]):6.0f} deg",
                   font=small, fill=(255, 190, 90))
            d.text((620, 44), f"red = skid steering   yaw {np.degrees(scene.yaw_acc[1]):6.0f} deg",
                   font=small, fill=(255, 110, 110))
            d.text((width - 170, 8), f"t = {max(0.0, t - SETTLE):4.1f} s", font=small, fill=(255, 255, 255))
            ff.stdin.write(img.tobytes())
    ff.stdin.close()
    ff.wait()
    renderer.close()
    print(f"{name}: servo {np.degrees(scene.yaw_acc[0]):.0f} deg, skid {np.degrees(scene.yaw_acc[1]):.0f} deg -> {path}")


# ---------------------------------------------------------------------------- viewer
def live(scene: Scene, names: list[str]):
    import mujoco.viewer
    with mujoco.viewer.launch_passive(scene.model, scene.data) as v:
        v.opt.geomgroup[3] = 0
        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.6, 150, -32
        while v.is_running():
            for name in names:
                sc = SCENARIOS[name]
                print(f"\n{name}: {sc['title']}   (orange = servo, red = skid)")
                scene.setup(sc)
                v.update_hfield(scene.hid)
                dt = scene.model.opt.timestep
                t0 = time.time()
                total = int((SETTLE + sc["seconds"] + 1.0) / dt)
                for k in range(total):
                    if not v.is_running():
                        return
                    t = k * dt
                    v_cmd, om = (sc["v"], sc["omega"]) if SETTLE <= t < SETTLE + sc["seconds"] else (0.0, 0.0)
                    scene.step(v_cmd, om)
                    if k % 10 == 0:
                        scene.frame_camera(v.cam)
                        v.sync()
                    lag = t0 + t - time.time()
                    if lag > 0:
                        time.sleep(lag)
                print(f"   servo {np.degrees(scene.yaw_acc[0]):6.0f} deg   skid {np.degrees(scene.yaw_acc[1]):6.0f} deg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS), action="append", help="default: all")
    ap.add_argument("--mu", type=float, help="override tire friction")
    ap.add_argument("--video", metavar="DIR", help="render mp4 files instead of opening a window")
    args = ap.parse_args()
    names = args.scenario or list(SCENARIOS)
    if args.mu is not None:
        for n in names:
            SCENARIOS[n]["mu"] = args.mu
            SCENARIOS[n]["title"] = re.sub(r"friction [0-9.]+", f"friction {args.mu}", SCENARIOS[n]["title"])
    scene = Scene()
    if args.video:
        os.makedirs(args.video, exist_ok=True)
        for n in names:
            render_video(scene, n, SCENARIOS[n], os.path.join(args.video, f"{n}.mp4"))
    else:
        live(scene, names)


if __name__ == "__main__":
    main()
