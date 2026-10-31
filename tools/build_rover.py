#!/usr/bin/env python3
"""Build the Papaya Pathfinder MJCF (model/rover.xml) and its meshes (model/assets/).

The upstream STL files (third_party/papaya_pathfinder/) are exported in the CAD assembly
frame, in millimetres:

    x_a  lateral, rover centreline at x_a = 137.5, larger x_a is the rover's LEFT side
    y_a  longitudinal, front steering axis at y_a = 0, rear at y_a = 340
    z_a  up, wheel axles at z_a = 0

Every pivot below was measured from the holes/bores in those meshes (see README).

This script re-expresses each part in the MuJoCo frame of the body that carries it
(x forward, y left, z up, metres) so the generated XML only contains joint locations.
Run it again whenever a mesh, a pivot or a mass changes:

    python3 tools/build_rover.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import trimesh

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "third_party", "papaya_pathfinder")
ASSETS = os.path.join(ROOT, "model", "assets")
XML_OUT = os.path.join(ROOT, "model", "rover.xml")

# --------------------------------------------------------------------------------------
# Measured assembly-frame geometry (mm)
# --------------------------------------------------------------------------------------
X_CENTER = 137.5        # rover centreline
Y_MIRROR = 170.0        # front/rear symmetry plane of the corner mounts (y = 0 <-> y = 340)

ROCKER_PIVOT = (207.5, 155.0, 104.7)   # arm <-> body hinge, axis lateral (left side x)
BOGIE_PIVOT = (207.5, 275.0, 105.0)    # arm <-> pivot (bogie) hinge, axis lateral
STEER_X = 258.7                        # steering (servo) axes, vertical
STEER_Z = 75.0
Y_FRONT, Y_MID, Y_REAR = 0.0, 180.0, 340.0
WHEEL_X = 297.0                        # wheel centre (left side); right side mirrored
WHEEL_R = 0.054                        # tire outer radius (m), 108 mm diameter
WHEEL_W = 0.040                        # tire width (m)
DIFF_PIVOT = (137.5, 195.0, 154.5)     # differential bar hinge on the body top, vertical axis
DIFF_ARM = 82.5                        # bar hole distance from its hinge (x_a = 220 / 55)
LINK_ATTACH = (218.0, 155.0, 156.0)    # link <-> arm hole (directly above rocker pivot)

# masses (kg) carried by the simple inertial/collision primitives of each body
M_CHASSIS = 1.20   # printed body + ESP32 + 2x BTS7960 + 3S LiPo + UBECs
M_ROCKER = 0.16    # arm + DS3218/MS24 servo
M_BOGIE = 0.22     # pivot part + rear servo + mid GA25 motor
M_STEER = 0.14     # motor mount + horn adapter + GA25 motor
M_WHEEL = 0.15     # rim + TPU tire
M_DIFF_BAR = 0.01
M_LINK = 0.005

# --------------------------------------------------------------------------------------
# Frame conversion helpers
# --------------------------------------------------------------------------------------
# assembly (x_a, y_a, z_a) -> mujoco (x, y, z) = (-y_a, x_a, z_a); a proper rotation
R_A2M = np.array([[0.0, -1.0, 0.0],
                  [1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0]])


def a2m(p_a, origin_a):
    """Assembly-frame point (mm) -> MuJoCo-frame point (m) relative to origin_a."""
    d = np.asarray(p_a, dtype=float) - np.asarray(origin_a, dtype=float)
    return R_A2M @ d / 1000.0


def mirror_lr(p):
    """Mirror an assembly point about the rover centreline (left <-> right)."""
    return (2 * X_CENTER - p[0], p[1], p[2])


def mirror_fr(p):
    """Mirror an assembly point about the corner symmetry plane (front <-> rear)."""
    return (p[0], 2 * Y_MIRROR - p[1], p[2])


def fmt(v):
    return " ".join(f"{float(x):.4g}" for x in v)


# --------------------------------------------------------------------------------------
# Mesh processing
# --------------------------------------------------------------------------------------
def load(name):
    return trimesh.load(os.path.join(SRC, name + ".stl"), force="mesh")


def export(mesh, name):
    mesh.export(os.path.join(ASSETS, name + ".stl"))
    print(f"  {name + '.stl':30s} faces={len(mesh.faces):6d}")


def to_body(mesh, origin_a, mirror_x=False, mirror_y=False):
    """Move an assembly-frame mesh into a body frame placed at origin_a (mm, assembly).

    Mirroring is applied in the assembly frame before the change of frame; trimesh flips
    the face winding automatically when the transform has a negative determinant.
    """
    m = mesh.copy()
    T = np.eye(4)
    if mirror_x:
        T[0, 0] = -1.0
        T[0, 3] = 2 * X_CENTER
    if mirror_y:
        T[1, 1] = -1.0
        T[1, 3] = 2 * Y_MIRROR
    m.apply_transform(T)
    T2 = np.eye(4)
    T2[:3, :3] = R_A2M / 1000.0
    T2[:3, 3] = -(R_A2M @ np.asarray(origin_a, dtype=float)) / 1000.0
    m.apply_transform(T2)
    return m


def wheel_mesh(mesh, left):
    """Rim/tire STLs are stored unassembled (axis z, hub face at z = 0). Centre them and
    rotate the axis onto +y (lateral); the hub face ends up inboard."""
    m = mesh.copy()
    lo, hi = m.bounds
    m.apply_translation(-(lo + hi) / 2.0)
    sign = -1.0 if left else 1.0          # rotation about x: z -> -/+ y
    rot = trimesh.transformations.rotation_matrix(sign * np.pi / 2.0, [1, 0, 0])
    m.apply_transform(rot)
    m.apply_scale(0.001)
    return m


def build_meshes():
    os.makedirs(ASSETS, exist_ok=True)
    print("meshes:")
    chassis_o = (X_CENTER, ROCKER_PIVOT[1], ROCKER_PIVOT[2])
    export(to_body(load("body"), chassis_o), "chassis")
    export(to_body(load("differential_bar"), DIFF_PIVOT), "diff_bar")

    link_o_l = (220.0, DIFF_PIVOT[1], DIFF_PIVOT[2])     # left bar-end hole
    export(to_body(load("differential_link_left"), link_o_l), "diff_link_L")
    export(to_body(load("differential_link_right"), mirror_lr(link_o_l)), "diff_link_R")

    arm, pivot = load("arm_left"), load("pivot_left")
    arm_r, pivot_r = load("arm_right"), load("pivot_right")
    servo, motor = load("servo_ds3218mg"), load("motor_ga25")
    mount, horn = load("motor_mount"), load("horn_adapter")

    # Left rocker: arm + front servo (servo STEP instance sits at the front-RIGHT corner)
    export(to_body(arm, ROCKER_PIVOT), "rocker_L")
    export(to_body(servo, ROCKER_PIVOT, mirror_x=True), "servo_front_L")
    export(to_body(arm_r, mirror_lr(ROCKER_PIVOT)), "rocker_R")
    export(to_body(servo, mirror_lr(ROCKER_PIVOT)), "servo_front_R")

    # Left bogie: pivot part + rear servo (front instance mirrored to the rear) + mid motor
    export(to_body(pivot, BOGIE_PIVOT), "bogie_L")
    export(to_body(servo, BOGIE_PIVOT, mirror_x=True, mirror_y=True), "servo_rear_L")
    mid_motor = motor.copy()
    mid_motor.apply_translation([0.0, Y_MID, 0.0])
    export(to_body(mid_motor, BOGIE_PIVOT, mirror_x=True), "motor_mid_L")
    export(to_body(pivot_r, mirror_lr(BOGIE_PIVOT)), "bogie_R")
    export(to_body(servo, mirror_lr(BOGIE_PIVOT), mirror_y=True), "servo_rear_R")
    export(to_body(mid_motor, mirror_lr(BOGIE_PIVOT)), "motor_mid_R")

    # Steering knuckles: motor mount + horn adapter + GA25 motor (STEP instances are
    # front-left for the mount and front-right for the horn/motor)
    steer_fl = (STEER_X, Y_FRONT, STEER_Z)
    steer_rl = (STEER_X, Y_REAR, STEER_Z)
    knuckle_l = [mount, to_mirrored(horn, True), to_mirrored(motor, True)]
    knuckle_r = [to_mirrored(mount, True), horn, motor]
    export(trimesh.util.concatenate([to_body(m, steer_fl) for m in knuckle_l]), "knuckle_FL")
    export(trimesh.util.concatenate([to_body(m, steer_rl, mirror_y=True) for m in knuckle_l]),
           "knuckle_RL")
    export(trimesh.util.concatenate([to_body(m, mirror_lr(steer_fl)) for m in knuckle_r]),
           "knuckle_FR")
    export(trimesh.util.concatenate(
        [to_body(m, mirror_lr(steer_rl), mirror_y=True) for m in knuckle_r]), "knuckle_RR")

    rim, tire = load("rim"), load("tire")
    export(wheel_mesh(rim, left=True), "rim_L")
    export(wheel_mesh(rim, left=False), "rim_R")
    export(wheel_mesh(tire, left=True), "tire_L")
    export(wheel_mesh(tire, left=False), "tire_R")


def to_mirrored(mesh, mirror_x):
    """Return a copy of an assembly-frame mesh mirrored about the centreline."""
    m = mesh.copy()
    if mirror_x:
        T = np.eye(4)
        T[0, 0] = -1.0
        T[0, 3] = 2 * X_CENTER
        m.apply_transform(T)
    return m


# --------------------------------------------------------------------------------------
# MJCF generation
# --------------------------------------------------------------------------------------
@dataclass
class Side:
    name: str            # "L" / "R"
    sgn: float           # +1 left, -1 right (mirrors y in the MuJoCo frame)


def wheel_body(name, pos, side, indent):
    pad = " " * indent
    return f"""{pad}<body name="wheel_{name}" pos="{fmt(pos)}">
{pad}  <joint name="wheel_{name}" class="wheel"/>
{pad}  <geom class="visual" mesh="rim_{side.name}" material="rim"/>
{pad}  <geom class="visual" mesh="tire_{side.name}" material="tire"/>
{pad}  <geom name="tire_{name}" class="tire" mass="{M_WHEEL}"/>
{pad}</body>
"""


def knuckle_body(name, pos, side, wheel_pos, indent, mount_box):
    pad = " " * indent
    return f"""{pad}<body name="knuckle_{name}" pos="{fmt(pos)}">
{pad}  <joint name="steer_{name}" class="steer"/>
{pad}  <geom class="visual" mesh="knuckle_{name}" material="mount"/>
{pad}  <geom class="collision" type="box" pos="{fmt(mount_box[0])}" size="{fmt(mount_box[1])}" mass="{M_STEER}"/>
{wheel_body(name, wheel_pos, side, indent + 2)}{pad}</body>
"""


def side_tree(side: Side):
    s = side.sgn
    chassis_o = (X_CENTER, ROCKER_PIVOT[1], ROCKER_PIVOT[2])
    rp = ROCKER_PIVOT if s > 0 else mirror_lr(ROCKER_PIVOT)
    rocker_pos = a2m(rp, chassis_o)

    steer_fl = (STEER_X, Y_FRONT, STEER_Z)
    steer_rl = (STEER_X, Y_REAR, STEER_Z)
    wheel_fl = (WHEEL_X, Y_FRONT, 0.0)
    wheel_ml = (WHEEL_X, Y_MID, 0.0)
    wheel_rl = (WHEEL_X, Y_REAR, 0.0)
    mir = (lambda p: p) if s > 0 else mirror_lr
    bp = mir(BOGIE_PIVOT)

    # motor mount bounding box (assembly, front-left instance): x 222.6..277.2, y +-13.5,
    # z -13.5..74.8 -> centre / half sizes relative to the steering axis
    mount_c = a2m((249.9, Y_FRONT, 30.65), steer_fl)
    mount_h = (0.0135, 0.0273, 0.0442)
    # bogie lower housing (mid motor), assembly x 222.5..277, y 166.5..200, z -13.5..13.5
    bogie_box_c = a2m(mir((249.75, 183.0, 0.0)), bp)
    bogie_box_h = (0.0167, 0.0273, 0.0135)
    # arm: box along the rocker from front to the bogie pivot (inertia / coarse collision)
    arm_c = a2m(mir((241.0, 140.0, 115.0)), rp)
    arm_h = (0.150, 0.030, 0.012)

    n = side.name
    xml = f"""    <!-- ===================== {'LEFT' if s > 0 else 'RIGHT'} rocker ===================== -->
    <body name="rocker_{n}" pos="{fmt(rocker_pos)}">
      <joint name="rocker_{n}" class="rocker"/>
      <geom class="visual" mesh="rocker_{n}" material="frame"/>
      <geom class="visual" mesh="servo_front_{n}" material="servo"/>
      <geom class="collision" type="box" pos="{fmt(arm_c)}" size="{fmt(arm_h)}" mass="{M_ROCKER}"/>
      <site name="link_anchor_{n}" pos="{fmt(a2m(mir(LINK_ATTACH), rp))}" size="0.003"/>
{knuckle_body('F' + n, a2m(mir(steer_fl), rp), side, a2m(mir(wheel_fl), mir(steer_fl)), 6, (mount_c, mount_h))}
      <!-- bogie -->
      <body name="bogie_{n}" pos="{fmt(a2m(bp, rp))}">
        <joint name="bogie_{n}" class="bogie"/>
        <geom class="visual" mesh="bogie_{n}" material="frame"/>
        <geom class="visual" mesh="servo_rear_{n}" material="servo"/>
        <geom class="visual" mesh="motor_mid_{n}" material="motor"/>
        <geom class="collision" type="box" pos="{fmt(bogie_box_c)}" size="{fmt(bogie_box_h)}" mass="{M_BOGIE}"/>
{wheel_body('M' + n, a2m(mir(wheel_ml), bp), side, 8)}
{knuckle_body('R' + n, a2m(mir(steer_rl), bp), side, a2m(mir(wheel_rl), mir(steer_rl)), 8, (mount_c, mount_h))}      </body>
    </body>
"""
    return xml


def diff_tree():
    chassis_o = (X_CENTER, ROCKER_PIVOT[1], ROCKER_PIVOT[2])
    bar_pos = a2m(DIFF_PIVOT, chassis_o)
    link_o_l = (220.0, DIFF_PIVOT[1], DIFF_PIVOT[2])
    link_l = a2m(link_o_l, DIFF_PIVOT)
    link_r = a2m(mirror_lr(link_o_l), DIFF_PIVOT)
    # far end of the link (arm hole), expressed in the link frame
    far_l = a2m((220.0, LINK_ATTACH[1], LINK_ATTACH[2]), link_o_l)
    far_r = a2m(mirror_lr((220.0, LINK_ATTACH[1], LINK_ATTACH[2])), mirror_lr(link_o_l))
    link_mid = (far_l[0] / 2, 0.0, far_l[2] / 2)
    return f"""    <!-- ===================== differential bar + links ===================== -->
    <body name="diff_bar" pos="{fmt(bar_pos)}">
      <joint name="diff_bar" type="hinge" axis="0 0 1" damping="0.01" armature="1e-5"/>
      <geom class="visual" mesh="diff_bar" material="frame"/>
      <geom class="collision" type="box" size="0.009 0.0915 0.0015" mass="{M_DIFF_BAR}" contype="0" conaffinity="0"/>
      <body name="diff_link_L" pos="{fmt(link_l)}">
        <joint name="diff_link_L" type="ball" damping="0.002" armature="1e-6"/>
        <geom class="visual" mesh="diff_link_L" material="link"/>
        <geom class="collision" type="capsule" fromto="0 0 0 {fmt(far_l)}" size="0.004" mass="{M_LINK}" contype="0" conaffinity="0"/>
        <site name="link_end_L" pos="{fmt(far_l)}" size="0.003"/>
      </body>
      <body name="diff_link_R" pos="{fmt(link_r)}">
        <joint name="diff_link_R" type="ball" damping="0.002" armature="1e-6"/>
        <geom class="visual" mesh="diff_link_R" material="link"/>
        <geom class="collision" type="capsule" fromto="0 0 0 {fmt(far_r)}" size="0.004" mass="{M_LINK}" contype="0" conaffinity="0"/>
        <site name="link_end_R" pos="{fmt(far_r)}" size="0.003"/>
      </body>
    </body>
""", far_l, far_r


def build_xml():
    chassis_o = (X_CENTER, ROCKER_PIVOT[1], ROCKER_PIVOT[2])
    # body.stl bbox: x 67.5..207.5, y 80..230, z 80..153
    chassis_c = a2m((X_CENTER, 155.0, 116.5), chassis_o)
    chassis_h = (0.075, 0.070, 0.0365)
    diff_xml, far_l, far_r = diff_tree()

    mesh_names = ["chassis", "diff_bar", "diff_link_L", "diff_link_R",
                  "rocker_L", "rocker_R", "servo_front_L", "servo_front_R",
                  "bogie_L", "bogie_R", "servo_rear_L", "servo_rear_R",
                  "motor_mid_L", "motor_mid_R",
                  "knuckle_FL", "knuckle_FR", "knuckle_RL", "knuckle_RR",
                  "rim_L", "rim_R", "tire_L", "tire_R"]
    mesh_xml = "\n".join(f'    <mesh name="{m}" file="{m}.stl"/>' for m in mesh_names)

    wheels = ["FL", "FR", "ML", "MR", "RL", "RR"]
    steers = ["FL", "FR", "RL", "RR"]
    drive_xml = "\n".join(
        f'    <velocity name="drive_{w}" joint="wheel_{w}" class="drive"/>' for w in wheels)
    steer_xml = "\n".join(
        f'    <position name="steer_{w}" joint="steer_{w}" class="servo"/>' for w in steers)

    xml = f"""<!-- GENERATED by tools/build_rover.py - edit that script, not this file. -->
<!--
  Papaya Pathfinder (tronxi/papaya-pathfinder, Apache-2.0): six-wheel rocker-bogie rover.
  Frame: x forward, y left, z up, metres, radians. The chassis origin sits on the rocker
  pivot axis (assembly y = 155, z = 104.7); the wheel axles are 0.1047 m below it and the
  tires have a 0.054 m radius, so the chassis origin rests 0.1587 m above flat ground.

  Kinematic tree (per side):
    chassis -(rocker hinge, lateral)- rocker -(steer hinge, vertical)- knuckle_F -(axle)- wheel_F
                                           \\-(bogie hinge, lateral)- bogie -(axle)- wheel_M
                                                                         \\-(steer)- knuckle_R -(axle)- wheel_R
  The differential bar is hinged on the chassis top; each end carries a link (ball joint)
  whose far end is pinned to the rocker by a connect constraint, so the two rockers rotate
  in opposite directions and the chassis pitch is their average, exactly like the real rover.
-->
<mujoco model="papaya_pathfinder">
  <compiler angle="radian" meshdir="assets" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>

  <default>
    <geom condim="3" friction="1.0 0.005 0.0001" solref="0.01 1"/>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="1" mass="0"/>
    </default>
    <default class="collision">
      <geom group="3" rgba="0.9 0.3 0.1 0.5"/>
    </default>
    <default class="tire">
      <geom type="cylinder" size="{WHEEL_R} {WHEEL_W / 2}" zaxis="0 1 0" group="3"
            rgba="0.1 0.6 0.9 0.4" friction="1.2 0.01 0.0002" priority="1"/>
    </default>
    <default class="wheel">
      <joint type="hinge" axis="0 1 0" damping="0.002" armature="0.0005"/>
    </default>
    <default class="steer">
      <joint type="hinge" axis="0 0 1" range="-1.1 1.1" damping="0.02" armature="0.0005"/>
    </default>
    <default class="rocker">
      <joint type="hinge" axis="0 1 0" range="-0.6 0.6" damping="0.3" armature="0.0005"/>
    </default>
    <default class="bogie">
      <joint type="hinge" axis="0 1 0" range="-0.8 0.8" damping="0.2" armature="0.0005"/>
    </default>
    <!-- GA25-370 20:1 gear motors driven by BTS7960 bridges: velocity servo in the sim -->
    <default class="drive">
      <velocity kv="0.08" ctrlrange="-45 45" forcerange="-0.45 0.45"/>
    </default>
    <!-- DS3218MG / MS24 steering servos -->
    <default class="servo">
      <position kp="6" kv="0.15" ctrlrange="-1.0 1.0" forcerange="-2.0 2.0"/>
    </default>
  </default>

  <asset>
    <material name="frame" rgba="0.95 0.55 0.10 1" specular="0.3" shininess="0.3"/>
    <material name="mount" rgba="0.35 0.35 0.38 1"/>
    <material name="servo" rgba="0.12 0.12 0.14 1"/>
    <material name="motor" rgba="0.75 0.75 0.78 1" specular="0.6" shininess="0.6"/>
    <material name="rim" rgba="0.9 0.9 0.9 1"/>
    <material name="tire" rgba="0.08 0.08 0.08 1"/>
    <material name="link" rgba="0.2 0.2 0.2 1"/>
{mesh_xml}
  </asset>

  <worldbody>
    <body name="chassis" pos="0 0 0.17">
      <freejoint name="chassis"/>
      <camera name="chase" mode="trackcom" pos="-1.1 0 0.55" xyaxes="0 -1 0 0.45 0 1"/>
      <camera name="onboard" pos="0.09 0 0.08" xyaxes="0 -1 0 0.25 0 1"/>
      <site name="imu" pos="0 0 0" size="0.005"/>
      <geom class="visual" mesh="chassis" material="frame"/>
      <geom name="chassis_box" class="collision" type="box" pos="{fmt(chassis_c)}" size="{fmt(chassis_h)}" mass="{M_CHASSIS}"/>

{side_tree(Side('L', +1.0))}
{side_tree(Side('R', -1.0))}
{diff_xml}    </body>
  </worldbody>

  <equality>
    <!-- pin the far end of each differential link to the hole on top of the rocker -->
    <connect name="diff_link_L" body1="diff_link_L" body2="rocker_L" anchor="{fmt(far_l)}" solref="0.004 1"/>
    <connect name="diff_link_R" body1="diff_link_R" body2="rocker_R" anchor="{fmt(far_r)}" solref="0.004 1"/>
  </equality>

  <contact>
    <!-- parts that are bolted together or always overlap must not collide -->
    <exclude body1="chassis" body2="rocker_L"/>
    <exclude body1="chassis" body2="rocker_R"/>
    <exclude body1="rocker_L" body2="bogie_L"/>
    <exclude body1="rocker_R" body2="bogie_R"/>
    <exclude body1="rocker_L" body2="knuckle_FL"/>
    <exclude body1="rocker_R" body2="knuckle_FR"/>
    <exclude body1="bogie_L" body2="knuckle_RL"/>
    <exclude body1="bogie_R" body2="knuckle_RR"/>
  </contact>

  <actuator>
{drive_xml}
{steer_xml}
  </actuator>

  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framequat name="orientation" objtype="site" objname="imu"/>
    <velocimeter name="velocity" site="imu"/>
  </sensor>
</mujoco>
"""
    with open(XML_OUT, "w") as f:
        f.write(xml)
    print(f"wrote {os.path.relpath(XML_OUT, ROOT)}")


if __name__ == "__main__":
    build_meshes()
    build_xml()
