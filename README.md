# Papaya Pathfinder rover in MuJoCo

Keyboard-driven MuJoCo simulation of the [Papaya Pathfinder](https://github.com/tronxi/papaya-pathfinder),
tronxi's open-source, fully 3D-printable six-wheel rocker-bogie rover (ESP32, 6x GA25 gear
motors, 4x steering servos). The MJCF is built from the real printed-part STL files plus the
servo and motor solids tessellated from the project's STEP assembly, with every joint placed
on the pivot holes measured in those meshes.

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 drive.py                     # mixed terrain
python3 drive.py --terrain rocks     # flat | hills | rocks | slope | obstacles | mixed
```

## Controls

| Key | Action |
| --- | --- |
| `W` / `S` (or Up / Down) | throttle forward / reverse (hold) |
| `A` / `D` (or Left / Right) | steer left / right, Ackermann about the mid axle (hold) |
| `Q` / `E` | spin in place (point turn, corner wheels tangent to a circle) |
| `Shift` | turbo: full speed instead of 55 % cruise |
| `Space` | brake, zero every command |
| `1` .. `6` | switch terrain: flat, hills, rocks, slope, obstacles, mixed |
| `R` | reset the rover at the spawn point |
| `C` | cycle camera: track / chase / onboard / free |
| `G` | show the collision primitives |
| `P` / `H` / `Esc` | pause / help overlay / quit |

Mouse: left-drag rotates, right-drag pans, scroll zooms (track and free cameras).

## Terrains

All terrains are 512x512 heightfields generated procedurally in `terrains.py` and uploaded
into the single `model/scene.xml` at runtime, so switching is instant.

- **flat** – plain ground.
- **hills** – rolling hills, about 0.4 m of relief.
- **rocks** – rock garden: gaussian bumps 2–8 cm tall plus pebble roughness.
- **slope** – 22° ramp (+x), 30° ramp (-x), 15° side slopes (±y).
- **obstacles** – staircase with 3/5/7/9 cm risers (+x), half-round logs of 4/6/8 cm radius
  (-x), an 11 cm block (+y), a washboard (-y).
- **mixed** – every type in a different direction from the spawn: hills (+x), rocks (+y),
  ramp (-x), steps (-y).

## Layout

```
drive.py              GLFW window + MuJoCo renderer, hold-to-drive keyboard loop, HUD
control.py            throttle/steer/spin -> wheel speeds + servo angles (true Ackermann)
terrains.py           heightfield generators
model/scene.xml       lights, heightfield, includes rover.xml
model/rover.xml       GENERATED rover MJCF (do not edit by hand)
model/assets/*.stl    GENERATED body-local meshes in metres
tools/build_rover.py  mesh conversion + MJCF generator; all pivots and masses live here
third_party/papaya_pathfinder/   upstream STLs (Apache-2.0) + servo/motor from the STEP file
tests/test_rover.py   headless checks: stance, straight line, turn radius, spin, climbing
```

Rebuild the model after changing anything in `tools/build_rover.py`:

```
python3 tools/build_rover.py
MUJOCO_GL=egl python3 -m pytest tests/
```

## Model notes

- Frame: x forward, y left, z up. Chassis origin on the rocker pivot axis, 0.1587 m above
  flat ground. Wheelbase 0.34 m (front axle +0.155, mid -0.025, rear -0.185), track 0.319 m,
  tires 108 mm diameter, 40 mm wide.
- Kinematics per side: `chassis -rocker hinge- rocker -steer- knuckle_F -axle- wheel_F`, and
  `rocker -bogie hinge- bogie` carrying `wheel_M` and `knuckle_R / wheel_R`.
- The differential bar is hinged on the chassis top. Each end carries a link on a ball joint
  whose far end is pinned to the top of the rocker by a `connect` constraint, so the rockers
  rotate in opposite directions and the chassis pitch is their average, like the real rover.
- Visual geometry is the real meshes (group 1, no collision). Collision and inertia come from
  simple primitives (group 3, press `G`): cylinders for the tires, boxes for chassis, arms,
  motor mounts and the bogie motor housing. Total mass 3.4 kg.
- Actuators: six velocity servos (GA25 gear motors, ±0.45 N·m, up to 45 rad/s) and four
  position servos (DS3218MG/MS24, ±2 N·m). The upstream firmware writes the same offset to
  every steering servo; the sim instead solves the Ackermann geometry about the mid axle
  with per-wheel speeds, so the wheels do not scrub. Max servo angle is 35° like the
  firmware's `SERVO_MAX_DELTA`.
- Mesh frames: the upstream STLs are exported in the CAD assembly frame (mm). The build
  script measured the pivots from the holes in those meshes: rocker pivot y=155 z=104.7,
  bogie pivot y=275 z=105, steering axes x=258.7 at y=0 / 340, wheel axles z=0, differential
  bar hinge (137.5, 195, 154.5), link holes 40 mm apart. See `third_party/.../README.md`.
