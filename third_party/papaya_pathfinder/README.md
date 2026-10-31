# Papaya Pathfinder source meshes

Printed-part STLs copied verbatim from
https://github.com/tronxi/papaya-pathfinder (`pathfinder/3d-models/`), Apache-2.0.
They are exported in the CAD assembly frame (millimetres): x = lateral
(x = 137.5 is the rover centreline, larger x is the rover's left), y = longitudinal
(y = 0 front steering axis, y = 340 rear), z = up (wheel axles at z = 0).

`servo_ds3218mg.stl` and `motor_ga25.stl` were tessellated from
`pathfinder/cad/papaya-pathfinder.step` (the steering servo assembly placed at the
front-right corner, and the GA25 gear motor placed at the front-right wheel).

`tools/build_rover.py` turns these into body-local, metre-scaled meshes under
`model/assets/` and writes `model/rover.xml`.
