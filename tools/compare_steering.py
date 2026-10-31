"""Servo steering (Ackermann + point turn) vs skid steering on the Papaya Pathfinder.

Builds variants of the same MJCF (same mass, same GA25 motors, same tires):
  servo        the real rover: six driven wheels + four corner steering servos (control.py)
  skid         steering joints welded straight; turn by left/right wheel speed difference at
               the kinematic speeds (v -/+ omega*W/2)
  skid_full    same, but the point turn spins the wheels at full speed (how people really
               skid-steer: overdrive and let the tires slide)
  skid_locked  skid with the rocker and bogie pivots welded too (rigid 6-wheel chassis)

and drives them through the same manoeuvres (same demanded chassis speed v and yaw rate
omega) on the same terrains, logging:
  yaw / radius actually achieved vs demanded     open-loop predictability
  drive-motor mechanical work                    battery use, sum |tau*w| dt
  peak and saturated drive torque                stall margin of the GA25 motors
  wheels carrying load (time-averaged normal force)   rocker-bogie wheel lift
  tire scrub                                     lateral slip distance / friction energy

    MUJOCO_GL=egl python3 tools/compare_steering.py --out DIR [--plot] [--quick]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import terrains  # noqa: E402
from control import (Command, RoverController, WHEELS, STEERED,  # noqa: E402
                     MAX_WHEEL_SPEED, MAX_SPIN_RATE)

MODEL_DIR = os.path.join(ROOT, "model")
SETTLE = 1.0          # s of rest before every manoeuvre
DRIVE_STALL = 0.45    # N m, forcerange of the drive class in rover.xml
CONFIGS = ("servo", "skid", "skid_full", "skid_locked")


# ----------------------------------------------------------------------------- models
def model_xml(skid: bool, locked: bool = False) -> str:
    """scene.xml + rover.xml flattened into one string, with joints removed as requested."""
    rover = open(os.path.join(MODEL_DIR, "rover.xml")).read()
    rover = rover.replace('meshdir="assets"', f'meshdir="{os.path.join(MODEL_DIR, "assets")}"')
    if skid:
        rover = re.sub(r'\n[ \t]*<joint name="steer_[A-Z]+" class="steer"/>', "", rover)
        rover = re.sub(r'\n[ \t]*<position name="steer_[A-Z]+" joint="steer_[A-Z]+" class="servo"/>',
                       "", rover)
        assert "steer_" not in rover.replace("servo_", ""), "steering not fully removed"
    if locked:
        rover = re.sub(r'\n[ \t]*<joint name="(rocker|bogie)_[LR]" class="(rocker|bogie)"/>', "", rover)
    scene = open(os.path.join(MODEL_DIR, "scene.xml")).read()
    body = rover.split('<mujoco model="papaya_pathfinder">', 1)[1].rsplit("</mujoco>", 1)[0]
    return scene.replace('<include file="rover.xml"/>', body)


_MODEL_CACHE: dict[tuple, mujoco.MjModel] = {}


def load(cfg: str, terrain: terrains.Terrain, tire_mu: float, stall: float) -> mujoco.MjModel:
    key = (cfg != "servo", cfg == "skid_locked")
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = mujoco.MjModel.from_xml_string(model_xml(*key))
    model = copy.deepcopy(_MODEL_CACHE[key])
    terrains.apply(model, terrain)
    for w in WHEELS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tire_{w}")
        model.geom_friction[gid, 0] = tire_mu        # priority=1 so this is the contact mu
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{w}")
        model.actuator_forcerange[aid] = (-stall, stall)
    return model


def spawn_pose(terrain: terrains.Terrain, x: float, y: float):
    """Position + quaternion that lays the rover on the local terrain slope."""
    e = 0.15
    gx = (terrain.height_at(x + e, y) - terrain.height_at(x - e, y)) / (2 * e)
    gy = (terrain.height_at(x, y + e) - terrain.height_at(x, y - e)) / (2 * e)
    n = np.array([-gx, -gy, 1.0])
    n /= np.linalg.norm(n)
    axis = np.cross([0, 0, 1.0], n)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        quat = np.array([1.0, 0, 0, 0])
    else:
        ang = np.arctan2(s, n[2])
        axis /= s
        quat = np.concatenate([[np.cos(ang / 2)], np.sin(ang / 2) * axis])
    pos = np.array([x, y, terrain.height_at(x, y)]) + n * 0.19
    return pos, quat


# ------------------------------------------------------------------------ controllers
class ServoDrive:
    """Wraps control.RoverController: (v, omega) -> Ackermann / point-turn command."""

    def __init__(self, model):
        self.c = RoverController(model)
        self.r = self.c.wheel_radius

    def apply(self, data, v: float, omega: float):
        if abs(v) < 1e-6:
            cmd = Command(spin=np.clip(omega / MAX_SPIN_RATE, -1, 1))
        else:
            cmd = Command(throttle=v / (MAX_WHEEL_SPEED * self.r),
                          steer=np.clip((omega / v) / self.c.max_curvature, -1, 1))
        self.c.apply(data, cmd)


class SkidDrive:
    """Differential drive: left wheels v - omega*W/2, right wheels v + omega*W/2 (times gain)."""

    def __init__(self, model, full_speed_spin=False):
        self.r = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "tire_FL")][0])
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        ys = [(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"wheel_{w}")]
               - data.xpos[chassis])[1] for w in WHEELS]
        self.half_track = float(np.mean(np.abs(ys)))
        self.ids = {w: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{w}")
                    for w in WHEELS}
        self.full = full_speed_spin

    def apply(self, data, v: float, omega: float):
        d = omega * self.half_track / self.r
        if self.full and abs(v) < 1e-6 and abs(omega) > 1e-6:
            d = np.sign(omega) * MAX_WHEEL_SPEED
        left, right = v / self.r - d, v / self.r + d
        for w, aid in self.ids.items():
            data.ctrl[aid] = np.clip(left if w.endswith("L") else right,
                                     -MAX_WHEEL_SPEED, MAX_WHEEL_SPEED)


# ------------------------------------------------------------------------- simulation
def yaw_of(data, bid):
    q = data.xquat[bid]
    return np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def run(cfg: str, terrain, spawn, v, omega, seconds, tire_mu=1.2, stall=DRIVE_STALL,
        record_path=False):
    model = load(cfg, terrain, tire_mu, stall)
    data = mujoco.MjData(model)
    pos, quat = spawn_pose(terrain, *spawn)
    data.qpos[0:3], data.qpos[3:7] = pos, quat
    ctrl = ServoDrive(model) if cfg == "servo" else SkidDrive(model, cfg == "skid_full")
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    terrain_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
    tire_gids = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tire_{w}"): w for w in WHEELS}
    drive_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"drive_{w}") for w in WHEELS]
    servo_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"steer_{w}")
                 for w in STEERED] if cfg == "servo" else []
    dt = model.opt.timestep

    for _ in range(int(SETTLE / dt)):                       # settle at rest
        ctrl.apply(data, 0.0, 0.0)
        mujoco.mj_step(model, data)
    x0, y0 = data.xpos[chassis][:2]

    yaw_prev = yaw_of(data, chassis)
    p_prev = data.xpos[chassis][:2].copy()
    m = dict(yaw=0.0, dist=0.0, work_drive=0.0, work_servo=0.0, tau_sq=0.0, peak_tau=0.0,
             sat_time=0.0, scrub_m=0.0, scrub_J=0.0, min_up=1.0)
    load_N = {w: 0.0 for w in WHEELS}
    path = []
    vel6, f6 = np.zeros(6), np.zeros(6)
    n_steps = int(seconds / dt)
    for k in range(n_steps):
        ctrl.apply(data, v, omega)
        mujoco.mj_step(model, data)
        yaw = yaw_of(data, chassis)
        m["yaw"] += (yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi
        yaw_prev = yaw
        p = data.xpos[chassis][:2].copy()
        m["dist"] += float(np.linalg.norm(p - p_prev))
        p_prev = p
        if record_path and k % 25 == 0:
            path.append((float(p[0]), float(p[1])))
        m["min_up"] = min(m["min_up"], float(data.xmat[chassis].reshape(3, 3)[2, 2]))
        tau = data.actuator_force[drive_ids]
        m["work_drive"] += float(np.sum(np.abs(tau * data.actuator_velocity[drive_ids]))) * dt
        m["tau_sq"] += float(np.sum(tau ** 2)) * dt
        m["peak_tau"] = max(m["peak_tau"], float(np.max(np.abs(tau))))
        if np.any(np.abs(tau) > 0.95 * stall):
            m["sat_time"] += dt
        if servo_ids:
            ts = data.actuator_force[servo_ids]
            m["work_servo"] += float(np.sum(np.abs(ts * data.actuator_velocity[servo_ids]))) * dt
        # tire contacts: normal load per wheel and tangential slip of the contact point
        per_wheel: dict[str, list] = {}
        for i in range(data.ncon):
            c = data.contact[i]
            g = c.geom1 if c.geom2 == terrain_gid else (c.geom2 if c.geom1 == terrain_gid else -1)
            if g not in tire_gids:
                continue
            bid = model.geom_bodyid[g]
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, vel6, 0)
            vp = vel6[3:] + np.cross(vel6[:3], c.pos - data.xpos[bid])
            n = c.frame[:3]
            vt = vp - n * np.dot(n, vp)
            mujoco.mj_contactForce(model, data, i, f6)
            load_N[tire_gids[g]] += f6[0] * dt
            per_wheel.setdefault(tire_gids[g], []).append((float(np.linalg.norm(vt)),
                                                           float(np.hypot(f6[1], f6[2]))))
        for lst in per_wheel.values():
            m["scrub_m"] += float(np.mean([s for s, _ in lst])) * dt
            m["scrub_J"] += sum(ft * s for s, ft in lst) * dt
    m["sat_frac"] = m["sat_time"] / seconds
    m["peak_frac"] = m["peak_tau"] / stall
    m["yaw_deg"] = float(np.degrees(m["yaw"]))
    m["yaw_rate"] = float(m["yaw"] / seconds)
    m["radius"] = m["dist"] / abs(m["yaw"]) if abs(m["yaw"]) > 1e-3 else float("inf")
    m["x_gain"] = float(data.xpos[chassis][0] - x0)
    m["rolled"] = m["min_up"] < 0.5
    m["load_N"] = {w: load_N[w] / seconds for w in WHEELS}           # time-averaged N
    share = sum(m["load_N"].values()) / 6
    m["wheels_loaded"] = int(sum(1 for w in WHEELS if m["load_N"][w] > 0.15 * share))
    m["J_per_rev"] = (m["work_drive"] + m["work_servo"]) / (abs(m["yaw"]) / (2 * np.pi)) \
        if abs(m["yaw"]) > 0.05 else float("inf")
    if record_path:
        m["path"] = path
    return m


# ---------------------------------------------------------------------------- cases
W = 2.5                      # rad/s point-turn demand (MAX_SPIN_RATE)
V = 0.6                      # m/s cruise demand (~35 % throttle)


def cases():
    flat, rocks, slope, obst, hills = (terrains.make(n) for n in
                                       ("flat", "rocks", "slope", "obstacles", "hills"))
    yield "point turn, flat, mu 1.2",        dict(terrain=flat, spawn=(0, 0), v=0, omega=W, seconds=4), CONFIGS
    yield "point turn, flat, mu 0.8",        dict(terrain=flat, spawn=(0, 0), v=0, omega=W, seconds=4, tire_mu=0.8), CONFIGS
    yield "point turn, flat, mu 0.5",        dict(terrain=flat, spawn=(0, 0), v=0, omega=W, seconds=4, tire_mu=0.5), CONFIGS
    yield "point turn, flat, mu 1.2, motors at rated 0.25 Nm", \
        dict(terrain=flat, spawn=(0, 0), v=0, omega=W, seconds=4, stall=0.25), ("servo", "skid", "skid_full")
    yield "point turn, rock garden",         dict(terrain=rocks, spawn=(0, 1.5), v=0, omega=W, seconds=4), ("servo", "skid", "skid_full")
    yield "point turn, hills",               dict(terrain=hills, spawn=(1.5, 0), v=0, omega=W, seconds=4), ("servo", "skid", "skid_full")
    yield "point turn, 15 deg side slope",   dict(terrain=slope, spawn=(0, 2.6), v=0, omega=W, seconds=4), ("servo", "skid", "skid_full")
    yield "point turn, 22 deg ramp",         dict(terrain=slope, spawn=(2.5, 0), v=0, omega=W, seconds=4), ("servo", "skid", "skid_full")
    yield "arc R=0.42 m (min servo radius), flat", dict(terrain=flat, spawn=(0, 0), v=V, omega=V / 0.42, seconds=5), ("servo", "skid", "skid_locked")
    yield "arc R=0.42 m, flat, mu 0.5",      dict(terrain=flat, spawn=(0, 0), v=V, omega=V / 0.42, seconds=5, tire_mu=0.5), ("servo", "skid")
    yield "arc R=1.0 m, flat",               dict(terrain=flat, spawn=(0, 0), v=V, omega=V / 1.0, seconds=5), ("servo", "skid", "skid_locked")
    yield "arc R=1.0 m, rock garden",        dict(terrain=rocks, spawn=(0, 1.5), v=V, omega=V / 1.0, seconds=5), ("servo", "skid")
    yield "arc R=1.0 m, hills",              dict(terrain=hills, spawn=(1.5, 0), v=V, omega=V / 1.0, seconds=5), ("servo", "skid")
    yield "straight, flat",                  dict(terrain=flat, spawn=(0, 0), v=V, omega=0, seconds=5), ("servo", "skid", "skid_locked")
    yield "straight, steps (obstacles)",     dict(terrain=obst, spawn=(0, 0), v=0.86, omega=0, seconds=4), ("servo", "skid", "skid_locked")
    yield "straight, rock garden",           dict(terrain=rocks, spawn=(0, 0), v=0.86, omega=0, seconds=4), ("servo", "skid", "skid_locked")


def mu_sweep(quick=False):
    """Point turn on flat ground vs tire friction, every config."""
    flat = terrains.make("flat")
    mus = (0.4, 0.8, 1.2) if quick else (0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2)
    out = []
    for mu in mus:
        for cfg in CONFIGS:
            m = run(cfg, flat, (0, 0), 0.0, W, 4.0, tire_mu=mu)
            m.update(case="mu sweep", cfg=cfg, tire_mu=mu)
            out.append(m)
            print(f"  mu {mu:3.1f} {cfg:12s} yaw rate {m['yaw_rate']:5.2f} rad/s  "
                  f"{m['J_per_rev']:6.1f} J/rev  wheels loaded {m['wheels_loaded']}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.getcwd(), help="directory for results.json / csv / png")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    rows = []
    hdr = (f"{'case':50s} {'cfg':12s} {'yaw°':>7s} {'R m':>6s} {'J':>6s} {'J/rev':>6s} {'pk%':>4s} "
           f"{'sat%':>4s} {'whl':>3s} {'scrub m':>8s} {'dx m':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for name, kw, cfgs in cases():
        for cfg in cfgs:
            m = run(cfg, record_path=True, **kw)
            m.update(case=name, cfg=cfg, v=kw["v"], omega=kw["omega"], seconds=kw["seconds"],
                     tire_mu=kw.get("tire_mu", 1.2), stall=kw.get("stall", DRIVE_STALL),
                     demanded_yaw_deg=float(np.degrees(kw["omega"] * kw["seconds"])),
                     demanded_radius=(kw["v"] / kw["omega"] if kw["omega"] and kw["v"] else 0.0))
            rows.append(m)
            r = f"{m['radius']:6.2f}" if np.isfinite(m["radius"]) and m["radius"] < 100 else "     -"
            jr = f"{m['J_per_rev']:6.1f}" if np.isfinite(m["J_per_rev"]) and m["omega"] else "     -"
            print(f"{name:50s} {cfg:12s} {m['yaw_deg']:7.1f} {r} {m['work_drive'] + m['work_servo']:6.1f} "
                  f"{jr} {100*m['peak_frac']:4.0f} {100*m['sat_frac']:4.0f} {m['wheels_loaded']:3d} "
                  f"{m['scrub_m']:8.3f} {m['x_gain']:6.2f}" + ("  ROLLED" if m["rolled"] else ""))
    print("\nfriction sweep, point turn on flat ground, 4 s:")
    sweep = mu_sweep(args.quick)

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(rows + sweep, f, indent=1)
    keys = ["case", "cfg", "tire_mu", "stall", "v", "omega", "seconds", "demanded_yaw_deg", "yaw_deg",
            "yaw_rate", "demanded_radius", "radius", "dist", "work_drive", "work_servo", "J_per_rev",
            "tau_sq", "peak_frac", "sat_frac", "wheels_loaded", "scrub_m", "scrub_J", "x_gain", "rolled"]
    with open(os.path.join(args.out, "results.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for m in rows + sweep:
            f.write(",".join(str(m.get(k, "")) for k in keys) + "\n")
    print(f"\nwrote {args.out}/results.json and results.csv")
    if args.plot:
        plot(rows, sweep, os.path.join(args.out, "steer_vs_skid.png"))
        print(f"wrote {args.out}/steer_vs_skid.png")


def plot(rows, sweep, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"servo": "#1f77b4", "skid": "#d62728", "skid_full": "#ff7f0e", "skid_locked": "#7f7f7f"}
    picks = ["point turn, flat, mu 1.2", "arc R=0.42 m (min servo radius), flat",
             "arc R=1.0 m, rock garden"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, case in zip(axes[0], picks):
        sel = [m for m in rows if m["case"] == case]
        for m in sel:
            p = np.array(m["path"])
            ax.plot(p[:, 0], p[:, 1], color=colors[m["cfg"]], lw=2,
                    label=f"{m['cfg']} ({m['yaw_deg']:.0f}°)")
            ax.plot(p[0, 0], p[0, 1], "ko", ms=4)
        if "arc" in case:
            R, p0 = sel[0]["demanded_radius"], np.array(sel[0]["path"][0])
            t = np.linspace(0, 2 * np.pi, 200)
            ax.plot(p0[0] + R * np.sin(t), p0[1] + R - R * np.cos(t), "k:", lw=1, label="demanded")
        ax.set_title(f"{case}: chassis path, {sel[0]['seconds']} s", fontsize=10)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("x m")
        ax.set_ylabel("y m")

    mus = sorted({m["tire_mu"] for m in sweep})
    for ax, key, title in zip(axes[1][:2], ("yaw_rate", "J_per_rev"),
                              ("point turn on flat: yaw rate achieved (demand 2.5 rad/s)",
                               "point turn on flat: motor energy per revolution")):
        for cfg in CONFIGS:
            ys = [[m[key] for m in sweep if m["cfg"] == cfg and m["tire_mu"] == mu][0] for mu in mus]
            ax.plot(mus, ys, "o-", color=colors[cfg], label=cfg)
        ax.set_xlabel("tire-ground friction coefficient")
        ax.set_ylabel("rad/s" if key == "yaw_rate" else "J per 360°")
        if key == "J_per_rev":
            ax.set_yscale("log")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    ax = axes[1][2]
    sel = {m["cfg"]: m for m in rows if m["case"] == "point turn, flat, mu 1.2"}
    xs = np.arange(6)
    for i, cfg in enumerate(("servo", "skid", "skid_locked")):
        ax.bar(xs + (i - 1) * 0.27, [sel[cfg]["load_N"][w] for w in WHEELS], 0.27,
               color=colors[cfg], label=cfg)
    ax.set_xticks(xs)
    ax.set_xticklabels(WHEELS)
    ax.set_ylabel("mean normal load N")
    ax.set_title("point turn, flat, mu 1.2: load per wheel (rocker-bogie lifts 3 wheels)", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.suptitle("Papaya Pathfinder in MuJoCo: servo steering vs skid steering (same mass, motors, tires)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
