"""Procedural heightfield terrains for the rover scene.

Every terrain is a square grid of NROW x NCOL normalised heights in [0, 1] plus the
MuJoCo hfield size (half extent x, half extent y, elevation scale, base thickness).
The area around the origin is always flattened to height 0 so the rover can spawn there.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

NROW = NCOL = 512          # must match <hfield nrow ncol> in model/scene.xml
MAX_HALF_EXTENT = 10.0     # must match <hfield size> x/y in model/scene.xml
ELEVATION = 3.0            # must match <hfield size> elevation in model/scene.xml
SPAWN_RADIUS = 0.7         # m of flat ground kept around the origin


@dataclass
class Terrain:
    name: str
    heights: np.ndarray        # (NROW, NCOL) in metres, >= 0
    half_extent: float         # m, terrain spans [-half_extent, half_extent] in x and y
    description: str

    def hfield_data(self) -> np.ndarray:
        """Row-major normalised heights the way MuJoCo stores them (row = y, col = x)."""
        return (self.heights / ELEVATION).astype(np.float32).ravel()

    def hfield_size(self) -> np.ndarray:
        return np.array([self.half_extent, self.half_extent, ELEVATION, 0.5])

    def height_at(self, x: float, y: float) -> float:
        c = int(np.clip((x / self.half_extent + 1) * 0.5 * (NCOL - 1), 0, NCOL - 1))
        r = int(np.clip((y / self.half_extent + 1) * 0.5 * (NROW - 1), 0, NROW - 1))
        return float(self.heights[r, c])


def _grid(half_extent: float):
    xs = np.linspace(-half_extent, half_extent, NCOL)
    ys = np.linspace(-half_extent, half_extent, NROW)
    return np.meshgrid(xs, ys)


def _flatten_spawn(h: np.ndarray, x: np.ndarray, y: np.ndarray, radius=SPAWN_RADIUS, blend=0.6):
    """Blend the heights to zero inside `radius` of the origin with a smooth ring."""
    r = np.hypot(x, y)
    w = np.clip((r - radius) / blend, 0.0, 1.0)
    w = w * w * (3 - 2 * w)
    return h * w


def _smoothstep(e0: float, e1: float, v: np.ndarray) -> np.ndarray:
    w = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
    return w * w * (3 - 2 * w)


def _smooth_noise(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, octaves, wavelength,
                  amplitude):
    """Cheap band-limited noise: a sum of randomly oriented sinusoids per octave."""
    h = np.zeros_like(x)
    for o in range(octaves):
        lam = wavelength / (2 ** o)
        amp = amplitude / (2 ** o)
        for _ in range(4):
            theta = rng.uniform(0, 2 * np.pi)
            phase = rng.uniform(0, 2 * np.pi)
            k = 2 * np.pi / lam
            h += amp / 4 * np.sin(k * (x * np.cos(theta) + y * np.sin(theta)) + phase)
    return h


def flat(half_extent=10.0, **_) -> Terrain:
    h = np.zeros((NROW, NCOL))
    return Terrain("flat", h, half_extent, "flat ground")


def hills(half_extent=8.0, seed=0, **_) -> Terrain:
    rng = np.random.default_rng(seed)
    x, y = _grid(half_extent)
    h = _smooth_noise(x, y, rng, octaves=3, wavelength=4.0, amplitude=0.16)
    h = _flatten_spawn(h, x, y)
    h -= h.min()
    return Terrain("hills", h, half_extent, "rolling hills, ~0.4 m relief")


def rocks(half_extent=6.0, seed=1, **_) -> Terrain:
    rng = np.random.default_rng(seed)
    x, y = _grid(half_extent)
    h = _smooth_noise(x, y, rng, octaves=2, wavelength=3.0, amplitude=0.12)
    # scattered rocks: gaussian bumps 4-14 cm wide, 2-8 cm tall
    for _ in range(220):
        cx, cy = rng.uniform(-half_extent, half_extent, 2)
        if np.hypot(cx, cy) < SPAWN_RADIUS:
            continue
        sx, sy = rng.uniform(0.04, 0.14, 2)
        amp = rng.uniform(0.02, 0.08)
        h += amp * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    # a few pebble fields: short-wavelength roughness
    h += 0.012 * _smooth_noise(x, y, rng, octaves=2, wavelength=0.25, amplitude=1.0)
    h = _flatten_spawn(h, x, y)
    h -= h.min()
    return Terrain("rocks", h, half_extent, "rock garden: bumps up to ~8 cm, pebbles")


def slope(half_extent=6.0, angle_deg=22.0, **_) -> Terrain:
    x, y = _grid(half_extent)
    k = np.tan(np.radians(angle_deg))
    # +x: ramp up to a plateau; -x: a steeper (angle + 8 deg) ramp; +-y: side slopes
    up = np.clip(x - 1.0, 0.0, 3.0) * k
    down = np.clip(-x - 1.0, 0.0, 2.5) * np.tan(np.radians(angle_deg + 8))
    side = np.clip(np.abs(y) - 1.2, 0.0, 3.0) * np.tan(np.radians(15))
    h = up + down + side
    h = _flatten_spawn(h, x, y, radius=0.9)
    return Terrain("slope", h, half_extent,
                   f"ramps: {angle_deg:.0f} deg (+x), {angle_deg + 8:.0f} deg (-x), 15 deg side")


def obstacles(half_extent=5.0, **_) -> Terrain:
    x, y = _grid(half_extent)
    h = np.zeros_like(x)
    # staircase in +x: risers of 3, 5, 7, 9 cm
    rise = 0.0
    for i, step in enumerate((0.03, 0.05, 0.07, 0.09)):
        rise += step
        h = np.where((x > 1.2 + 0.45 * i) & (np.abs(y) < 1.6), rise, h)
    h = np.where((x > 1.2 + 0.45 * 4) & (np.abs(y) < 1.6), rise, h)
    descent = rise * (1.0 - np.clip((x - 3.6) / 1.2, 0.0, 1.0))   # ramp back down behind
    h = np.where((x > 3.6) & (np.abs(y) < 1.6), descent, h)
    # logs / ridges in -x: half-cylinders 4, 6, 8 cm radius
    for i, r in enumerate((0.04, 0.06, 0.08)):
        cx = -1.3 - 0.7 * i
        d = np.abs(x - cx)
        ridge = np.sqrt(np.clip(r * r - d * d, 0.0, None))
        h = np.where((d < r) & (np.abs(y) < 1.6), np.maximum(h, ridge), h)
    # a single big boulder-ish block and a trench in +y
    h = np.where((np.abs(x) < 0.35) & (np.abs(y - 2.2) < 0.35), 0.11, h)
    h = np.where((np.abs(x) < 1.8) & (np.abs(y - 3.5) < 0.12), 0.0, h)
    # washboard in -y: 3 cm sinusoidal bumps every 25 cm
    wash = 0.015 * (1 + np.sin(2 * np.pi * y / 0.25))
    h = np.where((y < -1.3) & (y > -4.0) & (np.abs(x) < 1.6), wash, h)
    return Terrain("obstacles", h, half_extent,
                   "stairs (+x), logs (-x), block (+y), washboard (-y)")


def mixed(half_extent=7.0, seed=2, **_) -> Terrain:
    """One world with every terrain type in a different direction, blended without cliffs."""
    rng = np.random.default_rng(seed)
    x, y = _grid(half_extent)
    h = np.zeros_like(x)
    # +x: hills fading in between x = 0.8 and x = 2.5
    hill = _smooth_noise(x, y, rng, octaves=3, wavelength=3.5, amplitude=0.14)
    h += (hill - hill.min()) * _smoothstep(0.8, 2.5, x)
    # +y: rock garden
    for _ in range(200):
        cx, cy = rng.uniform(-half_extent, half_extent), rng.uniform(1.0, half_extent)
        sx, sy = rng.uniform(0.04, 0.14, 2)
        h += rng.uniform(0.02, 0.08) * np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))
    # -x: 20 deg ramp with sloped shoulders instead of cliffs
    ramp = np.clip(-x - 1.0, 0.0, 3.0) * np.tan(np.radians(20))
    h += ramp * (1.0 - _smoothstep(2.0, 3.0, np.abs(y)))
    # -y: staircase with 3 / 5 / 7 cm risers, tapered at the sides
    steps = np.zeros_like(x)
    rise = 0.0
    for i, step in enumerate((0.03, 0.05, 0.07)):
        rise += step
        steps = np.where(-y > 1.2 + 0.5 * i, rise, steps)
    h += steps * (1.0 - _smoothstep(1.8, 2.6, np.abs(x)))
    h = _flatten_spawn(h, x, y, radius=0.9, blend=0.5)
    h -= h.min()
    return Terrain("mixed", h, half_extent,
                   "hills (+x), rocks (+y), ramp (-x), steps (-y)")


TERRAINS = {
    "flat": flat,
    "hills": hills,
    "rocks": rocks,
    "slope": slope,
    "obstacles": obstacles,
    "mixed": mixed,
}
TERRAIN_ORDER = list(TERRAINS)


def make(name: str, **kwargs) -> Terrain:
    if name not in TERRAINS:
        raise KeyError(f"unknown terrain {name!r}; choose from {TERRAIN_ORDER}")
    return TERRAINS[name](**kwargs)


def apply(model, terrain: Terrain, hfield_name="terrain") -> None:
    """Write the terrain into an already compiled MjModel (call mjr_uploadHField after).

    Only the data and the x/y half extents change; the elevation scale and the bounding box
    MuJoCo derived at compile time stay valid as long as the terrain fits the XML size.
    """
    if terrain.half_extent > MAX_HALF_EXTENT or terrain.heights.max() > ELEVATION:
        raise ValueError("terrain exceeds the hfield size declared in model/scene.xml")
    hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
    nrow, ncol = model.hfield_nrow[hid], model.hfield_ncol[hid]
    if (nrow, ncol) != terrain.heights.shape:
        raise ValueError(f"scene hfield is {nrow}x{ncol}, terrain is {terrain.heights.shape}")
    adr = model.hfield_adr[hid]
    model.hfield_data[adr:adr + nrow * ncol] = terrain.hfield_data()
    model.hfield_size[hid] = terrain.hfield_size()
