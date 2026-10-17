"""Locomotion facts a game runtime needs so feet stay planted, and the runtime's motion settings.

STRIDE = how far a character travels in one walk/run CYCLE of its clip (the clip loops once per
cycle = two steps for people, four footfalls for animals). A game plays the clip at

    cycles per second = movement speed / stride        (frame = floor(phase * frames))

so the planted foot moves backward on screen exactly as fast as the ground goes by: no
skating, no moonwalking, at any speed. Stride is measured from the SKELETON (rig_stride: the
ankle/paw joints while they're planted move back at the body's speed; in-place clips have the
drift removed first), converted to sprite pixels with the sheet's camera scale (drawn sheets
know it exactly: keys.json camera.scale) or a scale measured from the idle frame (3D renders),
and as a fallback from the FRAMES (frame_stride: the lowest pixel rows of the side view shift
backward while a foot is planted).

Where it is stored (all written by the app, read by the runtimes):
    sheet.json  meta.stride {clip: {px, cycle_ms, frames, speed_px_s, heights, source}},
                meta.body_px (standing height in sheet px)
    unit.json   (hd2d.add_unit) stride {clip: {units, px}} with units = px / ppu (tiles)
    manifest    (gameexport) animations[clip].stride {px, units, cycle_ms}, body_px, px_scale,
                world_ppu (see world_scale)

MOTION = the runtime's tunables (acceleration, turn rate, walk/run blend, hit-stop, shake,
camera smoothing, hop height...). godot_runtime/pp_motion.gd implements them for Godot; the
algorithms are written out in docs/FOR_CLAUDE.md so a Unity (C#) runtime can follow.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LOCOMOTION = ("walk", "run", "sneak", "sprint", "jog", "trot", "gallop")
FEET_HINTS = ("foot_", "_paw", "paw_", "ankle", "toe")

# Runtime tunables (units: tiles and seconds; written to res://pp_runtime/motion.json and the
# export manifest "motion"). Unity/Godot runtimes read the same keys.
MOTION = {
    "accel": 14.0,             # tiles/s^2 to reach the target speed (walk 3 t/s in ~0.2 s)
    "decel": 18.0,             # tiles/s^2 to stop
    "turn_rate": 900.0,        # degrees/s: a 180 deg turn shows ~9 of 16 facings in 0.2 s
    "walk_speed": 2.6,         # tiles/s (tactics paths, autoplay goto default)
    "run_speed": 4.6,
    "run_blend": 0.72,         # speed / run_speed above which the run clip plays (walk below)
    "settle": 0.16,            # s: on stop, finish the step to the nearest contact frame
    "corner": 0.32,            # tiles: path corners rounded with this radius (tactics)
    "hop": 0.14,               # tiles above the higher ground when stepping a level up/down
    "step_time": 0.22,         # s per level change in a hop
    "rise": 0.07,              # s: free movement climbs onto a step this fast
    "gravity": 22.0,           # tiles/s^2: free movement drops off a ledge
    "foot_radius": 0.22,       # tiles: the ground under the feet = highest cell within this
    "body_radius": 0.24,       # tiles: action-mode collision circle
    "hitstop": 0.075,          # s: both fighters freeze on impact
    "knockback": 0.22,         # tiles a tactics hit pushes the target (it steps back after)
    "shake": 0.06,             # camera shake trauma per hit (squared, decays)
    "shake_decay": 2.6,        # trauma per second
    "anticipation": 0.12,      # s: wind-up hold before a blow (a lean back)
    "follow_through": 0.18,    # s: the last attack frame holds before idle
    "cam_smooth": 0.28,        # s: critically damped camera follow (smooth-damp time)
    "depth_bias": 0.35,        # tiles: sprite depth = its feet, this much toward the camera
    "default_stride": {"walk": 1.05, "run": 1.9},   # body heights per cycle when unmeasured
}


def _is_foot(name: str) -> bool:
    n = name.lower()
    return any(h in n for h in FEET_HINTS) and "toe_end" not in n


def is_locomotion(name: str) -> bool:
    n = name.lower()
    return any(n == k or n.startswith(k + "_") or n.endswith("_" + k) for k in LOCOMOTION)


# --------------------------------------------------------------------------- from the skeleton

def rig_stride(rig, clip, samples: int = 96) -> dict | None:
    """Stride of one looping clip in model units: {units, cycle_s, speed, heights, feet}.
    The feet (ankle/paw joints) are followed over one cycle with the clip's steady drift taken
    out (a clip that walks forward is made in place, like the sheet); while a foot is planted
    (within 4% of the body height of its lowest point) it moves backward at the body's speed.
    None when the rig has no feet or nothing is planted (a hop on the spot)."""
    if clip.duration <= 0:
        return None
    feet = [i for i, n in enumerate(rig.names) if _is_foot(n)]
    if not feet:
        return None
    rest = np.concatenate(rig.vertices(rig.worlds()))
    height = float(np.ptp(rest[:, 1])) or 1.0
    T = clip.duration
    ts = clip.start + T * np.arange(samples) / samples
    moved = rig.centroid(clip, clip.end) - rig.centroid(clip, clip.start)
    drift = moved * np.array([1.0, 0.0, 1.0]) / T
    P = np.array([[rig.worlds(clip, t)[i][:3, 3] for i in feet] for t in ts])   # (S, F, 3)
    P = P - drift[None, None, :] * (ts - clip.start)[:, None, None]
    dt = T / samples
    V = (np.roll(P, -1, axis=0) - P) / dt
    V[-1] = V[-2]                                          # no wrap jump on the last sample
    speeds = []
    for f in range(len(feet)):
        y = P[:, f, 1]
        planted = y <= y.min() + 0.04 * height
        if planted.sum() < 3:
            continue
        sp = np.hypot(V[planted, f, 0], V[planted, f, 2])
        speeds.append(np.median(sp))
    if not speeds:
        return None
    v = float(np.median(speeds))
    if v * T < 0.05 * height:                              # planted feet that don't slide
        v = float(np.hypot(drift[0], drift[2]))
    if v * T < 0.05 * height:
        return None
    return {"units": v * T, "cycle_s": T, "speed": v, "heights": v * T / height,
            "feet": [rig.names[i] for i in feet]}


def projected_height(rig, clip=None, elevation: float = 30.0) -> float:
    """Screen height (model units) of a pose seen by the sheet's camera at `elevation` (S view:
    y * cos e + z * sin e, like drawnrig.Camera / turnaround)."""
    pts = np.concatenate(rig.vertices(rig.worlds(clip, clip.start if clip is not None else 0.0)))
    e = math.radians(elevation)
    sy = pts[:, 1] * math.cos(e) - pts[:, 2] * math.sin(e)
    return float(np.ptp(sy))


# --------------------------------------------------------------------------- sheets

def _frames(meta: dict) -> list[tuple[str, dict, int]]:
    raw = meta.get("frames", [])
    items = list(raw.items()) if isinstance(raw, dict) else [(f["filename"], f) for f in raw]
    return [(n, f["frame"], int(f.get("duration", 100))) for n, f in items]


def rows_of(meta: dict) -> dict[str, list[tuple[dict, int]]]:
    """{"walk_E": [(rect, ms), ...]} in frame order."""
    out: dict[str, list] = {}
    for name, rect, ms in _frames(meta):
        out.setdefault(name.rsplit(" ", 1)[0], []).append((rect, ms))
    return out


def body_px(sheet: np.ndarray, meta: dict) -> float | None:
    """Standing height in sheet pixels: median opaque height of the idle frames facing the
    viewer (S, SE, SW; else any idle, else the first frames)."""
    rows = rows_of(meta)
    pick = [k for k in ("idle_S", "idle_SSE", "idle_SSW", "idle_SE", "idle_SW") if k in rows]
    pick = pick or [k for k in rows if k.startswith("idle_")] or list(rows)[:1]
    hs = []
    for k in pick[:3]:
        for rect, _ in rows[k]:
            a = sheet[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"], 3] >= 128
            ys = np.nonzero(a.any(1))[0]
            if len(ys):
                hs.append(ys.max() - ys.min() + 1)
    return float(np.median(hs)) if hs else None


def frame_stride(sheet: np.ndarray, meta: dict, clip: str) -> float | None:
    """Stride in sheet px from the FRAMES alone (no skeleton): the side view (E, else W) of the
    clip; between frames, the pixels of the lowest rows (the planted foot) shift backward by the
    body's step. Sum of the per-frame shifts over the cycle. None when unreadable."""
    rows = rows_of(meta)
    for d, sign in (("E", 1), ("W", -1)):
        key = f"{clip}_{d}"
        if key in rows and len(rows[key]) >= 4:
            break
    else:
        return None
    masks = []
    for rect, _ in rows[key]:
        masks.append(sheet[rect["y"]:rect["y"] + rect["h"], rect["x"]:rect["x"] + rect["w"], 3] >= 128)
    ground = max((np.nonzero(m.any(1))[0].max() for m in masks if m.any()), default=None)
    if ground is None:
        return None
    hgt = body_px(sheet, meta) or masks[0].shape[0] / 1.5
    band = max(2, int(round(0.05 * hgt)))
    bands = [m[max(0, ground - band + 1):ground + 1].astype(np.float32) for m in masks]
    W = bands[0].shape[1]
    limit = max(2, int(round(0.25 * hgt)))
    total = 0.0
    for i in range(len(bands)):
        a, b = bands[i], bands[(i + 1) % len(bands)]
        if a.sum() < 2 or b.sum() < 2:
            continue
        best, best_s = -1.0, 0
        for s in range(0, limit + 1):          # planted pixels move BACK: -x for E, +x for W
            dx = -s * sign
            if dx >= 0:
                ov = (a[:, :W - dx] * b[:, dx:]).sum() if dx < W else 0.0
            else:
                ov = (a[:, -dx:] * b[:, :W + dx]).sum() if -dx < W else 0.0
            if ov > best + 1e-6:
                best, best_s = ov, s
        total += best_s
    return float(total) if total > 0 else None


def measure_sheet(sheet_png, rigged=None, px_per_unit: float | None = None,
                  elevation: float = 30.0) -> dict:
    """Stride of every locomotion clip of a sheet (+ its standing height): from the skeleton
    when `rigged` is given (px_per_unit = the camera scale; None = measured from the idle frame),
    else from the frames. {"stride": {clip: {...}}, "body_px": h}."""
    from PIL import Image
    sheet_png = Path(sheet_png)
    meta = json.loads(sheet_png.with_suffix(".json").read_text(encoding="utf-8"))
    img = np.asarray(Image.open(sheet_png).convert("RGBA"))
    rows = rows_of(meta)
    clips = sorted({k.rsplit("_", 1)[0] for k in rows if is_locomotion(k.rsplit("_", 1)[0])})
    hgt = body_px(img, meta)
    out: dict = {}
    rig = None
    if rigged is not None and Path(rigged).exists():
        from .anim3d import load_rig
        try:
            rig = load_rig([rigged])
        except Exception:  # noqa: BLE001 - unreadable rig: the frames then
            rig = None
    by_name = {c.name: c for c in rig.clips} if rig is not None else {}
    if rig is not None and px_per_unit is None and hgt:
        idle = by_name.get("idle")
        ph = projected_height(rig, idle, elevation)
        px_per_unit = hgt / ph if ph > 0 else None
    for clip in clips:
        n = len(next(v for k, v in rows.items() if k.startswith(clip + "_")))
        ms = [m for _, m in next(v for k, v in rows.items() if k.startswith(clip + "_"))]
        cycle_ms = float(sum(ms))
        entry = None
        if rig is not None and clip in by_name and px_per_unit:
            st = rig_stride(rig, by_name[clip])
            if st is not None:
                px = st["units"] * px_per_unit
                entry = {"px": round(px, 2), "source": "rig", "heights": round(st["heights"], 3)}
        if entry is None:
            px = frame_stride(img, meta, clip)
            if px is not None:
                entry = {"px": round(px, 2), "source": "frames"}
        if entry is None and hgt:
            k = MOTION["default_stride"]["run" if "run" in clip else "walk"]
            entry = {"px": round(k * hgt, 2), "source": "default"}
        if entry is None:
            continue
        entry.update(frames=n, cycle_ms=round(cycle_ms, 1),
                     speed_px_s=round(entry["px"] / max(cycle_ms / 1000.0, 1e-6), 2))
        if hgt:
            entry["heights"] = round(entry["px"] / hgt, 3)
        out[clip] = entry
    return {"stride": out, "body_px": round(hgt, 1) if hgt else None}


def annotate_sheet(sheet_png, info: dict) -> Path:
    """Write measure_sheet's result into the sheet's json (meta.stride, meta.body_px)."""
    js = Path(sheet_png).with_suffix(".json")
    meta = json.loads(js.read_text(encoding="utf-8"))
    m = meta.setdefault("meta", {})
    m["stride"] = info.get("stride", {})
    if info.get("body_px"):
        m["body_px"] = info["body_px"]
    js.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return js


def sheet_stride(sheet_json) -> tuple[dict, float | None]:
    """(meta.stride, meta.body_px) of a sheet json (({}, None) when it has none)."""
    try:
        meta = json.loads(Path(sheet_json).read_text(encoding="utf-8")).get("meta", {})
    except (OSError, ValueError):
        return {}, None
    return dict(meta.get("stride") or {}), meta.get("body_px")


def ensure_stride(sheet_png, rigged=None, px_per_unit=None, elevation: float = 30.0) -> dict:
    """The sheet's stride info, measured + written into its json if missing."""
    stride, h = sheet_stride(Path(sheet_png).with_suffix(".json"))
    if stride:
        return {"stride": stride, "body_px": h}
    info = measure_sheet(sheet_png, rigged, px_per_unit, elevation)
    if info["stride"]:
        annotate_sheet(sheet_png, info)
    return info


def frame_counts_of(sheet_json) -> dict[str, int]:
    """{clip: frames} of an existing sheet (frames of its first direction)."""
    try:
        meta = json.loads(Path(sheet_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for key, fr in rows_of(meta).items():
        out.setdefault(key.rsplit("_", 1)[0], len(fr))
    return out


# --------------------------------------------------------------------------- world scale

def px_scale(frame_h: int, kind: str = "person", size: str | None = None) -> float:
    """How many times bigger than the library's 64 px-person scale a sheet was drawn (Neo16's
    144x96 knight = 1.5, a QA 96x64 elf = 1.0): frame height / the library frame for its kind
    and size. Divide by it (or multiply ppu by it) to put every sheet on one world scale."""
    from .library import DEFAULT_SIZE, SIZES
    sizes = SIZES.get(kind, SIZES["person"])
    w, h = sizes.get(size or DEFAULT_SIZE.get(kind, "normal"), sizes[DEFAULT_SIZE.get(kind, "normal")])
    return float(frame_h) / float(h)


def world_ppu(base_ppu: float, frame_h: int, kind: str = "person", size: str | None = None) -> float:
    """Sprite pixels per tile so this sheet stands at the same world scale as a library-scale
    one at base_ppu."""
    return float(base_ppu) * px_scale(frame_h, kind, size)


def write_motion_json(folder, extra: dict | None = None) -> Path:
    p = Path(folder) / "motion.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({**MOTION, **(extra or {})}, indent=1), encoding="utf-8")
    return p
