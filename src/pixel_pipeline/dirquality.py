"""Per-direction quality of a character's sheet, as the game shows it (2026-10-17, Kevin: "the
artifacts are more at specific angles").

    rep = check_sheet(sheet_png, keys_report)       # {"directions": {d: {...metrics, ok, why}}, ...}
    worst = to_redraw(rep, most=4)                  # the failing directions, worst first

Every direction of the sheet is measured on its idle frames and its first walk frame (the frames
the player sees while standing and while turning through the facings) against the reference
direction (S for people, E for animals: the drawing the user approved):

  specks    1-2 px colour islands per 100 body pixels (downsampling / bending noise)
  blotch    share of body pixels in big near-black clusters that the reference doesn't have
            (muddy side views: armour drawn as one dark mass)
  bits      body pixels in pieces cut off from the main body (torn bits floating beside it)
  flicker   share of body pixels that change colour between idle frames beyond the motion
            (bending tears that crawl while standing still)
  colour    Lab-histogram distance to the reference (drawnrig.colour_distance; a design change)
  sharp     local contrast relative to the reference's (soft or flat views)
  + from the drawing's own keys_report (drawnrig): IoU with the model, off-model share,
    whole-part colour misses / grey share of back views

A direction fails when a metric is past its LIMIT (relative to the reference where that makes
sense). Recalibrated 2026-10-17 on 11 drawn characters (the 2x tests, QA 2026-10-15, Neo16 test):
blotch counts only near-BLACK (dark AND colourless: a deep purple robe is a material), foreign
weighs lightness half (a lighter shade of the same cloth is no new colour), noise metrics are
judged against the views of the same half, look metrics (dark, colour, sharpness) against the
views turned the same way (people: within 90 deg of S, else turned away: E, W and the backs,
which drawnrig.relight_view lifts alike); HARD blotch 0.22 = a back drawn as one black mass fails
even when every back is like that. `to_redraw` gives the failing ones worst first; library's automatic chain redraws them
once with a new seed (a direction is never redrawn twice for the same reason) and keeps the
better of old and new (`better`). Pure numpy/OpenCV, ~0.2 s per character.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

DIRS16 = ("E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE")

# metric: (limit, relative-to-reference factor or None). fail when value > max(limit, ref*factor)
# for "worse is bigger" metrics; sharp fails when value < ref * SHARP_MIN.
# metric: (absolute limit, factor of the character's own median over its directions): a direction
# fails past BOTH (an outlier among its own directions AND bad in itself); None = absolute only.
LIMITS = {"specks": (4.0, 1.45), "blotch": (0.10, 1.6), "bits": (0.012, None), "flicker": (0.10, 1.6),
          "colour": (0.36, 1.35), "foreign": (0.04, 2.0), "off_model": (0.06, None), "miss": (0.2, None),
          "grey": (0.14, None)}
# past these a direction fails even when the character's other directions are as bad (a whole
# side drawn as one dark mass, noise everywhere)
# noise metrics are judged against the views of the same half (front: S half + E/W, back: N half:
# fronts carry more detail, so more 1-px specks); the look metrics (dark, colour, sharpness)
# against the views turned the same way (people: front within 90 deg of S, else turned away)
NOISE = ("specks", "flicker", "bits", "iou", "off_model")
BLOTCH_CHROMA = 12.0          # Lab chroma under this (and L < 18) = near-black for blotch
HARD = {"specks": 9.0, "blotch": 0.22, "bits": 0.03, "flicker": 0.22}
SHARP_HARD = 0.3
IOU_MIN = 0.82                  # fails under this AND 0.06 under the character's median
IOU_FLOOR = 0.70                # fails under this whatever the others do
SHARP_MIN = 0.55                # x the reference's local contrast, and under 0.8 x the median
WEIGHTS = {"specks": 0.25, "blotch": 3.0, "bits": 20.0, "flicker": 2.0, "colour": 2.0, "foreign": 6.0,
           "off_model": 4.0, "miss": 1.5, "grey": 1.5}


def _frames(sheet_png) -> tuple[np.ndarray, dict]:
    sheet = np.asarray(Image.open(sheet_png).convert("RGBA"))
    meta = json.loads(Path(sheet_png).with_suffix(".json").read_text(encoding="utf-8"))
    rects = {}
    for f in meta["frames"]:
        r = f["frame"]
        rects[f["filename"]] = (r["x"], r["y"], r["w"], r["h"])
    return sheet, rects


def _crop(sheet, r):
    x, y, w, h = r
    return sheet[y:y + h, x:x + w]


def _labels(rgba: np.ndarray):
    """Connected regions of one exact colour (4-neighbour) inside the body."""
    import cv2
    on = rgba[..., 3] > 0
    key = (rgba[..., 0].astype(np.int32) << 16) | (rgba[..., 1].astype(np.int32) << 8) | rgba[..., 2]
    key = np.where(on, key, -1)
    lab = np.zeros(key.shape, np.int32)
    n = 0
    for v in np.unique(key[on]):
        m = (key == v).astype(np.uint8)
        k, cc = cv2.connectedComponents(m, connectivity=4)
        cc = np.where(cc > 0, cc + n, 0)
        lab = np.where(m > 0, cc, lab)
        n += k - 1
    return lab, n, on


def specks(rgba: np.ndarray) -> float:
    """1-2 px single-colour islands per 100 body pixels, not counting the outline ring."""
    lab, n, on = _labels(rgba)
    if not on.any():
        return 0.0
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    small = (sizes[lab] <= 2) & on
    # the outline ring is meant to be thin: islands touching the transparent edge don't count
    import cv2
    inner = cv2.erode(on.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float(100.0 * (small & inner).sum() / max(1, on.sum()))


def blotch(rgba: np.ndarray, ref_dark: float) -> float:
    """Share of the body in big near-black clusters beyond what the reference has (L < 18,
    chroma < BLOTCH_CHROMA)."""
    import cv2
    from .color import rgb_to_lab
    on = rgba[..., 3] > 0
    if not on.any():
        return 0.0
    lab = rgb_to_lab(rgba[..., :3])
    L = lab[..., 0]
    # near-BLACK: dark AND colourless. A deep purple robe or a dark red cloak is a material, not
    # mud (recalibrated on 11 characters, 2026-10-17: the mage's shaded robe failed 6 views)
    dark = (L < 18) & (np.hypot(lab[..., 1], lab[..., 2]) < BLOTCH_CHROMA) & on
    inner = cv2.erode(on.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    dark &= inner                                   # the outline is allowed to be dark
    k, cc, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    big = np.zeros_like(dark)
    least = max(12, int(0.01 * on.sum()))
    for i in range(1, k):
        if stats[i, cv2.CC_STAT_AREA] >= least:
            big |= cc == i
    return float(max(0.0, big.sum() / on.sum() - ref_dark))


def dark_share(rgba: np.ndarray) -> float:
    return blotch(rgba, 0.0)


def bits(rgba: np.ndarray) -> float:
    """Body pixels outside the biggest piece (8-connected) / body pixels."""
    import cv2
    on = (rgba[..., 3] > 0).astype(np.uint8)
    if not on.any():
        return 0.0
    k, cc, stats, _ = cv2.connectedComponentsWithStats(on, connectivity=8)
    if k <= 2:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    return float((areas.sum() - areas.max()) / areas.sum())


def flicker(frames: list[np.ndarray]) -> float:
    """Share of body pixels whose colour changes between consecutive idle frames where the
    silhouette did not move (crawling tears / speckle while standing still)."""
    if len(frames) < 2:
        return 0.0
    tot, ch = 0, 0
    for a, b in zip(frames, frames[1:]):
        both = (a[..., 3] > 0) & (b[..., 3] > 0)
        # pixels whose 3x3 neighbourhood is body in both frames (not the moving edge)
        import cv2
        steady = cv2.erode(both.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        diff = np.abs(a[..., :3].astype(int) - b[..., :3].astype(int)).sum(-1) > 30
        tot += steady.sum()
        ch += (diff & steady).sum()
    return float(ch / max(1, tot))


def sharp(rgba: np.ndarray) -> float:
    from .drawnrig import sharpness
    on = rgba[..., 3] > 0
    return float(sharpness(rgba[..., :3], on)) if on.any() else 0.0


def colour(rgba: np.ndarray, ref: np.ndarray) -> float:
    from .drawnrig import colour_distance
    a, b = rgba[..., 3] > 0, ref[..., 3] > 0
    if not a.any() or not b.any():
        return 1.0
    return float(colour_distance(rgba[..., :3], a, ref[..., :3], b))


def foreign(rgba: np.ndarray, ref: np.ndarray, far: float = 22.0) -> float:
    """Share of body pixels whose colour is far (Lab distance, L half weight) from every colour of the
    reference frame: things the approved design doesn't have (a brown sack on a knight)."""
    from .color import rgb_to_lab
    a, b = rgba[..., 3] > 0, ref[..., 3] > 0
    if not a.any() or not b.any():
        return 0.0
    cols, inv = np.unique(rgba[a][:, :3].reshape(-1, 3), axis=0, return_inverse=True)
    rc = np.unique(ref[b][:, :3].reshape(-1, 3), axis=0)
    la = rgb_to_lab(cols[None].astype(np.uint8))[0]
    lb = rgb_to_lab(rc[None].astype(np.uint8))[0]
    # lightness counts half: a lighter / darker shade of the same material is not a new thing
    # (relit back views lift the cloak's shadows; drawnrig.relight_view)
    w = np.array([0.5, 1.0, 1.0], np.float32)
    d = np.sqrt((((la[:, None, :] - lb[None, :, :]) * w) ** 2).sum(-1)).min(1)
    return float((d[inv.ravel()] > far).mean())


def reference_of(names, kind: str = "person") -> str:
    ref = "E" if kind == "animal" else "S"
    return ref if ref in names else names[0]


def measure(sheet_png, keys_report: dict | None = None, kind: str = "person") -> dict:
    """{"reference", "directions": {d: metrics}} for every direction of the sheet."""
    sheet, rects = _frames(sheet_png)
    clips = sorted({k.split(" ")[0].rsplit("_", 1)[0] for k in rects})
    still = "idle" if "idle" in clips else (clips[0] if clips else "idle")   # standing frames
    names = [d for d in DIRS16 if f"{still}_{d} 0" in rects]
    if not names:
        raise ValueError(f"{sheet_png}: no frames to check")
    idle = {d: [_crop(sheet, rects[k]) for k in sorted((k for k in rects if k.startswith(f"{still}_{d} ")),
                                                        key=lambda s: int(s.rsplit(" ", 1)[1]))] for d in names}
    walk0 = {d: _crop(sheet, rects[f"walk_{d} 0"]) for d in names if f"walk_{d} 0" in rects}
    ref = reference_of(names, kind)
    r0 = idle[ref][0]
    ref_dark = dark_share(r0)
    ref_m = {"specks": np.mean([specks(f) for f in [r0] + ([walk0[ref]] if ref in walk0 else [])]),
             "flicker": flicker(idle[ref]), "sharp": sharp(r0)}
    out = {}
    for d in names:
        f0 = idle[d][0]
        frames = [f0] + ([walk0[d]] if d in walk0 else [])
        m = {"specks": float(np.mean([specks(f) for f in frames])),
             "blotch": float(np.mean([blotch(f, ref_dark) for f in frames])),
             "bits": float(max(bits(f) for f in frames)),
             "flicker": flicker(idle[d]),
             "colour": 0.0 if d == ref else colour(f0, r0),
             "foreign": 0.0 if d == ref else foreign(f0, r0),
             "sharp": sharp(f0) / max(1e-6, ref_m["sharp"]),
             "body": int((f0[..., 3] > 0).sum())}
        kr = (keys_report or {}).get(d) or {}
        for k_src, k in (("iou", "iou"), ("off_model", "off_model"), ("miss", "miss"), ("grey", "grey")):
            if isinstance(kr.get(k_src), (int, float)):
                m[k] = float(kr[k_src])
        if kr.get("candidate"):
            m["candidate"] = str(kr["candidate"])
        out[d] = m
    return {"reference": ref, "ref": {k: float(v) for k, v in ref_m.items()}, "directions": out}


def side_of(d: str, ref: str = "E") -> str:
    """front or back group of a direction name. People (reference S): back = turned 90 deg or
    more from the front (E, W and the N half: they show the cloak side and are relit alike,
    drawnrig.relight_view); otherwise (animals, reference E): back = the N half."""
    if ref == "S" and d in DIRS16:
        k = DIRS16.index(d)
        turn = abs(((k - DIRS16.index("S")) * 22.5 + 180.0) % 360.0 - 180.0)
        return "back" if turn >= 90.0 else "front"
    return "back" if "N" in d else "front"


def judge(rep: dict) -> dict:
    """Adds ok / why / badness per direction and a summary to a measure() report."""
    ref = rep["reference"]
    dirs = rep["directions"]
    meds = {}

    def group(d, k):
        return side_of(d) if k in NOISE else side_of(d, ref)
    for grp in ("front", "back"):          # back views are judged against back views (a cloak
        med = {}                           # seen from behind is darker and flatter than a face)
        for k in list(LIMITS) + ["iou", "sharp"]:
            vals = [m[k] for d, m in dirs.items() if k in m and group(d, k) == grp
                    and not (d == ref and k in ("colour", "foreign"))]
            if not vals:
                vals = [m[k] for d, m in dirs.items() if k in m]
            if vals:
                med[k] = float(np.median(vals))
        meds[grp] = med
    rep["median"] = {g: {k: round(v, 4) for k, v in m.items()} for g, m in meds.items()}
    fails = []
    for d, m in dirs.items():
        med = {k: meds[group(d, k)].get(k, 0.0) for k in set(meds["front"]) | set(meds["back"])}
        why, bad = [], 0.0
        for k, (lim, fac) in LIMITS.items():
            if k not in m:
                continue
            cap = max(lim, med.get(k, 0.0) * fac) if fac else lim
            cap = min(cap, HARD.get(k, cap))
            if m[k] > cap:
                why.append(k)
                bad += WEIGHTS[k] * (m[k] - cap) + 0.5
        if "iou" in m and ((m["iou"] < IOU_MIN and m["iou"] < med.get("iou", 1.0) - 0.06)
                           or m["iou"] < IOU_FLOOR):
            why.append("iou")
            bad += 4.0 * (IOU_MIN - m["iou"]) + 0.5
        if (m["sharp"] < SHARP_MIN and m["sharp"] < 0.8 * med.get("sharp", 1.0)) or m["sharp"] < SHARP_HARD:
            why.append("sharp")
            bad += 2.0 * (SHARP_MIN - m["sharp"]) + 0.5
        if d == rep["reference"]:
            why = [w for w in why if w not in ("colour", "sharp")]
        m["ok"] = not why
        m["why"] = why
        m["badness"] = round(bad, 3)
        if why:
            fails.append(d)
    n = len(rep["directions"])
    rep["passed"] = n - len(fails)
    rep["total"] = n
    rep["failed"] = sorted(fails, key=lambda d: -rep["directions"][d]["badness"])
    return rep


def check_sheet(sheet_png, keys_report: dict | None = None, kind: str = "person") -> dict:
    return judge(measure(sheet_png, keys_report, kind))


def to_redraw(rep: dict, most: int = 4, skip=()) -> list[str]:
    """The failing directions worst first (the reference never: the user approved it), at most
    `most`; `skip` = directions already redrawn for the same reasons."""
    ref = rep.get("reference")
    out = []
    for d in rep.get("failed", []):
        if d == ref:
            continue
        why = tuple(rep["directions"][d]["why"])
        if (d, why) in skip or d in {s[0] if isinstance(s, tuple) else s for s in skip}:
            continue
        out.append(d)
    return out[:most]


def better(old: dict, new: dict) -> bool:
    """Is the new measurement of one direction better than the old one?"""
    return (len(new.get("why", ())), new.get("badness", 0)) < (len(old.get("why", ())), old.get("badness", 0))


def summary_text(q: dict) -> str:
    """'16/16 directions passed' / '14/16 directions passed (redrawn: NE, ENE; still weak: E)'."""
    if not q:
        return ""
    s = f"{q.get('passed', '?')}/{q.get('total', '?')} directions passed"
    extra = []
    if q.get("redrawn"):
        extra.append("redrawn: " + ", ".join(q["redrawn"]))
    if q.get("failed"):
        extra.append("still weak: " + ", ".join(f"{d} ({'/'.join(q['directions'][d]['why'])})"
                                               for d in q["failed"] if d in q.get("directions", {})))
    return s + (f" ({'; '.join(extra)})" if extra else "")


def compact(rep: dict) -> dict:
    """What keys.json / asset.json keep (rounded)."""
    dirs = {d: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
            for d, m in rep.get("directions", {}).items()}
    out = {k: v for k, v in rep.items() if k != "directions"}
    out["directions"] = dirs
    return out
