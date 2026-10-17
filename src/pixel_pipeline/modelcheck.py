"""Quality checks and fixes for a 3D shape straight out of Hunyuan3D, before it is painted and rigged
(2026-10-17, Kevin: the elf model had huge ribbon arcs and floating bow pieces -> 8/16 directions
passed, the side elf 0/2). CPU only, no model, a few seconds.

    rep = check_picture(rgba)            # the cut-out front picture: thin curved things far out
    glb, rep = fix_shape(glb_bytes)      # the shape: floaters dropped, flyaway ribbons trimmed

Shape (voxels on a VOXELS grid over the shape's longest side):
  floaters   pieces not joined to the main body: dropped when small or apart from it
  ribbons    THIN parts (thinner than the arms: survive no opening of CORE_R x height) that reach
             far out from the thick body (> FAR_REACH x height) and are not straight (a spear or a
             sword is a straight rod and stays; a ribbon arc, a sash flying out, a bow floating
             behind the back curve or spread as sheets): trimmed at KEEP x height from the body,
             so a cape keeps its hem and a sash its knot
Picture (the same on the cut-out's alpha, 2D): the share of the subject in such flyaways; too much
-> the picture is drawn again (picture_for_model's check) before the GPU spends a minute on a
shape.

`score(rep)` = how bad (0 good); prompt_to_model keeps the better of up to SHAPE_TRIES shapes.
"""
from __future__ import annotations

import io

import numpy as np

VOXELS = 128
CORE_R = 0.018          # x height: opening radius; arms / legs / torso / head survive it
KEEP = 0.06             # x height: thin parts this near the thick body always stay
FAR_REACH = 0.15        # x height: thin parts reaching further out are flyaways ...
ROD = 0.22              # ... unless straight (second / first principal spread under this)
HANG_REACH = 0.10       # x height: horizontally this far from the body at or above it = out
OUT_SHARE = 0.30        # ... and a part with less of itself out than this hangs (cape, robe)
BIG_SHEET = 0.08        # a flyaway this big (share of the shape) is a garment (a cloak blown
                        # out): only what flies past 2 x HANG_REACH goes, the draped part stays
FLOAT_GAP = 0.02        # x height: a separate piece further than this from the body floats
FLOAT_KEEP = 0.04       # share of the body's volume: bigger close pieces stay (a shield)
PICTURE_FLYAWAY = 0.01   # share of the picture's subject in flyaways: draw it again past this
PICTURE_REACH = 0.18     # ... or one flyaway reaching this far (x height) out


# --------------------------------------------------------------------------- shared

def _pca_spread(coords: np.ndarray) -> float:
    """sqrt(second / first eigenvalue) of a point set: ~0 for a straight rod, larger for an arc
    or a sheet."""
    if len(coords) < 5:
        return 0.0
    c = coords - coords.mean(0)
    ev = np.sort(np.linalg.eigvalsh(np.cov(c.T)))[::-1]
    return float(np.sqrt(max(ev[1], 0.0) / max(ev[0], 1e-12)))


def _hang_distance(core: np.ndarray, up: int, top_first: bool) -> np.ndarray:
    """Per voxel / pixel: the horizontal distance to the thick body at its height OR ABOVE (what a
    cape, a robe or hair can hang from). Things hanging under the body stay near 0; a ribbon
    flying out to the side or an arc above the head are far."""
    from scipy import ndimage as ndi
    c = np.moveaxis(core, up, 0)
    if not top_first:                          # index 0 = the top
        c = c[::-1]
    acc = np.zeros(c.shape[1:], bool)
    out = np.empty(c.shape, np.float32)
    big = float(sum(c.shape))
    for i in range(c.shape[0]):
        acc |= c[i]
        out[i] = ndi.distance_transform_edt(~acc) if acc.any() else big
    if not top_first:
        out = out[::-1]
    return np.moveaxis(out, 0, up)


def _flyaways(solid: np.ndarray, height_vox: float, structure, up: int = 1,
              top_first: bool = False) -> tuple[np.ndarray, list[dict]]:
    """(mask of flyaway voxels / pixels to trim, their parts) in a solid 2D or 3D mask. up: the
    up axis; top_first: index 0 is the top (pictures) rather than the bottom (shapes, y up)."""
    from scipy import ndimage as ndi
    r = max(1.5, CORE_R * height_vox)
    inside = ndi.distance_transform_edt(solid)
    core = ndi.distance_transform_edt(~(inside > r)) <= r      # opening by a ball of radius r
    core &= solid
    if not core.any():
        return np.zeros_like(solid), []
    dist = ndi.distance_transform_edt(~core)
    far = solid & ~core & (dist > KEEP * height_vox)
    lab, n = ndi.label(far, structure=structure)
    trim = np.zeros_like(solid)
    parts = []
    if n == 0:
        return trim, parts
    hang = _hang_distance(core, up, top_first)
    reach = ndi.maximum(dist, lab, np.arange(1, n + 1))
    for i in range(1, n + 1):
        reach_i = float(reach[i - 1]) / height_vox
        if reach_i <= FAR_REACH:
            continue
        sel = lab == i
        coords = np.argwhere(sel).astype(np.float64)
        spread = _pca_spread(coords)
        out_share = float((hang[sel] > HANG_REACH * height_vox).mean())
        straight = spread < ROD
        hanging = out_share < OUT_SHARE         # a cape / robe / hem under the body
        parts.append({"reach": round(reach_i, 3), "spread": round(spread, 3),
                      "out": round(out_share, 3),
                      "size": round(float(sel.sum()) / max(1, solid.sum()), 4),
                      "kept": bool(straight or hanging)})
        if not (straight or hanging):
            if parts[-1]["size"] > BIG_SHEET:   # a whole cloak flying out: only its far flare goes
                trim |= sel & (hang > 2.0 * HANG_REACH * height_vox)
                parts[-1]["kept"] = "near part"
            else:
                trim |= sel
    return trim, parts


# --------------------------------------------------------------------------- picture

def check_picture(rgba: np.ndarray) -> dict:
    """Flyaways (ribbons, sashes, floating hair, bows far from the body) in a cut-out picture.
    {"flyaway": share of the subject, "reach": the furthest, "parts": [...], "bad": reason|None}."""
    from scipy import ndimage as ndi
    a = np.asarray(rgba)[..., 3] > 127
    if not a.any():
        return {"flyaway": 0.0, "reach": 0.0, "parts": [], "bad": None}
    ys, xs = np.nonzero(a)
    k = max(1, int(round(max(np.ptp(ys), np.ptp(xs)) / 256)))        # work at ~256 px
    small = a[::k, ::k]
    lab, n = ndi.label(small, structure=np.ones((3, 3)))
    if n > 1:                                                     # the subject = biggest piece
        sizes = np.bincount(lab.ravel())[1:]
        small = lab == (np.argmax(sizes) + 1)
    h = float(np.ptp(np.nonzero(small.any(1))[0]) + 1)
    trim, parts = _flyaways(small, h, np.ones((3, 3)), up=0, top_first=True)
    share = float(trim.sum() / max(1, small.sum()))
    reach = max((p["reach"] for p in parts if not p["kept"]), default=0.0)
    bad = None
    if share > PICTURE_FLYAWAY or reach > PICTURE_REACH:
        bad = (f"thin curved things flying out from the body ({share:.0%} of it, reaching "
               f"{reach:.2f} of its height: ribbons / sash / floating bow)")
    return {"flyaway": round(share, 4), "reach": round(reach, 3), "parts": parts, "bad": bad}


# --------------------------------------------------------------------------- shape

def _load(glb):
    import trimesh
    if isinstance(glb, trimesh.Trimesh):
        return glb
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb") if isinstance(glb, (bytes, bytearray)) \
        else trimesh.load(str(glb))
    if not isinstance(loaded, trimesh.Trimesh):
        loaded = loaded.to_geometry() if hasattr(loaded, "to_geometry") else loaded.dump(concatenate=True)
    return loaded


def _voxels(mesh, voxels: int = VOXELS, seed: int = 0):
    """Solid occupancy of a mesh: (solid, origin, step, face voxel index)."""
    from scipy import ndimage as ndi
    import trimesh
    lo, hi = mesh.bounds
    step = float((hi - lo).max()) / (voxels - 4)
    origin = lo - 2 * step
    shape = tuple(np.ceil((hi - origin) / step).astype(int) + 3)
    area = float(mesh.area)
    count = int(np.clip(area / (step * step) * 6, 50_000, 1_500_000))
    pts, _ = trimesh.sample.sample_surface(mesh, count, seed=seed)
    pts = np.vstack([pts, mesh.triangles_center])
    ijk = np.clip(((pts - origin) / step).astype(int), 0, np.array(shape) - 1)
    surf = np.zeros(shape, bool)
    surf[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    surf = ndi.binary_closing(surf, np.ones((3, 3, 3)), iterations=1) | surf
    solid = ndi.binary_fill_holes(surf)
    fc = np.clip(((mesh.triangles_center - origin) / step).astype(int), 0, np.array(shape) - 1)
    return solid, origin, step, fc


def check_shape(glb) -> dict:
    """check + fix plan without changing anything (fix_shape does both)."""
    return fix_shape(glb, apply=False)[1]


def fix_shape(glb, apply: bool = True, ribbons: bool = True):
    """(fixed glb bytes, report). Floaters dropped, flyaway ribbons / arcs trimmed (module doc).
    The report: {"floaters": n dropped, "floater_share", "ribbons": [...], "trimmed": share of the
    surface removed, "faces_before/after", "bad_after": what is still wrong}."""
    from scipy import ndimage as ndi
    mesh = _load(glb)
    solid, origin, step, fc = _voxels(mesh)
    s26 = np.ones((3, 3, 3))
    lab, n = ndi.label(solid, structure=s26)
    keep = np.zeros_like(solid)
    floaters = 0
    float_share = 0.0
    up = 1                                     # y up (Hunyuan / glTF)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        main = int(np.argmax(sizes)) + 1
        body = lab == main
        height_vox = float(np.ptp(np.nonzero(body.any(axis=tuple(a for a in range(3) if a != up)))[0]) + 1)
        gap = ndi.distance_transform_edt(~body)
        keep |= body
        for i in range(1, n + 1):
            if i == main:
                continue
            sel = lab == i
            near = float(gap[sel].min()) <= FLOAT_GAP * height_vox
            if near and sizes[i - 1] >= FLOAT_KEEP * sizes[main - 1]:
                keep |= sel                    # a big close piece (a shield, a separate boot)
            else:
                floaters += 1
                float_share += float(sizes[i - 1]) / float(sizes.sum())
    else:
        height_vox = float(solid.shape[up])
    trim, found = _flyaways(keep, height_vox, s26, up=up)
    if ribbons:                                # off when the words ask for ribbons / sashes
        keep &= ~trim
    ok_face = keep[fc[:, 0], fc[:, 1], fc[:, 2]]
    # faces whose centre voxel is empty after the closing (thin sheets between voxels): keep
    # them when a neighbour voxel is kept
    if (~ok_face).any():
        grown = ndi.binary_dilation(keep, s26)
        loose = ~ok_face & ~solid[fc[:, 0], fc[:, 1], fc[:, 2]]
        ok_face |= loose & grown[fc[:, 0], fc[:, 1], fc[:, 2]]
    area = mesh.area_faces
    trimmed = float(area[~ok_face].sum() / max(area.sum(), 1e-12))
    rep = {"floaters": floaters, "floater_share": round(float_share, 4),
           "ribbons": found, "trimmed": round(trimmed, 4),
           "faces_before": int(len(mesh.faces)), "faces_after": int(ok_face.sum())}
    bad = []
    if trimmed > 0.25:
        bad.append(f"{trimmed:.0%} of the surface was flyaways")
    rep["bad_after"] = bad
    if not apply or ok_face.all():
        return (glb if not apply else _to_bytes(mesh, glb)), rep
    import trimesh
    out = trimesh.Trimesh(mesh.vertices, mesh.faces[ok_face], process=False)
    out.remove_unreferenced_vertices()
    return out.export(file_type="glb"), rep


def _to_bytes(mesh, original):
    if isinstance(original, (bytes, bytearray)):
        return bytes(original)
    return mesh.export(file_type="glb")


def score(rep: dict, picture_match: float | None = None) -> float:
    """How bad a shape is after its fixes (0 = clean): trimmed surface, floaters, and a poor
    silhouette match with the picture (paint_from_picture's)."""
    s = 2.0 * rep.get("trimmed", 0.0) + 0.5 * rep.get("floater_share", 0.0) + 0.02 * rep.get("floaters", 0)
    if picture_match is not None:
        s += 2.0 * max(0.0, 0.7 - picture_match)
    return float(s)
