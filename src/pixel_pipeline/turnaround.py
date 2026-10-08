"""Turnarounds: one subject facing 4, 8 or 16 directions, as sprites sharing one palette.

Two sources:

3D model (.glb .gltf .obj .ply .stl .off), e.g. from an image-to-3D model (Hunyuan3D,
    Stable Fast 3D, TripoSR, TRELLIS, or an online one like Tripo or Meshy). The model is
    rendered from every direction with an orthographic camera tilted down by `elevation`
    degrees (0 = side view, about 30 = classic top-down / isometric), so every direction has
    the same scale and lighting. Pre-rendered games (Diablo, StarCraft, Fallout) made their
    8- and 16-direction sprites the same way.

Photos or a video taken while walking around a subject that stays still: start in front of
    it (it faces you), walk all the way around to your right keeping it centred, at the
    height your game's camera looks from. One photo per direction, in order, or a video that
    ends where it started. reverse=True if you walked to your left. mirror=True: capture only
    the front-to-back half (directions // 2 + 1 views) and mirror the other side; only for
    left/right-symmetric subjects (markings flip).

Directions follow Godot's angles: index 0 faces right (E) and they run clockwise on screen,
E, ESE, SE, SSE, S (facing the viewer), ... N (facing away) ... ENE. The Godot script written
next to the sprite frames picks the facing with
DIRECTIONS[wrapi(roundi(direction.angle() / TAU * N), 0, N)].
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .aseprite import write_aseprite
from .export import pack_sheet, write_sheet
from .godot import write_sprite_frames
from .pipeline import Sprite, sprites_from_rgba
from .settings import Settings
from .video import _crop_pad, align_frames

COMPASS16 = ("E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW",
             "W", "WNW", "NW", "NNW", "N", "NNE", "NE", "ENE")
DIRECTION_COUNTS = (4, 8, 16)
MODEL_SUFFIXES = {".glb", ".gltf", ".obj", ".ply", ".stl", ".off"}


class ModelUnavailable(RuntimeError):
    """trimesh (the 3D model loader) isn't installed."""


def direction_names(n: int) -> list[str]:
    if n not in DIRECTION_COUNTS:
        raise ValueError(f"directions must be 4, 8 or 16, not {n}")
    return list(COMPASS16[:: 16 // n])


def is_model(path) -> bool:
    return Path(path).suffix.lower() in MODEL_SUFFIXES


# --------------------------------------------------------------------------- 3D model

def _trimesh():
    try:
        import trimesh
    except ImportError as e:
        raise ModelUnavailable('Rendering 3D models needs trimesh: pip install "pixel-pipeline[3d]"'
                               ' (or pip install trimesh)') from e
    return trimesh


def _texture(mesh) -> tuple[np.ndarray | None, np.ndarray]:
    """(texture RGBA array or None, base colour factor RGBA 0..1) of a textured mesh."""
    mat = getattr(mesh.visual, "material", None)
    img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    factor = getattr(mat, "baseColorFactor", None)
    factor = np.ones(4) if factor is None else np.asarray(factor, np.float64).reshape(-1)[:4]
    if factor.max() > 1.0:
        factor = factor / 255.0
    tex = np.asarray(img.convert("RGBA")) if img is not None else None
    return tex, factor


def _sample_mesh(mesh, count: int, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points spread evenly over the surface, with their colours and face normals."""
    v = np.asarray(mesh.vertices, np.float64)
    f = np.asarray(mesh.faces, np.int64)
    tri = v[f]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    if area.sum() <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.uint8), np.zeros((0, 3))
    face = rng.choice(len(f), size=count, p=area / area.sum())
    r1, r2 = rng.random(count), rng.random(count)
    s = np.sqrt(r1)
    bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
    pts = (tri[face] * bary[..., None]).sum(1)
    normals = cross[face] / np.maximum(area[face], 1e-12)[:, None]

    colours = np.full((count, 3), 180.0)
    kind = getattr(mesh.visual, "kind", None)
    if kind == "texture" and getattr(mesh.visual, "uv", None) is not None:
        tex, factor = _texture(mesh)
        uv = np.asarray(mesh.visual.uv, np.float64)[f[face]]
        uv = (uv * bary[..., None]).sum(1) % 1.0
        if tex is not None:
            th, tw = tex.shape[:2]
            x = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
            y = np.clip(((1 - uv[:, 1]) * th).astype(np.int64), 0, th - 1)
            colours = tex[y, x, :3].astype(np.float64)
        colours = colours * factor[:3]
    elif kind == "vertex":
        vc = np.asarray(mesh.visual.vertex_colors, np.float64)[:, :3]
        colours = (vc[f[face]] * bary[..., None]).sum(1)
    elif kind == "face":
        colours = np.asarray(mesh.visual.face_colors, np.float64)[face, :3]
    return pts, np.clip(colours, 0, 255).astype(np.uint8), normals


def load_model_points(path, points: int = 2_000_000, up: str = "y", seed: int = 0):
    """Sample a 3D model into coloured points, standing on y = 0, centred on the y axis,
    front facing +z (glTF convention). up: the model's up axis, 'y' (glTF) or 'z'."""
    trimesh = _trimesh()
    loaded = trimesh.load(str(path), process=False)
    meshes = [loaded] if isinstance(loaded, trimesh.Trimesh) else loaded.dump()
    meshes = [m for m in meshes if isinstance(m, trimesh.Trimesh) and len(m.faces)]
    if not meshes:
        raise ValueError(f"No triangles found in {path}")
    areas = np.array([m.area for m in meshes])
    counts = np.maximum(1, np.round(points * areas / areas.sum()).astype(int))
    rng = np.random.default_rng(seed)
    parts = [_sample_mesh(m, c, rng) for m, c in zip(meshes, counts)]
    pts = np.concatenate([p[0] for p in parts])
    cols = np.concatenate([p[1] for p in parts])
    nrm = np.concatenate([p[2] for p in parts])
    if up == "z":  # z-up -> y-up: (x, y, z) -> (x, z, -y)
        pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], 1)
        nrm = np.stack([nrm[:, 0], nrm[:, 2], -nrm[:, 1]], 1)
    elif up != "y":
        raise ValueError("up must be 'y' or 'z'")
    lo, hi = pts.min(0), pts.max(0)
    pts = pts - np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return pts, cols, nrm


def _fill_holes(rgb: np.ndarray, alpha: np.ndarray, passes: int = 2) -> None:
    """Fill single empty pixels left between splatted points (in place)."""
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    for _ in range(passes):
        filled = alpha.astype(np.float32)
        n = cv2.filter2D(filled, -1, k, borderType=cv2.BORDER_CONSTANT)
        hole = (alpha == 0) & (n >= 5)
        if not hole.any():
            break
        summed = np.stack([cv2.filter2D(rgb[..., c] * filled, -1, k, borderType=cv2.BORDER_CONSTANT)
                           for c in range(3)], -1)
        rgb[hole] = summed[hole] / n[hole][:, None]
        alpha[hole] = 1


def render_model(path, directions: int = 16, elevation: float = 30.0, size: int = 512,
                 up: str = "y", yaw: float = 0.0, light: float = 0.35,
                 points: int | None = None) -> list[np.ndarray]:
    """Render a 3D model facing every direction (in direction_names order) as RGBA images of
    size x size, all at one scale with the model's ground point in the same place.
    yaw: degrees to turn the model if its front isn't +z. light: 0 = flat texture colours,
    1 = full shading from the upper left."""
    names = direction_names(directions)
    pts, cols, nrm = load_model_points(path, points or 10 * size * size, up)
    e = math.radians(elevation)
    ce, se = math.cos(e), math.sin(e)
    radius = float(np.sqrt(pts[:, 0] ** 2 + pts[:, 2] ** 2).max())
    top = float(pts[:, 1].max()) * ce + radius * se
    bottom = -radius * se
    scale = size * 0.94 / max(2 * radius, top - bottom, 1e-9)
    ox = size / 2
    oy = (size - (top - bottom) * scale) / 2 + top * scale  # screen row of the ground point
    light_dir = np.array([-0.45, 0.6, 0.66])
    light_dir /= np.linalg.norm(light_dir)
    base = cols.astype(np.float32)
    views = []
    for k in range(len(names)):
        a = k * 360.0 / len(names)                   # facing angle on screen (Godot)
        t = math.radians(90.0 - a + yaw)            # model turn about its up axis
        ct, st = math.cos(t), math.sin(t)
        x = pts[:, 0] * ct + pts[:, 2] * st
        z = -pts[:, 0] * st + pts[:, 2] * ct
        y = pts[:, 1]
        sy = y * ce - z * se                         # up on screen
        depth = -y * se - z * ce                     # smaller = nearer the camera
        ix = np.floor(ox + x * scale).astype(np.int64)
        iy = np.floor(oy - sy * scale).astype(np.int64)
        ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
        lin = iy[ok] * size + ix[ok]
        order = np.lexsort((depth[ok], lin))
        first = np.unique(lin[order], return_index=True)[1]
        pick = np.nonzero(ok)[0][order[first]]
        rgb = np.zeros((size * size, 3), np.float32)
        alpha = np.zeros(size * size, np.uint8)
        c = base[pick]
        if light > 0:
            nx = nrm[pick, 0] * ct + nrm[pick, 2] * st
            nz = -nrm[pick, 0] * st + nrm[pick, 2] * ct
            ny = nrm[pick, 1]
            cam = np.stack([nx, ny * ce - nz * se, ny * se + nz * ce], 1)
            lam = np.abs(cam @ light_dir)            # meshes may have flipped normals
            c = c * ((1 - light) + light * lam)[:, None]
        rgb[lin[order[first]]] = c
        alpha[lin[order[first]]] = 1
        rgb, alpha = rgb.reshape(size, size, 3), alpha.reshape(size, size)
        _fill_holes(rgb, alpha)
        out = np.zeros((size, size, 4), np.uint8)
        out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        out[..., 3] = alpha * 255
        views.append(out)
    return views


def union_crop(views: list[np.ndarray], margin: float = 0.04, threshold: int = 128) -> list[np.ndarray]:
    """Crop every view to the box holding the subject in all of them (keeps one scale)."""
    mask = np.zeros(views[0].shape[:2], bool)
    for v in views:
        mask |= v[..., 3] >= threshold
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return views
    m = int(round(max(xs.max() - xs.min(), ys.max() - ys.min()) * margin))
    box = (xs.min() - m, ys.min() - m, xs.max() + 1 + m, ys.max() + 1 + m)
    return [_crop_pad(v, *box) for v in views]


# --------------------------------------------------------------------------- photos / video

def capture_count(directions: int, mirror: bool = False) -> int:
    """Views to capture: every direction, or front-to-back half when mirroring."""
    direction_names(directions)
    return directions // 2 + 1 if mirror else directions


def _same_height(images: list[np.ndarray], threshold: int) -> list[np.ndarray]:
    """Scale every (cut-out) view so the subject has the median height: walking around
    it, the camera never stays at exactly the same distance."""
    heights = []
    for im in images:
        ys = np.nonzero((im[..., 3] >= threshold).any(1))[0]
        heights.append(ys.max() - ys.min() + 1 if len(ys) else 0)
    target = float(np.median([h for h in heights if h] or [1]))
    out = []
    for im, h in zip(images, heights):
        if not h or abs(h - target) < 1:
            out.append(im)
            continue
        k = target / h
        size = (max(1, round(im.shape[1] * k)), max(1, round(im.shape[0] * k)))
        out.append(cv2.resize(im, size, interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC))
    return out


def views_from_capture(images: list[np.ndarray], directions: int = 16, reverse: bool = False,
                       mirror: bool = False, same_height: bool = True, margin: float = 0.04,
                       threshold: int = 128) -> list[np.ndarray]:
    """Cut-out photos/frames taken walking around the subject -> one view per direction
    (direction_names order), aligned on the ground at one scale."""
    want = capture_count(directions, mirror)
    if len(images) != want:
        what = "the front-to-back half" if mirror else "all the way around"
        raise ValueError(f"{directions} directions{' mirrored' if mirror else ''} need {want} "
                         f"views ({what}), got {len(images)}")
    if same_height:
        images = _same_height(images, threshold)
    aligned = align_frames(images, "track", margin, threshold)
    step = -1 if reverse else 1
    by_dir: list[np.ndarray | None] = [None] * directions
    for k, im in enumerate(aligned):        # first view faces the camera (S)
        by_dir[(directions // 4 + step * k) % directions] = im
    for j in range(directions):
        if by_dir[j] is None:              # mirrored side: E <-> W, NE <-> NW ...
            mj = round(((180.0 - j * 360.0 / directions) % 360.0) / (360.0 / directions)) % directions
            by_dir[j] = np.ascontiguousarray(by_dir[mj][:, ::-1])
    return by_dir


# --------------------------------------------------------------------------- one call

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif", ".bmp", ".tif", ".tiff"}


def natural_key(path: Path):
    """photo2 before photo10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def turnaround_from(sources: list[Path], settings: Settings, directions: int = 16, *,
                    elevation: float = 30.0, up: str = "y", yaw: float = 0.0, light: float = 0.35,
                    reverse: bool = False, mirror: bool = False, start: float | None = None,
                    end: float | None = None) -> tuple[list[Sprite], str]:
    """A 3D model, a video, or photos/folders -> (sprites in direction order, what was read)."""
    from .pipeline import isolate, load_image
    from .video import is_video, read_frames
    sources = [Path(p) for p in sources]
    first = sources[0]
    if len(sources) == 1 and is_model(first):
        views = union_crop(render_model(first, directions, elevation, up=up, yaw=yaw, light=light),
                           settings.margin, settings.alpha_threshold)
        what = f"{first.name}: 3D model rendered at {elevation:g} degrees"
    else:
        want = capture_count(directions, mirror)
        if len(sources) == 1 and is_video(first):
            images = read_frames(first, want, start, end, inclusive=mirror).frames
            what = f"{first.name}: {len(images)} video frames"
        else:
            paths: list[Path] = []
            for p in sources:
                paths += sorted((q for q in p.iterdir() if q.suffix.lower() in IMAGE_SUFFIXES),
                                key=natural_key) if p.is_dir() else [p]
            images = [load_image(q) for q in paths]
            what = f"{len(images)} photos"
        images = [isolate(im, settings) for im in images]
        views = views_from_capture(images, directions, reverse, mirror,
                                   margin=settings.margin, threshold=settings.alpha_threshold)
    return turnaround_sprites(views, settings), what


# --------------------------------------------------------------------------- sprites + files

def turnaround_sprites(views: list[np.ndarray], settings: Settings) -> list[Sprite]:
    """Views (all the same size and framing) -> sprites with one palette and one scale."""
    s = Settings(**{**settings.to_dict(), "crop_to_subject": False})
    return sprites_from_rgba(views, s, shared_palette=True)


def godot_script(names: list[str], frames_name: str) -> str:
    dirs = ", ".join(f'&"{n}"' for n in names)
    return f"""extends AnimatedSprite2D
## Turns to face a direction. Set sprite_frames to {frames_name}, then call face(velocity).
## Directions run clockwise from facing right, matching Vector2.angle().

const DIRECTIONS: Array[StringName] = [{dirs}]


func face(direction: Vector2) -> void:
\tif direction.is_zero_approx():
\t\treturn
\tvar steps := DIRECTIONS.size()
\tplay(DIRECTIONS[wrapi(roundi(direction.angle() / TAU * steps), 0, steps)])
"""


def write_turnaround(sprites: list[Sprite], out_png, aseprite: bool = False, godot: bool = False,
                     columns: int | None = None) -> list[Path]:
    """Sheet PNG + JSON (one tag per direction); optionally .aseprite (tags) and Godot
    SpriteFrames (one animation per direction) + a facing script."""
    out = Path(out_png)
    names = direction_names(len(sprites))
    frames = [s.rgba for s in sprites]
    columns = columns or min(len(frames), 8)
    out_json = write_sheet(frames, out, columns, 100, None, names)
    meta = json.loads(out_json.read_text(encoding="utf-8"))
    meta["meta"]["frameTags"] = [{"name": n, "from": i, "to": i, "direction": "forward"}
                                 for i, n in enumerate(names)]
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written = [out, out_json]
    if aseprite:
        ase = out.with_suffix(".aseprite")
        write_aseprite(ase, frames, sprites[0].palette, durations=100,
                       tags=[(n, i, i) for i, n in enumerate(names)])
        written.append(ase)
    if godot:
        _, rects = pack_sheet(frames, columns)
        anims = {n: {"frames": [i], "duration_ms": 100, "loop": False} for i, n in enumerate(names)}
        tres = write_sprite_frames(out.with_suffix(".tres"), out.name, rects, animations=anims)
        gd = out.with_name(out.stem + "_facing.gd")
        gd.write_text(godot_script(names, tres.name), encoding="utf-8")
        written += [tres, gd]
    return written


def write_preview(sprites: list[Sprite], out_gif, scale: int = 6, ms: int = 120,
                  background=(28, 32, 46)) -> Path:
    """Animated GIF that spins through every direction (clockwise), enlarged."""
    out_gif = Path(out_gif)
    frames = []
    for s in sprites:
        h, w = s.rgba.shape[:2]
        im = Image.fromarray(s.rgba, "RGBA").resize((w * scale, h * scale), Image.NEAREST)
        bg = Image.new("RGBA", im.size, background + (255,))
        bg.alpha_composite(im)
        frames.append(bg.convert("RGB"))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], duration=ms, loop=0)
    return out_gif
