"""Prompt or picture -> coloured 3D model (.glb) on your own GPU, through ComfyUI.

picture  generate_images() with a prompt shaped for 3D (whole subject, facing the viewer,
         plain white background, even light) and no pixel-art LoRA: a clean picture
         rebuilds into a better model than pixel art does.
shape    ComfyUI's built-in Hunyuan3D 2 nodes, wired like ComfyUI's own "Image to Model
         (Hunyuan3d 2.1)" template: ImageOnlyCheckpointLoader -> CLIPVisionEncode ->
         Hunyuan3Dv2Conditioning -> EmptyLatentHunyuan3Dv2 -> ModelSamplingAuraFlow ->
         KSampler -> VAEDecodeHunyuan3D -> VoxelToMesh -> SaveGLB. Shape only, about 10 GB
         of VRAM, so it fits a 12 GB card.
colour   paint_from_picture(): the picture is projected onto the parts of the model it
         shows. Parts it can't see (the back) take the colours of a blurred copy of the
         picture straight through the model, so a red cape stays red behind, unless a
         back-view picture is given.

Any API-format workflow with an {{image}} input that ends in a SaveGLB node can replace
the shape step (`template`), e.g. ComfyUI's TRELLIS.2 nodes on a bigger GPU; a model that
comes back with its own colours is kept as it is.
"""
from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .generate import (ComfyClient, ComfyUIModelMissing, ComfyUIWorkflowError, fill_template,
                       generate_images, load_template, template_placeholders)
from .pipeline import ImageSource, load_image
from .turnaround import _sample_mesh, _trimesh

log = logging.getLogger(__name__)

SHAPE_CHECKPOINTS = ("hunyuan_3d_v2.1.safetensors", "hunyuan3d-dit-v2-0.safetensors",
                     "hunyuan3d-dit-v2-mini.safetensors", "hunyuan3d-dit-v2-mv.safetensors")
MODEL_PROMPTS = {
    "person": "{prompt}, single character, full body from head to feet, standing, facing the "
              "viewer, front view, arms slightly away from the body, centered, plain white "
              "background, soft even lighting, no shadows, clean shapes, game character concept art",
    "animal": "{prompt}, single animal, whole body in frame, three-quarter front view, standing, "
              "centered, plain white background, soft even lighting, no shadows, clean shapes, "
              "game creature concept art",
    "object": "{prompt}, single object, whole object in frame, three-quarter front view, "
              "centered, plain white background, soft even lighting, no shadows, clean shapes, "
              "game prop concept art",
}
MODEL_NEGATIVE = ("cropped, cut off, out of frame, partial, multiple subjects, crowd, text, "
                  "watermark, busy background, scenery, floor, dramatic lighting, harsh shadows, "
                  "motion blur, blurry, pixel art, sprite sheet")


@dataclass
class ShapeOptions:
    """Hunyuan3D settings (ComfyUI's template values)."""
    checkpoint: str | None = None   # None = first installed Hunyuan3D checkpoint
    steps: int = 30
    cfg: float = 5.0
    octree: int = 256               # mesh detail: 256 (default), 384/512 = finer, slower
    threshold: float = 0.6
    template: str | None = None     # your own workflow: {{image}} in, SaveGLB out
    timeout: float = 1800.0


@dataclass
class ModelResult:
    model: Path            # coloured .glb (front +z, up +y)
    picture: Path          # the front picture it was built from
    raw: Path | None       # the uncoloured shape ComfyUI returned
    seed: int | None


# --------------------------------------------------------------------------- picture

def model_prompt(prompt: str, subject: str = "person") -> tuple[str, str]:
    prompt = (prompt or "").strip().rstrip(".")
    if not prompt:
        raise ValueError("Empty prompt: describe what to make.")
    return MODEL_PROMPTS.get(subject, MODEL_PROMPTS["object"]).format(prompt=prompt), MODEL_NEGATIVE


def picture_for_model(prompt: str, subject: str = "person", *, client: ComfyClient | None = None,
                      seed: int | None = None, **gen_opts) -> tuple[Image.Image, int]:
    """A clean front picture to build the model from (SDXL, no pixel-art LoRA)."""
    positive, negative = model_prompt(prompt, subject)
    opts = {"lora": "none", **gen_opts}
    (img, used), = generate_images(positive, negative, client=client, seed=seed, aspect=1.0, **opts)
    return img, used


def cut_out(picture: ImageSource, cutout_model: str = "isnet-general-use") -> np.ndarray:
    """RGBA with the background removed (kept if the picture is already transparent)."""
    from .pipeline import remove_background
    rgba = load_image(picture)
    if (rgba[..., 3] < 255).any():
        return rgba
    out = rgba.copy()
    out[..., 3] = remove_background(np.ascontiguousarray(rgba[..., :3]), cutout_model)
    return out


def framed_on_white(rgba: np.ndarray, size: int = 1024, fill: float = 0.85) -> Image.Image:
    """Subject centred on a white square, filling `fill` of it: what Hunyuan3D expects."""
    ys, xs = np.nonzero(rgba[..., 3] >= 128)
    if not len(xs):
        raise ValueError("The picture is empty after background removal.")
    crop = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape[:2]
    k = size * fill / max(h, w)
    crop = cv2.resize(crop, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    a = crop[..., 3:4].astype(np.float32) / 255
    rgb = crop[..., :3].astype(np.float32) * a + 255 * (1 - a)
    canvas = np.full((size, size, 3), 255, np.uint8)
    y0, x0 = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    canvas[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = np.rint(rgb).astype(np.uint8)
    return Image.fromarray(canvas)


# --------------------------------------------------------------------------- shape

def hunyuan3d_workflow(image: str, *, seed: int, checkpoint: str, steps: int = 30,
                       cfg: float = 5.0, octree: int = 256, threshold: float = 0.6,
                       prefix: str = "pixel-pipeline/model") -> dict:
    """API-format workflow: uploaded picture -> Hunyuan3D 2 shape -> SaveGLB."""
    wf: dict[str, dict] = {}

    def node(nid: str, cls: str, title: str, **inputs) -> None:
        wf[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}

    multiview = "-mv" in checkpoint.lower()
    v21 = "2.1" in checkpoint or "2_1" in checkpoint
    node("1", "ImageOnlyCheckpointLoader", "Load Hunyuan3D", ckpt_name=checkpoint)
    node("2", "LoadImage", "Front picture", image=image)
    node("3", "CLIPVisionEncode", "Encode picture", clip_vision=["1", 1], image=["2", 0], crop="center")
    if multiview:
        node("4", "Hunyuan3Dv2ConditioningMultiView", "Condition (front view)", front=["3", 0])
    else:
        node("4", "Hunyuan3Dv2Conditioning", "Condition", clip_vision_output=["3", 0])
    node("5", "EmptyLatentHunyuan3Dv2", "Empty 3D latent", resolution=4096 if v21 else 3072,
         batch_size=1)
    node("6", "ModelSamplingAuraFlow", "Sampling shift", model=["1", 0], shift=1.0)
    node("7", "KSampler", "Sampler", model=["6", 0], seed=seed, steps=steps, cfg=cfg,
         sampler_name="euler", scheduler="normal", positive=["4", 0], negative=["4", 1],
         latent_image=["5", 0], denoise=1.0)
    node("8", "VAEDecodeHunyuan3D", "Decode to voxels", samples=["7", 0], vae=["1", 2],
         num_chunks=8000, octree_resolution=octree)
    node("9", "VoxelToMesh", "Voxels to mesh", voxel=["8", 0], algorithm="surface net",
         threshold=threshold)
    node("10", "SaveGLB", "Save model", mesh=["9", 0], filename_prefix=prefix)
    return wf


def _shape_checkpoint(client: ComfyClient, wanted: str | None) -> str:
    installed = client.list_models("checkpoints")
    if wanted:
        if wanted in installed:
            return wanted
        raise ComfyUIModelMissing(
            f"Checkpoint '{wanted}' isn't installed in ComfyUI/models/checkpoints.", "checkpoints",
            wanted, installed)
    for name in SHAPE_CHECKPOINTS:
        if name in installed:
            return name
    for name in installed:
        low = name.lower().replace("_", "").replace("-", "")
        if "hunyuan3d" in low:
            return name
    raise ComfyUIModelMissing(
        "No Hunyuan3D checkpoint in ComfyUI/models/checkpoints. Download "
        "hunyuan_3d_v2.1.safetensors (see docs/LOCAL_AI.md, 'Prompt to 3D model') and restart "
        "ComfyUI.", "checkpoints", SHAPE_CHECKPOINTS[0], installed)


def shape_from_picture(picture: Image.Image, *, client: ComfyClient | None = None,
                       seed: int | None = None, options: ShapeOptions | None = None) -> bytes:
    """Front picture (subject on white) -> 3D model file bytes (.glb) from ComfyUI."""
    o = options or ShapeOptions()
    client = client or ComfyClient()
    client.ping()
    name = client.upload_image(picture)
    seed = random.randrange(2**32) if seed is None else int(seed)
    if o.template:
        wf = load_template(o.template)
        if "image" not in template_placeholders(wf):
            raise ValueError("The 3D workflow template needs an {{image}} placeholder (LoadImage's "
                             "image) for the picture.")
        wf = fill_template(wf, {"image": name, "seed": seed, "steps": o.steps, "cfg": o.cfg,
                                "prefix": "pixel-pipeline/model"})
    else:
        wf = hunyuan3d_workflow(name, seed=seed, checkpoint=_shape_checkpoint(client, o.checkpoint),
                                steps=o.steps, cfg=o.cfg, octree=o.octree, threshold=o.threshold)
    pid = client.queue(wf)
    log.info("Queued 3D shape %s (seed %d)", pid, seed)
    entry = client.wait(pid, o.timeout)
    files = client.fetch_files(pid, entry)
    if not files:
        raise ComfyUIWorkflowError("ComfyUI finished but saved no 3D model. The workflow needs a "
                                   "SaveGLB node.")
    return files[0][1]


# --------------------------------------------------------------------------- colour

# Candidate orientations, most likely first: (front axis, up axis) of the model file.
_AXES = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
         "+z": (0, 0, 1), "-z": (0, 0, -1)}
_ORDER = [("+z", "+y"), ("-z", "+y"), ("+x", "+y"), ("-x", "+y"), ("-y", "+z"), ("+y", "+z"),
          ("+x", "+z"), ("-x", "+z")]


def _orientations():
    seen = list(_ORDER)
    for f in _AXES:
        for u in _AXES:
            if (f, u) not in seen and abs(np.dot(_AXES[f], _AXES[u])) < 0.5:
                seen.append((f, u))
    for f, u in seen:
        fv, uv = np.array(_AXES[f], float), np.array(_AXES[u], float)
        yield (f, u), np.stack([np.cross(uv, fv), uv, fv])   # rows: right, up, front


def _silhouette(xy: np.ndarray, size: int = 96) -> np.ndarray:
    """Points (x right, y up) -> filled mask, scaled by height, bottom-centred."""
    lo, hi = xy.min(0), xy.max(0)
    h = max(hi[1] - lo[1], 1e-9)
    k = (size - 4) / h
    cx = (lo[0] + hi[0]) / 2
    col = np.round(size / 2 + (xy[:, 0] - cx) * k).astype(int)
    row = np.round(size - 2 - (xy[:, 1] - lo[1]) * k).astype(int)
    ok = (col >= 0) & (col < size) & (row >= 0) & (row < size)
    mask = np.zeros((size, size), np.uint8)
    mask[row[ok], col[ok]] = 1
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def _picture_silhouette(alpha: np.ndarray, size: int = 96) -> np.ndarray:
    ys, xs = np.nonzero(alpha >= 128)
    pts = np.stack([xs, -ys], 1).astype(float)
    step = max(1, len(pts) // 60000)
    return _silhouette(pts[::step], size)


def orient_to_picture(points: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotation (3x3) that turns the model so the picture is its front view (front +z,
    up +y), found by matching silhouettes; and the match (IoU, 0-1). Ties go to the usual
    file conventions (glTF first)."""
    target = _picture_silhouette(alpha)
    best, best_iou = None, -1.0
    for _, rot in _orientations():
        p = points @ rot.T
        sil = _silhouette(p[:, :2])
        inter = np.logical_and(sil, target).sum()
        iou = inter / max(1, np.logical_or(sil, target).sum())
        if iou > best_iou + 0.01:
            best, best_iou = rot, iou
    return best, float(best_iou)


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0, y0 = u.astype(int), v.astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


class _View:
    """A picture of the model from the front (+z) or back, fitted to the model's silhouette."""

    def __init__(self, rgba: np.ndarray, samples: np.ndarray, back: bool = False,
                 max_side: int = 768):
        h, w = rgba.shape[:2]
        k = min(1.0, max_side / max(h, w))
        if k < 1:
            rgba = cv2.resize(rgba, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
        if back:  # seen from behind, the model's +x is on the picture's left
            rgba = np.ascontiguousarray(rgba[:, ::-1])
        self.rgba = rgba
        self.sign = -1.0 if back else 1.0
        alpha = rgba[..., 3]
        ys, xs = np.nonzero(alpha >= 128)
        if not len(xs):
            raise ValueError("The picture is empty after background removal.")
        lo, hi = samples.min(0), samples.max(0)
        self.scale = (ys.max() - ys.min() + 1) / max(hi[1] - lo[1], 1e-9)
        self.px0 = (xs.min() + xs.max() + 1) / 2 - self.scale * (lo[0] + hi[0]) / 2
        self.py0 = ys.max() + 1 + self.scale * lo[1]
        a = alpha.astype(np.float32) / 255
        sigma = max(1.0, 0.03 * (ys.max() - ys.min()))
        pre = cv2.GaussianBlur(rgba[..., :3].astype(np.float32) * a[..., None], (0, 0), sigma)
        wa = cv2.GaussianBlur(a, (0, 0), sigma)
        filled = pre / np.maximum(wa, 1e-4)[..., None]
        # spread colour beyond the silhouette so points mapped just outside it stay coloured
        inside = (wa > 0.05).astype(np.uint8)
        for _ in range(6):
            grown = cv2.dilate(filled * inside[..., None], np.ones((5, 5), np.uint8))
            cnt = cv2.dilate(inside, np.ones((5, 5), np.uint8))
            filled = np.where(inside[..., None] > 0, filled, grown)
            inside = cnt
        self.blurred = filled
        # depth buffer from the dense samples: nearest = largest sign * z
        u, v = self.project(samples)
        depth = self.sign * samples[:, 2]
        hh, ww = alpha.shape
        ui, vi = np.clip(np.round(u).astype(int), 0, ww - 1), np.clip(np.round(v).astype(int), 0, hh - 1)
        zbuf = np.full(hh * ww, -np.inf)
        np.maximum.at(zbuf, vi * ww + ui, depth)
        zbuf = zbuf.reshape(hh, ww)
        self.zbuf = cv2.dilate(np.where(np.isfinite(zbuf), zbuf, -1e9).astype(np.float32),
                               np.ones((3, 3), np.uint8))
        self.eps = 0.02 * float(samples[:, 2].max() - samples[:, 2].min() + 1e-9)

    def project(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # the back picture is mirrored on load, so +x is to the right in both views
        return self.px0 + self.scale * p[:, 0], self.py0 - self.scale * p[:, 1]

    def visible(self, verts: np.ndarray, normals: np.ndarray) -> np.ndarray:
        u, v = self.project(verts)
        hh, ww = self.zbuf.shape
        ui, vi = np.clip(np.round(u).astype(int), 0, ww - 1), np.clip(np.round(v).astype(int), 0, hh - 1)
        facing = self.sign * normals[:, 2] > 0.15
        alpha = self.rgba[vi, ui, 3] >= 128
        return facing & alpha & (self.sign * verts[:, 2] >= self.zbuf[vi, ui] - self.eps)

    def colours(self, verts: np.ndarray, sharp: bool) -> np.ndarray:
        u, v = self.project(verts)
        img = self.rgba[..., :3].astype(np.float32) if sharp else self.blurred
        return _bilinear(img, u, v)


def paint_from_picture(mesh, picture: np.ndarray, back: np.ndarray | None = None,
                       samples: int = 400_000, min_match: float = 0.5):
    """Colour a shape-only mesh from its front picture (RGBA, background removed).

    Returns (trimesh with vertex colours, turned to front +z / up +y; silhouette match 0-1).
    `back`: optional back-view picture (RGBA) for the parts the front can't see."""
    trimesh = _trimesh()
    if not isinstance(mesh, trimesh.Trimesh):
        loaded = trimesh.load(io.BytesIO(mesh), file_type="glb") if isinstance(mesh, bytes) \
            else trimesh.load(str(mesh))
        mesh = loaded if isinstance(loaded, trimesh.Trimesh) else loaded.to_geometry()
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces), process=True)
    mesh = _dense(mesh)
    rng = np.random.default_rng(0)
    pts, _, _ = _sample_mesh(mesh, samples, rng)
    rot, match = orient_to_picture(pts, picture[..., 3])
    if match < min_match:
        log.warning("The model's outline only matches the picture %.0f%%; colours may be off.",
                    100 * match)
    verts = np.asarray(mesh.vertices) @ rot.T
    pts = pts @ rot.T
    centre = np.array([(pts[:, 0].min() + pts[:, 0].max()) / 2, pts[:, 1].min(),
                       (pts[:, 2].min() + pts[:, 2].max()) / 2])
    verts, pts = verts - centre, pts - centre
    out = trimesh.Trimesh(verts, np.asarray(mesh.faces), process=False)
    normals = np.asarray(out.vertex_normals)

    front = _View(picture, pts)
    colours = front.colours(verts, sharp=False)               # guess for every vertex
    seen = front.visible(verts, normals)
    colours[seen] = front.colours(verts[seen], sharp=True)
    if back is not None:
        rear = _View(back, pts, back=True)
        rear_seen = rear.visible(verts, normals) & ~seen
        colours[rear_seen] = rear.colours(verts[rear_seen], sharp=True)
        seen |= rear_seen
    colours = _smooth_hidden(out, colours, seen)
    rgba = np.concatenate([np.clip(colours, 0, 255), np.full((len(verts), 1), 255.0)], 1)
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.rint(rgba).astype(np.uint8))
    return out, match


def _dense(mesh, detail: int = 120, max_faces: int = 3_000_000):
    """Colours live on vertices, so split long edges: no edge longer than 1/detail of the
    model's size (image-to-3D meshes are already this fine; hand-made ones often aren't)."""
    trimesh = _trimesh()
    size = float(np.ptp(np.asarray(mesh.vertices), axis=0).max())
    edges = np.asarray(mesh.edges_unique_length)
    if not len(edges) or edges.max() <= size / detail:
        return mesh
    v, f = trimesh.remesh.subdivide_to_size(np.asarray(mesh.vertices), np.asarray(mesh.faces),
                                            max_edge=size / detail, max_iter=10)
    if len(f) > max_faces:
        log.warning("Model has %d faces after subdividing; colours may be blurry.", len(f))
    return trimesh.Trimesh(v, f, process=True)


def _smooth_hidden(mesh, colours: np.ndarray, fixed: np.ndarray, rounds: int = 4) -> np.ndarray:
    """Blend guessed colours into their neighbours so the seam with the painted side is soft."""
    edges = np.asarray(mesh.edges_unique)
    if not len(edges):
        return colours
    a, b = edges[:, 0], edges[:, 1]
    out = colours.astype(np.float64)
    for _ in range(rounds):
        acc = np.zeros_like(out)
        cnt = np.zeros(len(out))
        np.add.at(acc, a, out[b])
        np.add.at(acc, b, out[a])
        np.add.at(cnt, a, 1)
        np.add.at(cnt, b, 1)
        avg = acc / np.maximum(cnt, 1)[:, None]
        free = ~fixed & (cnt > 0)
        out[free] = 0.5 * out[free] + 0.5 * avg[free]
    return out


def has_colours(glb: bytes) -> bool:
    """Did the model come with its own colours (texture or vertex colours)?"""
    trimesh = _trimesh()
    loaded = trimesh.load(io.BytesIO(glb), file_type="glb")
    meshes = [loaded] if isinstance(loaded, trimesh.Trimesh) else list(loaded.geometry.values())
    for m in meshes:
        kind = getattr(m.visual, "kind", None)
        if kind == "texture":
            return True
        if kind in ("vertex", "face"):
            cols = np.asarray(m.visual.vertex_colors if kind == "vertex" else m.visual.face_colors)
            if len(cols) and np.ptp(cols[:, :3], axis=0).max() > 8:
                return True
    return False


# --------------------------------------------------------------------------- all together

def prompt_to_model(prompt: str | None, out_glb, *, subject: str = "person",
                    picture: ImageSource | None = None, back: ImageSource | None = None,
                    shape: bytes | str | Path | None = None, client: ComfyClient | None = None,
                    seed: int | None = None, shape_options: ShapeOptions | None = None,
                    cutout_model: str = "isnet-general-use", progress=None,
                    **gen_opts) -> ModelResult:
    """Prompt (or a picture, or a picture + a ready shape) -> coloured .glb.

    Writes <out>.glb (coloured), <out>.front.png (the picture) and <out>.shape.glb (the
    shape as ComfyUI made it). progress(step, text) is called as each step starts."""
    out = Path(out_glb)
    out.parent.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda step, text: log.info("%s: %s", step, text))
    used_seed = seed
    if picture is None:
        if not prompt:
            raise ValueError("Give a prompt or a picture.")
        say("picture", f"Making a front picture of “{prompt}” on your GPU…")
        img, used_seed = picture_for_model(prompt, subject, client=client, seed=seed, **gen_opts)
        picture = img
    say("cutout", "Cutting out the subject…")
    rgba = cut_out(picture, cutout_model)
    pic_path = out.with_suffix(".front.png")
    Image.fromarray(rgba).save(pic_path)
    back_rgba = cut_out(back, cutout_model) if back is not None else None

    raw_path = None
    if shape is None:
        say("shape", "Building the 3D shape on your GPU (Hunyuan3D, about a minute)…")
        glb = shape_from_picture(framed_on_white(rgba), client=client, seed=seed,
                                 options=shape_options)
        raw_path = out.with_suffix(".shape.glb")
        raw_path.write_bytes(glb)
    else:
        glb = shape if isinstance(shape, bytes) else Path(shape).read_bytes()

    if has_colours(glb):
        say("colour", "The model came with its own colours; keeping them.")
        out.write_bytes(glb)
    else:
        say("colour", "Painting the model from the picture…")
        mesh, match = paint_from_picture(glb, rgba, back_rgba)
        if match < 0.5:
            say("colour", f"Warning: the shape only matches the picture {match:.0%}.")
        mesh.export(out)
    return ModelResult(out, pic_path, raw_path, used_seed)
