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
views    views_for_model(): real back (and left / right side) pictures to paint from. The
         shape is rendered from behind (or the side) with the camera found for the front
         picture, shaded grey and tinted with the front's colours, and SDXL redraws that as
         "back view of <prompt>, same outfit, same colours" (img2img, same seed, no pixel
         LoRA). The new picture keeps the shape's outline, so it lands on the model exactly.

Any API-format workflow with an {{image}} input that ends in a SaveGLB node can replace
the shape step (`template`), e.g. ComfyUI's TRELLIS.2 nodes on a bigger GPU; a model that
comes back with its own colours is kept as it is.
"""
from __future__ import annotations

import io
import logging
import random
from dataclasses import dataclass, field
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
    "person": "{prompt}, single character, full body from head to feet, standing in an A-pose, "
              "facing the viewer, front view, arms held down and out away from the body, legs "
              "slightly apart, centered, plain white "
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
    views: dict = field(default_factory=dict)   # back / left / right -> picture made for it


VIEWS = ("front", "front+back", "all")
VIEW_TURN = {"back": 180.0, "left": -90.0, "right": 90.0}   # camera walks round (degrees)
VIEW_PROMPTS = {
    "back": "back view of {prompt}, seen from behind, the same {what} from the back, same "
            "outfit, same colours, same materials, whole body in frame, plain white background, "
            "soft even lighting, no shadows, clean shapes, game {kind} concept art",
    "left": "side view of {prompt}, profile facing left, the same {what} seen from its left "
            "side, same outfit, same colours, same materials, whole body in frame, plain white "
            "background, soft even lighting, no shadows, clean shapes, game {kind} concept art",
    "right": "side view of {prompt}, profile facing right, the same {what} seen from its "
             "right side, same outfit, same colours, same materials, whole body in frame, plain "
             "white background, soft even lighting, no shadows, clean shapes, game {kind} "
             "concept art",
}
VIEW_NEGATIVE = {"back": "front view, face, looking at the viewer, ",
                 "left": "front view, back view, facing the viewer, ",
                 "right": "front view, back view, facing the viewer, "}


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
    client.ensure_running()
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


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.logical_and(a, b).sum() / max(1, np.logical_or(a, b).sum()))


def _ry(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rx(deg: float) -> np.ndarray:
    """Tilt for a camera `deg` above the horizon: the top of the model leans toward it."""
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def orient_to_picture(points: np.ndarray, alpha: np.ndarray, upright: bool = True
                      ) -> tuple[np.ndarray, float]:
    """Rotation (3x3) that turns the model so the picture is its front view (front +z,
    up +y), found by matching silhouettes; and the match (IoU, 0-1).

    upright (models from an image-to-3D AI: y is up): only turns about the vertical axis, in
    5 degree steps, so a three-quarter picture is matched at its real angle and the model is
    never tipped over (a wolf that matched better standing on end). Use
    orient_to_picture_view for the camera height too. upright=False: the 24 axis-aligned
    orientations of an arbitrary file (ties go to the usual conventions, glTF first)."""
    if upright:
        rot, _, iou = orient_to_picture_view(points, alpha)
        return rot, iou
    target = _picture_silhouette(alpha)
    best, best_iou = None, -1.0
    for _, rot in _orientations():
        p = points @ rot.T
        iou = _iou(_silhouette(p[:, :2]), target)
        if iou > best_iou + 0.01:
            best, best_iou = rot, iou
    return best, float(best_iou)


def orient_to_picture_view(points: np.ndarray, alpha: np.ndarray,
                           elevations=(0, 10, 20, 30, 40)) -> tuple[np.ndarray, np.ndarray, float]:
    """For an upright model: (turn about y, camera tilt, IoU) that make the model's outline
    match the picture: the picture is the model seen from the front (+z) by a camera
    `tilt` degrees above it (photos are often taken from above, e.g. a dog on a leash)."""
    target = _picture_silhouette(alpha)
    step = max(1, len(points) // 40000)
    pts = points[::step]

    def score(yaw, elev):
        p = pts @ (_rx(elev) @ _ry(yaw)).T
        return _iou(_silhouette(p[:, :2]), target)
    best = max(((score(y, e), y, e) for e in elevations for y in range(0, 360, 5)))
    _, y0, e0 = best
    best = max([best] + [(score(y0 + dy, e0 + de), y0 + dy, e0 + de)
                         for dy in (-4, -2, 2, 4) for de in (-5, -2.5, 0, 2.5, 5)
                         if 0 <= e0 + de <= 60])
    iou, yaw, elev = best
    return _ry(yaw), _rx(elev), float(iou)


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
        self._refine_fit(samples, alpha >= 128)
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
        # everything further out takes the nearest subject colour (was black: when the shape
        # doesn't match the picture well, e.g. a turned head, whole parts land out here)
        outside = (inside == 0).astype(np.uint8)
        if outside.any() and not outside.all():
            _, labels = cv2.distanceTransformWithLabels(outside, cv2.DIST_L2, 5,
                                                        labelType=cv2.DIST_LABEL_PIXEL)
            iy, ix = np.nonzero(outside == 0)
            lut = np.zeros((int(labels.max()) + 1, 3), np.float32)
            lut[labels[iy, ix]] = filled[iy, ix]
            filled = np.where(outside[..., None] > 0, lut[labels], filled)
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

    def _refine_fit(self, samples: np.ndarray, inside: np.ndarray, work: int = 160):
        """Nudge scale and position so the model's outline lies on the picture's (bounding
        boxes alone are off when, say, a tail or a far leg sticks out in one of them)."""
        h, w = inside.shape
        k = work / max(h, w)
        small = cv2.resize(inside.astype(np.uint8), (max(1, round(w * k)), max(1, round(h * k))),
                           interpolation=cv2.INTER_AREA) > 0
        sh, sw = small.shape
        pts = samples[::max(1, len(samples) // 30000)]
        cx, cy = (pts[:, 0].min() + pts[:, 0].max()) / 2, (pts[:, 1].min() + pts[:, 1].max()) / 2

        def mask(scale, px0, py0):
            u = np.round((px0 + scale * pts[:, 0]) * k).astype(int)
            v = np.round((py0 - scale * pts[:, 1]) * k).astype(int)
            ok = (u >= 0) & (u < sw) & (v >= 0) & (v < sh)
            m = np.zeros((sh, sw), np.uint8)
            m[v[ok], u[ok]] = 1
            return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        best = (_iou(mask(self.scale, self.px0, self.py0), small), self.scale, self.px0, self.py0)
        base = best
        for f in np.linspace(0.9, 1.1, 9):
            sc = base[1] * f
            # keep the model's centre where it was, then shift
            ox = base[2] + base[1] * cx - sc * cx
            oy = base[3] - base[1] * cy + sc * cy
            for dx in range(-8, 9, 2):
                for dy in range(-8, 9, 2):
                    cand = (sc, ox + dx / k, oy + dy / k)
                    iou = _iou(mask(*cand), small)
                    if iou > best[0] + 1e-4:
                        best = (iou, *cand)
        self.match, self.scale, self.px0, self.py0 = best

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
                       samples: int = 400_000, min_match: float = 0.5, upright: bool = True,
                       left: np.ndarray | None = None, right: np.ndarray | None = None):
    """Colour a shape-only mesh from its front picture (RGBA, background removed).

    Returns (trimesh with vertex colours, turned to front +z / up +y; silhouette match 0-1).
    `back`, `left`, `right`: optional pictures from behind and from the character's left and
    right side (the same camera walked round 180 / 90 degrees; views_for_model makes them)
    for the parts the front can't see. Each part takes the picture that faces it best (the
    front picture first: it is the one the shape was made from)."""
    out, match, _ = _paint(mesh, picture, {"back": back, "left": left, "right": right},
                           samples, min_match, upright)
    return out, match


def _load_shape(mesh):
    trimesh = _trimesh()
    if not isinstance(mesh, trimesh.Trimesh):
        loaded = trimesh.load(io.BytesIO(mesh), file_type="glb") if isinstance(mesh, bytes) \
            else trimesh.load(str(mesh))
        mesh = loaded if isinstance(loaded, trimesh.Trimesh) else loaded.to_geometry()
    mesh = trimesh.Trimesh(np.asarray(mesh.vertices, float), np.asarray(mesh.faces), process=True)
    return _dense(mesh)


# view weights: a part seen square-on in two pictures takes the front one first
_VIEW_WEIGHT = {"front": 1.0, "back": 0.9, "left": 0.8, "right": 0.8}


def _paint(mesh, picture: np.ndarray, others: dict, samples: int = 400_000,
           min_match: float = 0.5, upright: bool = True):
    """paint_from_picture + the camera tilt it found: (mesh, match, tilt)."""
    trimesh = _trimesh()
    mesh = _load_shape(mesh)
    rng = np.random.default_rng(0)
    pts, _, _ = _sample_mesh(mesh, samples, rng)
    if upright:
        rot, tilt, match = orient_to_picture_view(pts, picture[..., 3])
    else:
        (rot, match), tilt = orient_to_picture(pts, picture[..., 3], upright=False), np.eye(3)
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

    # the picture is the model seen from the front by a camera `tilt` above it
    cv, cp, cn = verts @ tilt.T, pts @ tilt.T, normals @ tilt.T
    front = _View(picture, cp)
    colours = front.colours(cv, sharp=False)                  # guess for every vertex
    seen = front.visible(cv, cn)
    colours[seen] = front.colours(cv[seen], sharp=True)       # what the picture shows, sharp
    best = np.where(seen, cn[:, 2] * _VIEW_WEIGHT["front"], -np.inf)
    owner = np.where(seen, 0, -1)
    views = [(name, pic) for name, pic in others.items() if pic is not None]
    for k, (name, pic) in enumerate(views, 1):
        # the same camera after walking round the model
        turn = tilt @ _ry(VIEW_TURN[name])
        vv, vp, vn = verts @ turn.T, pts @ turn.T, normals @ turn.T
        view = _View(pic, vp)
        vis = view.visible(vv, vn)
        score = np.where(vis, vn[:, 2] * _VIEW_WEIGHT[name], -np.inf)
        win = score > best
        if win.any():
            colours[win] = view.colours(vv[win], sharp=True)
        best, owner = np.maximum(best, score), np.where(win, k, owner)
    seen = owner >= 0
    match = max(match, front.match)
    colours = _smooth_hidden(out, colours, seen)
    rgba = np.concatenate([np.clip(colours, 0, 255), np.full((len(verts), 1), 255.0)], 1)
    out.visual = trimesh.visual.ColorVisuals(out, vertex_colors=np.rint(rgba).astype(np.uint8))
    return out, match, tilt


# --------------------------------------------------------------------------- back + side views

def _render_view(pts: np.ndarray, cols: np.ndarray, nrm: np.ndarray, size: int = 768,
                 light: float = 0.45) -> np.ndarray:
    """Orthographic render of coloured points seen from +z (x right, y up): RGBA, the subject
    filling 90% of the height, grey shading from the upper left mixed into the colours."""
    from .turnaround import _fill_holes
    lo, hi = pts.min(0), pts.max(0)
    k = 0.9 * size / max(float(hi[1] - lo[1]), float(hi[0] - lo[0]), 1e-9)
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    ix = np.floor(size / 2 + (pts[:, 0] - cx) * k).astype(np.int64)
    iy = np.floor(size / 2 - (pts[:, 1] - cy) * k).astype(np.int64)
    ok = (ix >= 0) & (ix < size) & (iy >= 0) & (iy < size)
    lin = iy[ok] * size + ix[ok]
    depth = -pts[ok, 2]                                    # smaller = nearer the camera
    zbuf = np.full(size * size, np.inf)
    np.minimum.at(zbuf, lin, depth)
    near = depth <= zbuf[lin]
    pick, lin = np.nonzero(ok)[0][near], lin[near]
    ldir = np.array([-0.4, 0.55, 0.73])
    ldir /= np.linalg.norm(ldir)
    lam = np.abs(nrm[pick] @ ldir)
    rgb = np.zeros((size * size, 3), np.float32)
    alpha = np.zeros(size * size, np.uint8)
    rgb[lin] = cols[pick].astype(np.float32) * ((1 - light) + light * lam)[:, None]
    alpha[lin] = 1
    rgb, alpha = rgb.reshape(size, size, 3), alpha.reshape(size, size)
    _fill_holes(rgb, alpha, passes=4)
    closed = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    if (closed > alpha).any():                             # gaps the hole fill missed
        grown = cv2.dilate(rgb * alpha[..., None], np.ones((5, 5), np.uint8))
        cnt = cv2.dilate(alpha.astype(np.float32), np.ones((5, 5), np.uint8))
        add = (closed > 0) & (alpha == 0)
        rgb[add] = grown[add] / np.maximum(cnt[add], 1)[:, None]
        alpha = closed
    rgb = cv2.GaussianBlur(rgb, (0, 0), 0.8)
    out = np.zeros((size, size, 4), np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = alpha * 255
    return out


def _keep_outline(drawn: np.ndarray, init: np.ndarray) -> np.ndarray:
    """The AI's colours inside the shape's own outline (init alpha): where the AI drew
    background inside it, the nearest colour it drew; if it drew nothing usable, the init."""
    inside = init[..., 3] >= 128
    mine = drawn[..., 3] >= 128
    out = init.copy()
    if (mine & inside).sum() < 0.3 * max(1, inside.sum()):
        return out
    rgb = drawn[..., :3].copy()
    lost = inside & ~mine
    if lost.any():
        _, near = cv2.distanceTransformWithLabels((~mine).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        ky, kx = np.nonzero(mine)
        lut = np.zeros((int(near.max()) + 1, 3), np.uint8)
        lut[near[ky, kx]] = rgb[ky, kx]
        rgb[lost] = lut[near[lost]]
    out[..., :3] = np.where(inside[..., None], rgb, 0)
    out[..., 3] = np.where(inside, 255, 0)
    return out


def view_inits(shape, front_picture: np.ndarray, which=("back",), samples: int = 400_000,
               size: int = 768) -> dict[str, np.ndarray]:
    """The starting pictures for views_for_model: the shape seen from behind / the side by
    the front picture's camera walked round, grey-shaded and tinted with the colours the
    front picture gives it (RGBA, subject centred)."""
    from .turnaround import _sample_mesh
    painted, _, tilt = _paint(shape, front_picture, {}, samples)
    pts, cols, nrm = _sample_mesh(painted, samples, np.random.default_rng(1))
    out = {}
    for name in which:
        turn = tilt @ _ry(VIEW_TURN[name])
        out[name] = _render_view(pts @ turn.T, cols, nrm @ turn.T, size)
    return out


def view_prompt(prompt: str, view: str, subject: str = "person") -> tuple[str, str]:
    prompt = (prompt or "").strip().rstrip(".") or {"person": "a character", "animal": "an animal"}.get(
        subject, "an object")
    what, kind = {"person": ("character", "character"), "animal": ("animal", "creature")}.get(
        subject, ("object", "prop"))
    return (VIEW_PROMPTS[view].format(prompt=prompt, what=what, kind=kind),
            VIEW_NEGATIVE[view] + MODEL_NEGATIVE)


def views_for_model(front_picture, prompt: str | None, subject: str = "person",
                    client: ComfyClient | None = None, *, shape, which=("back",),
                    seed: int | None = None, denoise: float = 0.65, cutout_model: str = "isnet-general-use",
                    progress=None, inits: dict | None = None, **gen_opts) -> dict[str, np.ndarray]:
    """Matching back (and left / right side) pictures of a character for painting its shape.

    front_picture: the front picture (RGBA, background removed). shape: the uncoloured
    model (.glb path or bytes) built from it. which: any of back, left, right. Each view
    starts from the shape rendered from there (view_inits: keeps the outline and puts the
    front's colours roughly in place) and SDXL redraws it from words, img2img at `denoise`
    with the same seed and no pixel-art LoRA. Returns {view: RGBA} in the shape's own
    outline, ready for paint_from_picture(back=, left=, right=). inits: filled with the
    starting pictures (to look at)."""
    from .generate import generate_images
    front = load_image(front_picture)
    bad = [w for w in which if w not in VIEW_TURN]
    if bad:
        raise ValueError(f"Unknown view(s) {', '.join(bad)} (use back, left, right)")
    say = progress or (lambda text: log.info("%s", text))
    starts = view_inits(shape, front, which)
    if inits is not None:
        inits.update(starts)
    seed = random.randrange(2**32) if seed is None else int(seed)
    out = {}
    for name, init in starts.items():
        say(f"Drawing the {name if name == 'back' else name + ' side'} of it on your GPU…")
        framed = np.asarray(framed_on_white(init, 1024).convert("RGB"))
        mask = np.asarray(framed_on_white(_white_mask(init), 1024).convert("L")) < 128
        positive, negative = view_prompt(prompt, name, subject)
        (img, _), = generate_images(positive, negative, client=client, seed=seed,
                                    init_image=Image.fromarray(framed), denoise=denoise,
                                    aspect=1.0, **{"lora": "none", **gen_opts})
        img = np.asarray(img.convert("RGB"))
        if img.shape[:2] != framed.shape[:2]:
            img = cv2.resize(img, (framed.shape[1], framed.shape[0]), interpolation=cv2.INTER_AREA)
        drawn = cut_out(img, cutout_model)
        start = np.dstack([framed, np.where(mask, 255, 0).astype(np.uint8)])
        out[name] = _keep_outline(drawn, start)
    return out


def _white_mask(rgba: np.ndarray) -> np.ndarray:
    """The subject black on white (framed like the picture by framed_on_white)."""
    out = np.zeros_like(rgba)
    out[..., 3] = rgba[..., 3]
    return out


def views_to_make(views: str | None, from_words: bool, back_given: bool) -> tuple[str, ...]:
    """Which pictures prompt_to_model has the GPU draw: views front / front+back / all
    (None = front+back from words, front for your own picture); a back you gave wins."""
    views = views or ("front+back" if from_words else "front")
    if views not in VIEWS:
        raise ValueError(f"views must be one of {', '.join(VIEWS)}")
    want = {"front": (), "front+back": ("back",), "all": ("back", "left", "right")}[views]
    return tuple(v for v in want if not (v == "back" and back_given))


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
            # ComfyUI's SaveGLB adds a plain grey material with no image and no uvs: not colours
            mat = getattr(m.visual, "material", None)
            image = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if image is not None and getattr(m.visual, "uv", None) is not None:
                return True
            continue
        if kind in ("vertex", "face"):
            cols = np.asarray(m.visual.vertex_colors if kind == "vertex" else m.visual.face_colors)
            if len(cols) and np.ptp(cols[:, :3], axis=0).max() > 8:
                return True
    return False


# --------------------------------------------------------------------------- only these colours

COLOUR_NAMES = {
    "black": (28, 28, 32), "white": (238, 238, 236), "grey": (128, 128, 130),
    "gray": (128, 128, 130), "dark grey": (70, 70, 74), "light grey": (180, 180, 182),
    "brown": (110, 70, 42), "dark brown": (62, 40, 26), "tan": (196, 158, 110),
    "cream": (238, 224, 194), "gold": (212, 160, 60), "red": (190, 40, 40),
    "orange": (226, 120, 40), "yellow": (236, 204, 60), "pink": (205, 110, 125),
    "green": (60, 140, 60), "blue": (50, 90, 190), "purple": (120, 60, 160),
}


def parse_colours(spec) -> np.ndarray:
    """'black, white' / '#1c1c20 #eeeeec' / names and hex mixed -> (n, 3) uint8."""
    if isinstance(spec, np.ndarray):
        return spec.astype(np.uint8).reshape(-1, 3)
    out = []
    for part in str(spec).replace(";", ",").split(","):
        word = " ".join(part.lower().split())
        if not word:
            continue
        if word in COLOUR_NAMES:
            out.append(COLOUR_NAMES[word])
            continue
        hexes = word.split()
        if all(len(h.lstrip("#")) == 6 for h in hexes):
            try:
                out += [tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)) for h in hexes]
                continue
            except ValueError:
                pass
        raise ValueError(f"Unknown colour '{part.strip()}'. Use names ({', '.join(COLOUR_NAMES)}) "
                         "or hex like #1c1c20, separated by commas.")
    if not out:
        raise ValueError("Give at least one colour, e.g. 'black, white'.")
    return np.array(out, np.uint8)


def _nearest(rgb: np.ndarray, palette: np.ndarray, stretch: bool = False) -> np.ndarray:
    """Index of the nearest palette colour (Lab). stretch: first map the colours' lightness
    range (2nd-98th percentile) onto the palette's, so a white dog in shade still reads
    white and black fur in sun still reads black."""
    from .color import rgb_to_lab
    lab = rgb_to_lab(rgb.reshape(-1, 3).astype(np.float32))
    pal = rgb_to_lab(palette.astype(np.float32))
    if stretch and len(pal) > 1 and len(lab) > 1:
        lo, hi = np.percentile(lab[:, 0], [2, 98])
        if hi - lo > 1:
            lab[:, 0] = pal[:, 0].min() + (lab[:, 0] - lo) / (hi - lo) * np.ptp(pal[:, 0])
    return np.argmin(((lab[:, None] - pal[None]) ** 2).sum(-1), 1)


def picture_to_colours(rgba: np.ndarray, palette, speck: float = 0.015) -> np.ndarray:
    """The picture in only these colours, without specks thinner than `speck` of the
    subject's height (sun glare on fur, stray hairs); they take the nearest colour around."""
    pal = parse_colours(palette)
    out = rgba.copy()
    inside = rgba[..., 3] >= 128
    if not inside.any():
        return out
    label = np.full(inside.shape, -1, np.int32)
    label[inside] = _nearest(rgba[..., :3][inside], pal, stretch=True)
    ys = np.nonzero(inside)[0]
    k = max(3, int(speck * (ys.max() - ys.min() + 1)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    kept = np.full(inside.shape, -1, np.int32)
    for i in range(len(pal)):
        mask = cv2.morphologyEx((label == i).astype(np.uint8), cv2.MORPH_OPEN, kernel)
        kept[(mask > 0) & inside] = i
    lost = inside & (kept < 0)
    if lost.any() and (kept >= 0).any():
        _, near = cv2.distanceTransformWithLabels((kept < 0).astype(np.uint8), cv2.DIST_L2, 5,
                                                  labelType=cv2.DIST_LABEL_PIXEL)
        ky, kx = np.nonzero(kept >= 0)
        lut = np.zeros(int(near.max()) + 1, np.int32)
        lut[near[ky, kx]] = kept[ky, kx]
        kept[lost] = lut[near[lost]]
    elif lost.any():
        kept[lost] = label[lost]
    out[..., :3][inside] = pal[kept[inside]]
    return out


def snap_to_colours(mesh, palette, rounds: int = 6):
    """Every vertex to the nearest of these colours, then lone specks to the colour most of
    their neighbours have (blurred guesses on hidden sides would otherwise be in-betweens)."""
    pal = parse_colours(palette)
    cols = np.asarray(mesh.visual.vertex_colors)[:, :3]
    label = _nearest(cols, pal)
    edges = np.asarray(mesh.edges_unique)
    if len(edges) and len(pal) > 1:
        a, b = edges[:, 0], edges[:, 1]
        n = len(label)
        for _ in range(rounds):
            votes = np.zeros((n, len(pal)))
            np.add.at(votes, (a, label[b]), 1)
            np.add.at(votes, (b, label[a]), 1)
            total = votes.sum(1)
            best = votes.argmax(1)
            sure = (total > 0) & (votes[np.arange(n), best] >= 0.6 * total)
            label = np.where(sure, best, label)
    rgba = np.concatenate([pal[label], np.full((len(label), 1), 255, np.uint8)], 1)
    mesh.visual = _trimesh().visual.ColorVisuals(mesh, vertex_colors=rgba)
    return mesh


# --------------------------------------------------------------------------- all together

def prompt_to_model(prompt: str | None, out_glb, *, subject: str = "person",
                    picture: ImageSource | None = None, back: ImageSource | None = None,
                    shape: bytes | str | Path | None = None, client: ComfyClient | None = None,
                    seed: int | None = None, shape_options: ShapeOptions | None = None,
                    cutout_model: str = "isnet-general-use", progress=None,
                    mirror_back: bool = False, colours=None, views: str | None = None,
                    left: ImageSource | None = None, right: ImageSource | None = None,
                    view_denoise: float = 0.65, **gen_opts) -> ModelResult:
    """Prompt (or a picture, or a picture + a ready shape) -> coloured .glb.

    Writes <out>.glb (coloured), <out>.front.png (the picture) and <out>.shape.glb (the
    shape as ComfyUI made it). progress(step, text) is called as each step starts.
    mirror_back: no back picture -> paint the far side from the picture mirrored (animals
    and people seen from the side, marked about the same on both sides). colours: only
    these colours ('black, white', hex), e.g. a black-and-white dog photographed in shade.
    views: front / front+back / all: the GPU also draws the back (and both sides) from the
    shape (views_for_model), saved as <out>.back.png etc. Default front+back when made from
    words, front for your own picture. left / right: side pictures you already have."""
    out = Path(out_glb)
    out.parent.mkdir(parents=True, exist_ok=True)
    say = progress or (lambda step, text: log.info("%s: %s", step, text))
    used_seed = seed
    from_words = picture is None
    make_views = views_to_make(views, from_words, back is not None or mirror_back)
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
    if back_rgba is None and mirror_back:
        back_rgba = np.ascontiguousarray(rgba[:, ::-1])
    paint_rgba = rgba
    if colours is not None:
        paint_rgba = picture_to_colours(rgba, colours)
        if back_rgba is not None:
            back_rgba = picture_to_colours(back_rgba, colours)

    raw_path = None
    if shape is None:
        say("shape", "Building the 3D shape on your GPU (Hunyuan3D, about a minute)…")
        glb = shape_from_picture(framed_on_white(rgba), client=client, seed=seed,
                                 options=shape_options)
        raw_path = out.with_suffix(".shape.glb")
        raw_path.write_bytes(glb)
    else:
        glb = shape if isinstance(shape, bytes) else Path(shape).read_bytes()

    sides = {"left": cut_out(left, cutout_model) if left is not None else None,
             "right": cut_out(right, cutout_model) if right is not None else None}
    made: dict[str, Path] = {}
    if has_colours(glb):
        say("colour", "The model came with its own colours; keeping them.")
        out.write_bytes(glb)
    else:
        make_views = tuple(v for v in make_views if v == "back" or sides[v] is None)
        if make_views:
            say("views", "Drawing the back" + (" and sides" if len(make_views) > 1 else "")
                + " of it on your GPU…")
            starts: dict = {}
            drawn = views_for_model(rgba, prompt, subject, client, shape=glb, which=make_views,
                                    seed=used_seed, denoise=view_denoise,
                                    cutout_model=cutout_model, inits=starts,
                                    progress=lambda text: say("views", text), **gen_opts)
            for name, pic in drawn.items():
                path = out.with_suffix(f".{name}.png")
                Image.fromarray(pic).save(path)
                Image.fromarray(starts[name]).save(out.with_suffix(f".{name}.start.png"))
                made[name] = path
                if colours is not None:
                    pic = picture_to_colours(pic, colours)
                if name == "back":
                    back_rgba = pic
                else:
                    sides[name] = pic
        elif colours is not None:
            sides = {k: picture_to_colours(v, colours) if v is not None else None
                     for k, v in sides.items()}
        say("colour", "Painting the model from the picture" + ("s…" if made else "…"))
        mesh, match = paint_from_picture(glb, paint_rgba, back_rgba, left=sides["left"],
                                         right=sides["right"])
        if match < 0.5:
            say("colour", f"Warning: the shape only matches the picture {match:.0%}.")
        if colours is not None:
            mesh = snap_to_colours(mesh, colours)
        mesh.export(out)
    return ModelResult(out, pic_path, raw_path, used_seed, made)


def export_for_rigging(model, out_obj) -> Path:
    """The model as .obj (y up, facing +z) to upload to Mixamo for a skeleton and animations.
    Mixamo drops the colours: give the original back with turnaround --colours-from."""
    trimesh = _trimesh()
    mesh = trimesh.load(str(model), force="mesh", process=False)
    out_obj = Path(out_obj)
    mesh.export(str(out_obj))
    return out_obj
