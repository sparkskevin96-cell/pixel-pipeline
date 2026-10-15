"""Game-ready 3D models from the app's characters, items and props.

    <asset>/model_game/
        lod0.glb lod1.glb lod2.glb   textured (+ normal map), skinned + every clip of
                                     rigged.glb (autorig, motions, mocap, springs) for
                                     characters; plain meshes for items/props
        lod0_albedo.png lod0_normal.png …   the textures (also embedded in the .glb files)
        lod0.obj + .mtl              static items/props only: opens in any engine, Unity
                                     natively
        <name>.ppmodel               Unity: all LODs + skeleton + clips, read by
                                     PixelPipelineModelImporter.cs (no glTF package needed)
        preview.png / preview.gif    turntable, wire.png (wireframe + triangle counts)
        info.json                    triangles, textures, clips, timings, notes

Pipeline: heavy painted mesh -> gamemesh.build (clean remesh, symmetry, smoothing, QEM LODs)
-> uvbake.unwrap (LSCM charts, raster packing) -> bake colour from the painted model and
normals from the cleaned high-poly -> skin weights moved over from the heavy rigged mesh
(closest point on the same side, barycentric blend of the corners' weights) -> glTF.

Looks: "smooth" (1024 texture, bilinear, normal map) or "pixel" (pixel-textured low poly:
small texture, colours snapped to the project palette or k-means, NEAREST sampling)."""
from __future__ import annotations

import copy
import io
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import gamemesh as gm
from . import uvbake as ub

log = logging.getLogger(__name__)

LOOKS = ("smooth", "pixel")
FOLDER = "model_game"
TEXTURE = {"smooth": 1024, "pixel": 256}
NEAREST, LINEAR, NEAREST_MIP_LINEAR, LINEAR_MIP_LINEAR = 9728, 9729, 9986, 9987


# --------------------------------------------------------------------------- source

@dataclass
class Source:
    path: Path
    v: np.ndarray
    f: np.ndarray
    colours: np.ndarray                  # (n, 3) sRGB 0..255
    joints: np.ndarray | None = None     # (n, 4) skin joint slots
    weights: np.ndarray | None = None    # (n, 4)
    doc: dict | None = None              # the rigged glTF (nodes, skins, animations)
    bin: bytes = b""

    @property
    def skinned(self) -> bool:
        return self.joints is not None


def load_source(path) -> Source:
    """A painted model.glb or a rigged.glb (first skinned mesh) -> Source."""
    from .anim3d import Gltf, _linear_to_srgb, _quat_matrix
    path = Path(path)
    g = Gltf(path)
    doc = g.doc
    skinned = bool(doc.get("skins"))
    vs, fs, cs, js, ws = [], [], [], [], []
    base = 0
    # world matrices for static nodes
    nodes = doc.get("nodes", [])
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def world(i):
        m = np.eye(4)
        while i is not None:
            n = nodes[i]
            if "matrix" in n:
                l = np.asarray(n["matrix"], np.float64).reshape(4, 4).T
            else:
                l = np.eye(4)
                l[:3, :3] = _quat_matrix(n.get("rotation", [0, 0, 0, 1])) * np.asarray(
                    n.get("scale", [1, 1, 1]), np.float64)
                l[:3, 3] = n.get("translation", [0, 0, 0])
            m = l @ m
            i = parent.get(i)
        return m

    for ni, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        if skinned and "skin" not in node:
            continue
        mat = np.eye(4) if "skin" in node else world(ni)
        for pr in doc["meshes"][node["mesh"]]["primitives"]:
            if pr.get("mode", 4) != 4:
                continue
            at = pr["attributes"]
            v = g.accessor(at["POSITION"])[:, :3]
            v = v @ mat[:3, :3].T + mat[:3, 3]
            f = (g.accessor(pr["indices"]).reshape(-1, 3) if "indices" in pr
                 else np.arange(len(v)).reshape(-1, 3))
            if "COLOR_0" in at:
                c = g.accessor(at["COLOR_0"])[:, :3]
                acc = doc["accessors"][at["COLOR_0"]]
                # our rigged files store linear floats (glTF rule); trimesh's model.glb stores
                # the sRGB bytes as they are
                if acc["componentType"] == 5126:
                    c = _linear_to_srgb(c)
                c = c * 255.0
            else:
                fac = (doc.get("materials", [{}])[pr["material"]].get("pbrMetallicRoughness", {})
                       .get("baseColorFactor", [0.8, 0.8, 0.8, 1]) if "material" in pr
                       else [0.8, 0.8, 0.8, 1])
                c = np.tile(_linear_to_srgb(np.asarray(fac[:3])) * 255, (len(v), 1))
                tex = _texture_colours(g, doc, pr, at)
                if tex is not None:
                    c = tex
            vs.append(v)
            fs.append(f + base)
            cs.append(c)
            if skinned:
                js.append(g.accessor(at["JOINTS_0"]).astype(np.int64))
                ws.append(g.accessor(at["WEIGHTS_0"]))
            base += len(v)
        if skinned:
            break
    if not vs:
        raise ValueError(f"No triangles in {path.name}")
    src = Source(path, np.concatenate(vs), np.concatenate(fs).astype(np.int64), np.concatenate(cs))
    if skinned:
        src.joints, src.weights = np.concatenate(js), np.concatenate(ws)
        src.doc, src.bin = doc, (g.buffers[0] if g.buffers else b"")
    return src


def _texture_colours(g, doc, pr, at):
    """Per-vertex colours sampled from a base colour texture (for textured inputs)."""
    from .anim3d import _linear_to_srgb  # noqa: F401
    if "TEXCOORD_0" not in at or "material" not in pr:
        return None
    pbr = doc["materials"][pr["material"]].get("pbrMetallicRoughness", {})
    if "baseColorTexture" not in pbr:
        return None
    img = g.image(pbr["baseColorTexture"]["index"])
    if img is None:
        return None
    uv = g.accessor(at["TEXCOORD_0"])
    h, w = img.shape[:2]
    x = np.clip((np.mod(uv[:, 0], 1) * w).astype(int), 0, w - 1)
    y = np.clip((np.mod(uv[:, 1], 1) * h).astype(int), 0, h - 1)
    return img[y, x, :3].astype(np.float64)


# --------------------------------------------------------------------------- skin transfer

def transfer_weights(src: Source, surface: gm.Surface, v, n, influences: int = 4):
    """Skin weights for new vertices from the heavy mesh: closest point on the same side,
    the three corners' weights blended by barycentrics, top `influences`, normalised."""
    fi, b = ub._same_side(surface, v, n, reach=float(np.ptp(v, 0).max()))
    corners = surface.f[fi]                                   # (m, 3) heavy vertex ids
    J = int(src.joints.max()) + 1
    dense = np.zeros((len(v), J))
    rows = np.arange(len(v))
    for k in range(3):
        jj = src.joints[corners[:, k]]
        ww = src.weights[corners[:, k]] * b[:, k:k + 1]
        for s in range(jj.shape[1]):
            np.add.at(dense, (rows, jj[:, s]), ww[:, s])
    top = np.argsort(-dense, axis=1)[:, :influences]
    wt = np.take_along_axis(dense, top, 1)
    wt = np.where(wt < 1e-4, 0.0, wt)
    s = wt.sum(1, keepdims=True)
    empty = s[:, 0] <= 0
    if empty.any():                                           # nothing nearby: nearest corner
        top[empty, 0] = src.joints[corners[empty, 0], 0]
        wt[empty] = (1, 0, 0, 0)[:influences]
        s = wt.sum(1, keepdims=True)
    wt = wt / s
    order = np.argsort(-wt, 1)
    return np.take_along_axis(top, order, 1), np.take_along_axis(wt, order, 1)


# --------------------------------------------------------------------------- one LOD

@dataclass
class Lod:
    index: int
    mesh: gm.Mesh                 # welded low mesh
    atlas: ub.Atlas
    bake: ub.Bake
    joints: np.ndarray | None = None
    weights: np.ndarray | None = None
    glb: Path | None = None

    @property
    def tris(self) -> int:
        return len(self.atlas.f)


def _palette(palette) -> np.ndarray | None:
    if palette is None or (isinstance(palette, str) and palette in ("", "auto")):
        return None
    if isinstance(palette, str):
        from .palettes import load_palette
        try:
            return np.asarray(load_palette(palette), np.float64)[:, :3]
        except Exception as e:                    # unknown palette name: k-means instead
            log.warning("palette %s: %s", palette, e)
            return None
    return np.asarray(palette, np.float64)[:, :3]


def make_lod(index, low: gm.Mesh, src: Source, colour_surface: gm.Surface, high: gm.Mesh | None,
             *, size: int, look: str, normal_map: bool, palette=None, colours: int = 32) -> Lod:
    at = ub.unwrap(low, size, padding=2 if look == "smooth" else 1)
    vn = gm.vertex_normals(low.v, low.f)
    ss = 4 if look == "pixel" else 1
    bk = ub.bake(at, vn, (colour_surface, src.colours), high if normal_map else None,
                 supersample=ss, gutter=8 if look == "smooth" else 4)
    if look == "pixel":
        bk.colour = ub.quantize_texture(bk.colour, bk.mask, colours, _palette(palette))
    lod = Lod(index, low, at, bk)
    if src.skinned:
        lod.joints, lod.weights = transfer_weights(src, colour_surface, at.v, bk.vertex_normals)
    return lod


# --------------------------------------------------------------------------- glTF writing

def _png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _add_mesh(g, lod: Lod, look: str, name: str) -> int:
    """Mesh + material + textures into a GlbWriter; returns the mesh index."""
    at, bk = lod.atlas, lod.bake
    attrs = {"POSITION": g.accessor(at.v.astype(np.float32), "VEC3"),
             "NORMAL": g.accessor(bk.vertex_normals.astype(np.float32), "VEC3"),
             "TEXCOORD_0": g.accessor(at.uv.astype(np.float32), "VEC2")}
    if bk.normal is not None:
        attrs["TANGENT"] = g.accessor(bk.tangents.astype(np.float32), "VEC4")
    if lod.joints is not None:
        attrs["JOINTS_0"] = g.accessor(lod.joints.astype(np.uint8) if lod.joints.max() < 256
                                       else lod.joints.astype(np.uint16), "VEC4",
                                       5121 if lod.joints.max() < 256 else 5123)
        attrs["WEIGHTS_0"] = g.accessor(lod.weights.astype(np.float32), "VEC4")
    doc = g.doc
    for k in ("images", "textures", "samplers", "materials", "meshes"):
        doc.setdefault(k, [])
    pixel = look == "pixel"
    doc["samplers"].append({"magFilter": NEAREST if pixel else LINEAR,
                            "minFilter": NEAREST_MIP_LINEAR if pixel else LINEAR_MIP_LINEAR,
                            "wrapS": 33071, "wrapT": 33071})            # clamp: it's an atlas
    smp = len(doc["samplers"]) - 1

    def tex(img):
        doc["images"].append({"bufferView": g.view(_png(img)), "mimeType": "image/png"})
        doc["textures"].append({"source": len(doc["images"]) - 1, "sampler": smp})
        return len(doc["textures"]) - 1

    mat = {"name": f"{name}_lod{lod.index}",
           "pbrMetallicRoughness": {"baseColorTexture": {"index": tex(bk.colour[..., :3].copy())},
                                    "metallicFactor": 0.0, "roughnessFactor": 0.9}}
    if bk.normal is not None:
        mat["normalTexture"] = {"index": tex(bk.normal)}
    doc["materials"].append(mat)
    doc["meshes"].append({"name": f"{name}_lod{lod.index}", "primitives": [{
        "attributes": attrs, "indices": g.accessor(at.f.astype(np.uint32).reshape(-1), "SCALAR", 5125),
        "material": len(doc["materials"]) - 1}]})
    return len(doc["meshes"]) - 1


def write_glb(lod: Lod, out, *, look: str, name: str, src: Source) -> Path:
    from .anim3d import GlbWriter
    g = GlbWriter()
    if src.skinned:
        from .springs import _compact
        doc = copy.deepcopy(src.doc)
        for k in ("meshes", "materials", "images", "textures", "samplers"):
            doc.pop(k, None)
        for e in ("extensionsUsed", "extensionsRequired"):
            doc.pop(e, None)
        doc["buffers"] = [{}]
        g.doc, g.bin = doc, bytearray(src.bin)
        mi = _add_mesh(g, lod, look, name)
        body = [i for i, n in enumerate(doc["nodes"]) if "skin" in n]
        for i in body:
            doc["nodes"][i]["mesh"] = mi
            doc["nodes"][i]["name"] = name
        g.doc, g.bin = _compact(g.doc, g.bin)
    else:
        mi = _add_mesh(g, lod, look, name)
        g.doc["nodes"] = [{"name": name, "mesh": mi}]
        g.doc["scenes"] = [{"nodes": [0]}]
        g.doc["scene"] = 0
    g.doc["asset"]["generator"] = "pixel-pipeline gamemodel"
    return g.save(out)


def write_obj(lod: Lod, folder: Path, stem: str) -> list[Path]:
    """Wavefront .obj + .mtl + albedo png (static models; any engine opens it)."""
    at, bk = lod.atlas, lod.bake
    tex = folder / f"{stem}_albedo.png"
    if not tex.exists():
        Image.fromarray(bk.colour[..., :3]).save(tex)
    mtl = folder / f"{stem}.mtl"
    mtl.write_text(f"newmtl {stem}\nKd 1 1 1\nKa 0 0 0\nKs 0 0 0\nd 1\nillum 1\nmap_Kd {tex.name}\n",
                   encoding="utf-8")
    lines = [f"mtllib {mtl.name}", f"o {stem}"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in at.v]
    lines += [f"vt {u:.6f} {1 - v:.6f}" for u, v in at.uv]           # obj: v up
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in bk.vertex_normals]
    lines.append(f"usemtl {stem}")
    lines += ["f " + " ".join(f"{i}/{i}/{i}" for i in tri + 1) for tri in at.f]
    obj = folder / f"{stem}.obj"
    obj.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [obj, mtl, tex]


# --------------------------------------------------------------------------- Unity (.ppmodel)

def _unity_clips(src: Source) -> tuple[list, list, list]:
    """Bones [{name, parent, position, rotation}] and clips in glTF space from the rigged doc."""
    from .anim3d import Gltf  # noqa: F401
    doc = src.doc
    nodes = doc["nodes"]
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
    skin = doc["skins"][0]
    bone_nodes = [i for i, n in enumerate(nodes) if "mesh" not in n and "skin" not in n]
    index = {ni: k for k, ni in enumerate(bone_nodes)}
    bones = []
    for ni in bone_nodes:
        n = nodes[ni]
        bones.append({"name": n.get("name", f"node{ni}"),
                      "parent": index.get(parent.get(ni), -1),
                      "position": [float(x) for x in n.get("translation", [0, 0, 0])],
                      "rotation": [float(x) for x in n.get("rotation", [0, 0, 0, 1])]})
    acc = _AccessorReader(doc, src.bin)
    clips = []
    for a in doc.get("animations", []):
        curves, length = [], 0.0
        for ch in a["channels"]:
            node = ch["target"]["node"]
            if node not in index or ch["target"]["path"] not in ("rotation", "translation"):
                continue
            s = a["samplers"][ch["sampler"]]
            t = acc(s["input"]).reshape(-1)
            val = acc(s["output"])
            if s.get("interpolation") == "CUBICSPLINE":
                val = val.reshape(len(t), 3, -1)[:, 1]
            if ch["target"]["path"] == "rotation":
                val = val.copy()
                for k in range(1, len(val)):
                    if val[k] @ val[k - 1] < 0:
                        val[k] = -val[k]
            curves.append({"bone": index[node], "property": ch["target"]["path"],
                           "times": [round(float(x), 5) for x in t],
                           "values": [round(float(x), 6) for x in val.reshape(-1)],
                           "step": s.get("interpolation") == "STEP"})
            length = max(length, float(t.max()) if len(t) else 0.0)
        loop = bool((a.get("extras") or {}).get("loop", a.get("name", "") in
                                                ("idle", "walk", "run")))
        clips.append({"name": a.get("name", f"clip{len(clips)}"), "loop": loop,
                      "length": length, "curves": curves})
    return bones, clips, [index[j] for j in skin["joints"]]


class _AccessorReader:
    def __init__(self, doc, bin_):
        self.doc, self.bin = doc, bin_

    def __call__(self, i):
        from .anim3d import COMPONENTS, WIDTHS
        a = self.doc["accessors"][i]
        v = self.doc["bufferViews"][a["bufferView"]]
        dt = np.dtype(COMPONENTS[a["componentType"]]).newbyteorder("<")
        n = WIDTHS[a["type"]]
        start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = v.get("byteStride") or n * dt.itemsize
        arr = np.ndarray((a["count"], n), dt, self.bin, start, (stride, dt.itemsize))
        return arr.astype(np.float64)


def write_ppmodel(lods: list[Lod], folder: Path, name: str, look: str, src: Source,
                  kind: str) -> list[Path]:
    """One JSON file for Unity's PixelPipelineModelImporter: every LOD (glTF space; the
    importer mirrors x), textures as sibling PNGs, the skeleton and the clips."""
    files = []
    data = {"format": "pixel-pipeline-model", "version": 1, "name": name, "kind": kind,
            "look": look, "filter": "point" if look == "pixel" else "bilinear", "lods": [],
            "skinned": src.skinned, "bones": [], "skinBones": [], "clips": []}
    for lod in lods:
        at, bk = lod.atlas, lod.bake
        alb = folder / f"lod{lod.index}_albedo.png"
        if not alb.exists():
            Image.fromarray(bk.colour[..., :3]).save(alb)
        entry = {"albedo": alb.name, "normal": "",
                 "vertices": np.round(at.v, 6).reshape(-1).tolist(),
                 "normals": np.round(bk.vertex_normals, 5).reshape(-1).tolist(),
                 "uv": np.round(at.uv, 6).reshape(-1).tolist(),
                 "triangles": at.f.reshape(-1).astype(int).tolist(),
                 "tangents": np.round(bk.tangents, 5).reshape(-1).tolist() if bk.normal is not None else [],
                 "boneIndices": [], "boneWeights": []}
        if bk.normal is not None:
            nrm = folder / f"lod{lod.index}_normal.png"
            if not nrm.exists():
                Image.fromarray(bk.normal).save(nrm)
            entry["normal"] = nrm.name
        if lod.joints is not None:
            entry["boneIndices"] = lod.joints.reshape(-1).astype(int).tolist()
            entry["boneWeights"] = np.round(lod.weights, 5).reshape(-1).tolist()
        data["lods"].append(entry)
    if src.skinned:
        data["bones"], data["clips"], data["skinBones"] = _unity_clips(src)
    p = folder / f"{_stem(name)}.ppmodel"
    p.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    files.append(p)
    return files


def _stem(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "model"


# --------------------------------------------------------------------------- previews

def preview_images(lod: Lod, folder: Path, look: str, frames: int = 8, size: int = 256) -> list[Path]:
    from .render3d import turntable
    at, bk = lod.atlas, lod.bake
    imgs = turntable(at.v, at.f, frames, uv=at.uv, texture=bk.colour, normal_map=bk.normal,
                     tangents=bk.tangents, normals=bk.vertex_normals, nearest=look == "pixel",
                     size=size, elev=15.0)
    strip = np.concatenate(imgs, 1)
    out = folder / "preview.png"
    Image.fromarray(strip).save(out)
    gif = folder / "preview.gif"
    pil = [Image.fromarray(_on_grey(i)) for i in imgs]
    pil[0].save(gif, save_all=True, append_images=pil[1:], duration=150, loop=0)
    thumb = folder / "thumb.png"
    Image.fromarray(imgs[1]).resize((96, 96), Image.LANCZOS).save(thumb)
    return [out, gif, thumb]


def _on_grey(img: np.ndarray) -> np.ndarray:
    bg = np.array([46, 48, 56], np.float64)
    a = img[..., 3:4] / 255.0
    return np.clip(img[..., :3] * a + bg * (1 - a), 0, 255).astype(np.uint8)


def wire_image(src: Source, lods: list[Lod], high: gm.Mesh, out: Path, size: int = 384,
               yaw: float = 25.0, look: str = "smooth") -> Path:
    """Before / after sheet: heavy painted model, its wireframe, game LODs textured + wire."""
    from PIL import ImageDraw
    from .render3d import render
    lo, hi = src.v.min(0), src.v.max(0)
    frame = ((lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2 * 0.85)
    grid = [[], [], []]                    # rows: source | textured LODs | wireframes
    grid[0].append((render(src.v, src.f, colours=src.colours, size=size, yaw=yaw, frame=frame),
                    f"heavy model, vertex colours  {len(src.f):,} tris"))
    grid[0].append((render(src.v, src.f, size=size, yaw=yaw, frame=frame),
                    "heavy model shape (lumpy)"))
    grid[0].append((render(high.v, high.f, size=size, yaw=yaw, frame=frame),
                    f"cleaned + smoothed  {len(high.f):,} tris"))
    for lod in lods:
        at, bk = lod.atlas, lod.bake
        grid[1].append((render(at.v, at.f, uv=at.uv, texture=bk.colour, normal_map=bk.normal,
                               tangents=bk.tangents, normals=bk.vertex_normals, size=size,
                               yaw=yaw, frame=frame, nearest=look == "pixel"),
                        f"LOD{lod.index}  {lod.tris:,} tris  {bk.colour.shape[0]}px "
                        f"{'+ normal map' if bk.normal is not None else ''}"))
        grid[2].append((render(at.v, at.f, normals=bk.vertex_normals, size=size, yaw=yaw,
                               frame=frame, wire=True), f"LOD{lod.index} wireframe"))
    cols = max(len(r) for r in grid)
    head = 22
    sheet = Image.new("RGB", (cols * size, 3 * (size + head)), (36, 38, 46))
    d = ImageDraw.Draw(sheet)
    for r, row in enumerate(grid):
        for c, (img, label) in enumerate(row):
            sheet.paste(Image.fromarray(_on_grey(img)), (c * size, r * (size + head) + head))
            d.text((c * size + 6, r * (size + head) + 5), label, fill=(230, 230, 235))
    sheet.save(out)
    return out


# --------------------------------------------------------------------------- one call

@dataclass
class GameModel:
    folder: Path
    lods: list
    info: dict
    files: list = field(default_factory=list)

    def lod_path(self, i: int = 0) -> Path:
        return self.folder / f"lod{i}.glb"


def default_look(style=None) -> str:
    """'pixel' for pixel-art projects (house words or HD-2D tiles), else 'smooth'."""
    if style is None:
        return "smooth"
    words = f"{getattr(style, 'style_prompt', '')}".lower()
    if "pixel" in words or "16-bit" in words or getattr(style, "tile_style", "") == "hd2d":
        return "pixel"
    return "smooth"


def symmetry_plane(src: Source):
    """(point, normal) of a character's left/right mirror plane from its rig, else None."""
    if not src.skinned:
        return None
    info = ((src.doc["skins"][0].get("extras") or {}).get("pixel_pipeline_rig") or {})
    j = info.get("joints") or {}
    pairs = [("thigh_l", "thigh_r"), ("upper_arm_l", "upper_arm_r"), ("upper_front_l", "upper_front_r"),
             ("upper_back_l", "upper_back_r")]
    for a, b in pairs:
        if a in j and b in j:
            pa, pb = np.asarray(j[a]), np.asarray(j[b])
            n = pa - pb
            n[1] = 0.0
            if np.linalg.norm(n) > 1e-6:
                return (pa + pb) / 2, n / np.linalg.norm(n)
    return None


def make_game_model(model, out_dir=None, *, kind: str = "character", name: str | None = None,
                    budget: int | None = None, look: str = "smooth", texture: int | None = None,
                    normal_map: bool | None = None, symmetry: str = "auto", palette=None,
                    colours: int = 32, voxels: int = 256, lods: int = 3, unity: bool = True,
                    progress=None) -> GameModel:
    """A painted (and maybe rigged) heavy model -> model_game/ with LOD glbs, textures,
    previews, info.json (+ .obj for static models, .ppmodel for Unity)."""
    say = progress or (lambda t: None)
    if look not in LOOKS:
        raise ValueError(f"look must be one of {LOOKS}")
    model = Path(model)
    out = Path(out_dir) if out_dir else model.parent / FOLDER
    out.mkdir(parents=True, exist_ok=True)
    name = name or model.parent.name
    from . import cache
    key = cache.digest(model, kind, name, budget, look, texture, normal_map, symmetry,
                       None if palette is None else (np.asarray(palette) if not isinstance(
                           palette, (str, Path)) else str(palette)),
                       colours, voxels, lods, unity, gm.fast_available(),
                       cache.code_digest("gamemesh", "uvbake", "gamemodel", "render3d", "anim3d"))
    done = _unchanged(out, key)
    if done is not None:                  # same model, same options, same code: nothing to do
        say("The game model is up to date (same model and options).")
        return done
    times = {}
    t0 = time.perf_counter()
    say("Reading the model…")
    src = load_source(model)
    times["load"] = time.perf_counter() - t0
    budget = budget or gm.BUDGETS.get(kind, 4000)
    texture = texture or TEXTURE[look]
    normal_map = (look == "smooth") if normal_map is None else normal_map
    say("Cleaning and remeshing…")
    opts = gm.Options(budget=budget, lods=gm.LOD_SHARES[:max(1, lods)], voxels=voxels,
                      symmetry=symmetry if kind in ("character", "animal") or symmetry == "on"
                      else "off", plane=symmetry_plane(src))
    meshes = gm.build(src.v, src.f, opts)
    times.update(meshes.timings)
    t = time.perf_counter()
    say("Baking textures…")
    surface = gm.Surface(src.v, src.f)
    high_surface = gm.Surface(meshes.high.v, meshes.high.f)
    def one(i):
        return make_lod(i, meshes.lods[i], src, surface,
                        high_surface if normal_map and i < 2 else None,
                        size=max(64, texture >> i), look=look, normal_map=normal_map and i < 2,
                        palette=palette, colours=colours)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max(1, len(meshes.lods))) as ex:   # LODs side by side (the same
        out_lods = list(ex.map(one, range(len(meshes.lods))))  # bakes, ~25% sooner)
    times["bake"] = time.perf_counter() - t
    t = time.perf_counter()
    say("Writing files…")
    files = []
    for lod in out_lods:
        Image.fromarray(lod.bake.colour[..., :3]).save(out / f"lod{lod.index}_albedo.png")
        files.append(out / f"lod{lod.index}_albedo.png")
        if lod.bake.normal is not None:
            Image.fromarray(lod.bake.normal).save(out / f"lod{lod.index}_normal.png")
            files.append(out / f"lod{lod.index}_normal.png")
        lod.glb = write_glb(lod, out / f"lod{lod.index}.glb", look=look, name=_stem(name), src=src)
        files.append(lod.glb)
    if not src.skinned:
        files += write_obj(out_lods[0], out, "lod0")
    if unity:
        files += write_ppmodel(out_lods, out, name, look, src, kind)
    times["write"] = time.perf_counter() - t
    t = time.perf_counter()
    say("Rendering previews…")
    files += preview_images(out_lods[0], out, look)
    files.append(wire_image(src, out_lods, meshes.high, out / "wire.png", look=look))
    times["preview"] = time.perf_counter() - t
    times["total"] = time.perf_counter() - t0
    clips = [a.get("name") for a in (src.doc or {}).get("animations", [])]
    info = {"name": name, "kind": kind, "look": look, "source": model.name,
            "source_tris": int(len(src.f)), "high_tris": meshes.high.tris,
            "lods": [{"file": f"lod{l.index}.glb", "tris": l.tris, "verts": int(len(l.atlas.v)),
                      "texture": int(l.bake.colour.shape[0]), "charts": l.atlas.charts,
                      "normal_map": l.bake.normal is not None} for l in out_lods],
            "skinned": src.skinned, "clips": clips,
            "bones": len(src.doc["skins"][0]["joints"]) if src.skinned else 0,
            "symmetrized": round(meshes.symmetric, 3), "notes": meshes.notes,
            "timings": {k: round(v, 2) for k, v in times.items()}, "created": time.time()}
    info["cache_key"] = key
    info["files"] = sorted({f.name for f in files} | {"info.json"})
    (out / "info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    files.append(out / "info.json")
    log.info("game model %s: %s in %.0fs", name, [l.tris for l in out_lods], times["total"])
    return GameModel(out, out_lods, info, files)


def _unchanged(out: Path, key: str) -> "GameModel | None":
    """The model_game in `out` when it was built from exactly these inputs (cache key in its
    info.json) and every file it wrote is still there."""
    try:
        info = json.loads((out / "info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    names = info.get("files") or []
    if info.get("cache_key") != key or not names or not all((out / n).exists() for n in names):
        return None
    return GameModel(out, [], info, [out / n for n in names])


# --------------------------------------------------------------------------- library assets

def source_model(asset) -> tuple[Path, str]:
    """(model file, kind) for a library character / item / prop (rigged.glb wins)."""
    folder = Path(asset.folder if hasattr(asset, "folder") else asset)
    kind = getattr(asset, "kind", None) or "prop"
    if kind in ("person",):
        kind = "character"
    extra = getattr(asset, "extra", {}) or {}
    if kind == "character" and (extra.get("body") == "animal" or getattr(asset, "kind", "") == "animal"):
        kind = "animal"
    if (folder / "character.json").exists() and kind not in ("character", "animal"):
        kind = "character"
    for f in ("rigged.glb", "model.glb"):
        if (folder / f).exists():
            return folder / f, kind
    raise ValueError(f"'{folder.name}' has no 3D model (model.glb) to make a game model from.")


def game_model_for(asset, progress=None, **options) -> GameModel:
    """Build <asset>/model_game for a library character, item or prop (look defaults to the
    project's: pixel for pixel-art houses)."""
    model, kind = source_model(asset)
    if "look" not in options:
        try:
            from .assets import current_project
            options["look"] = default_look(current_project().style)
        except Exception:
            options["look"] = "smooth"
    if "palette" not in options and options.get("look") == "pixel":
        try:
            from .assets import current_project
            pal = current_project().style.palette
            options["palette"] = None if pal == "auto" else pal
        except Exception:
            pass
    name = getattr(asset, "name", None) or Path(model).parent.name
    return make_game_model(model, Path(model).parent / FOLDER, kind=kind, name=name,
                           progress=progress, **options)


def has_game_model(asset) -> bool:
    folder = Path(asset.folder if hasattr(asset, "folder") else asset)
    return (folder / FOLDER / "lod0.glb").exists()
