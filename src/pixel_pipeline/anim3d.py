"""Animated 3D models -> animated direction sprites (walk_E, walk_SE ... for Godot).

Give a rigged, animated glTF (.glb/.gltf): e.g. a Mixamo character (Mixamo -> FBX ->
Blender or Godot -> glTF), or an auto-rigged model from Tripo/Meshy. Every animation is
played, sampled at `frames` evenly spaced times, and every pose is rendered from every
direction with the turnaround renderer: one scale, one ground point and one palette for
all of them. Animations can also come from separate files (one Mixamo download per
animation): the first file gives the model, later files add their animations to the
matching bones by name.

The glTF reader here is small and dependency-free: node hierarchy, skins (linear blend
skinning), animations (LINEAR, STEP and CUBICSPLINE translation/rotation/scale) and the
base colour (baseColorTexture, baseColorFactor, COLOR_0, alpha MASK/BLEND cut-off). Morph
targets are ignored. Draco/meshopt-compressed files must be re-exported uncompressed.
"""
from __future__ import annotations

import base64
import io
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .aseprite import write_aseprite
from .export import pack_sheet, write_sheet
from .godot import write_sprite_frames
from .pipeline import Sprite
from .settings import Settings
from .turnaround import direction_names, render_points, turnaround_sprites, union_crop

COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
              5125: np.uint32, 5126: np.float32}
WIDTHS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
UNSUPPORTED = {"KHR_draco_mesh_compression": "Draco compression",
               "EXT_meshopt_compression": "meshopt compression",
               "KHR_mesh_quantization": "mesh quantization"}
GENERIC_CLIP = re.compile(r"mixamo|^take|layer|^armature|^action|^animation|^default|^anim\d*$|^$",
                          re.IGNORECASE)


# --------------------------------------------------------------------------- glTF reading

class Gltf:
    """The parts of a .glb/.gltf file needed to pose and colour it."""

    def __init__(self, path):
        self.path = Path(path)
        data = self.path.read_bytes()
        bin_chunk = None
        if data[:4] == b"glTF":
            off, doc = 12, None
            while off + 8 <= len(data):
                length, kind = struct.unpack_from("<II", data, off)
                chunk = data[off + 8: off + 8 + length]
                if kind == 0x4E4F534A:
                    doc = json.loads(chunk)
                elif kind == 0x004E4942:
                    bin_chunk = chunk
                off += 8 + length
            if doc is None:
                raise ValueError(f"{self.path.name}: no JSON chunk")
        else:
            doc = json.loads(data)
        self.doc = doc
        needed = [UNSUPPORTED[e] for e in doc.get("extensionsRequired", []) if e in UNSUPPORTED]
        if needed:
            raise ValueError(f"{self.path.name} uses {', '.join(needed)}: re-export it without "
                             "(Blender: File > Export > glTF, untick Compression)")
        self.buffers = [self._uri(b.get("uri"), bin_chunk) for b in doc.get("buffers", [])]
        self.nodes = doc.get("nodes", [])

    def _uri(self, uri, bin_chunk) -> bytes:
        if uri is None:
            return bin_chunk or b""
        if uri.startswith("data:"):
            return base64.b64decode(uri.split(",", 1)[1])
        return (self.path.parent / uri).read_bytes()

    def view(self, i: int) -> bytes:
        v = self.doc["bufferViews"][i]
        start = v.get("byteOffset", 0)
        return self.buffers[v["buffer"]][start:start + v["byteLength"]]

    def accessor(self, i: int) -> np.ndarray:
        """(count, components) float64 (or int64 for unnormalized integers)."""
        a = self.doc["accessors"][i]
        dt = np.dtype(COMPONENTS[a["componentType"]]).newbyteorder("<")
        n, count = WIDTHS[a["type"]], a["count"]
        if "bufferView" in a:
            v = self.doc["bufferViews"][a["bufferView"]]
            stride = v.get("byteStride") or n * dt.itemsize
            raw = self.buffers[v["buffer"]]
            start = v.get("byteOffset", 0) + a.get("byteOffset", 0)
            arr = np.ndarray((count, n), dt, raw, start, (stride, dt.itemsize)).copy()
        else:
            arr = np.zeros((count, n), dt)
        sparse = a.get("sparse")
        if sparse:
            ix, vals = sparse["indices"], sparse["values"]
            idt = np.dtype(COMPONENTS[ix["componentType"]]).newbyteorder("<")
            rows = np.frombuffer(self.view(ix["bufferView"]), idt, sparse["count"],
                                 ix.get("byteOffset", 0))
            arr[rows] = np.frombuffer(self.view(vals["bufferView"]), dt, sparse["count"] * n,
                                      vals.get("byteOffset", 0)).reshape(-1, n)
        if a.get("normalized") and dt.kind in "iu":
            peak = float(np.iinfo(dt).max)
            return np.maximum(arr.astype(np.float64) / peak, -1.0)
        return arr.astype(np.float64 if dt.kind == "f" else np.int64)

    def image(self, texture: int) -> np.ndarray | None:
        """RGBA array of a texture, or None if it can't be decoded."""
        tex = self.doc.get("textures", [])[texture]
        src = tex.get("source")
        for ext in ("EXT_texture_webp", "KHR_texture_basisu", "MSFT_texture_dds"):
            src = tex.get("extensions", {}).get(ext, {}).get("source", src) if src is None else src
        if src is None:
            return None
        img = self.doc["images"][src]
        try:
            data = (self.view(img["bufferView"]) if "bufferView" in img
                    else self._uri(img.get("uri"), None))
            return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA"))
        except Exception:      # KTX2 / DDS and broken images: fall back to the colour factor
            return None


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = np.asarray(q, np.float64) / max(np.linalg.norm(q), 1e-12)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _trs(t, r, s) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _quat_matrix(r) * np.asarray(s, np.float64)[None, :]
    m[:3, 3] = t
    return m


def _slerp(a, b, u):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = float(a @ b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        q = a + u * (b - a)
    else:
        th = math.acos(min(d, 1.0))
        q = (math.sin((1 - u) * th) * a + math.sin(u * th) * b) / math.sin(th)
    return q / max(np.linalg.norm(q), 1e-12)


def _srgb_to_linear(c):
    c = np.asarray(c, np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


@dataclass
class Channel:
    node: int
    path: str                 # translation / rotation / scale
    times: np.ndarray
    values: np.ndarray        # (keys, n), or (keys, 3, n) in/value/out for CUBICSPLINE
    interpolation: str

    def at(self, t: float) -> np.ndarray:
        times, vals = self.times, self.values
        cubic = self.interpolation == "CUBICSPLINE"
        if t <= times[0] or len(times) == 1:
            out = vals[0, 1] if cubic else vals[0]
        elif t >= times[-1]:
            out = vals[-1, 1] if cubic else vals[-1]
        else:
            k = int(np.searchsorted(times, t, side="right")) - 1
            t0, t1 = times[k], times[k + 1]
            u = (t - t0) / max(t1 - t0, 1e-12)
            if self.interpolation == "STEP":
                out = vals[k]
            elif cubic:
                dt = t1 - t0
                p0, m0, p1, m1 = vals[k, 1], vals[k, 2] * dt, vals[k + 1, 1], vals[k + 1, 0] * dt
                out = ((2 * u ** 3 - 3 * u ** 2 + 1) * p0 + (u ** 3 - 2 * u ** 2 + u) * m0
                       + (-2 * u ** 3 + 3 * u ** 2) * p1 + (u ** 3 - u ** 2) * m1)
            elif self.path == "rotation":
                out = _slerp(vals[k], vals[k + 1], u)
            else:
                out = vals[k] + u * (vals[k + 1] - vals[k])
        if self.path == "rotation":
            out = out / max(np.linalg.norm(out), 1e-12)
        return out


@dataclass
class Clip:
    name: str
    start: float
    end: float
    channels: list[Channel]
    source: str = ""
    loop: bool = True

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class _Primitive:
    node: int
    skin: int | None
    positions: np.ndarray        # (n, 3) mesh space
    faces: np.ndarray            # (m, 3)
    joints: np.ndarray | None    # (n, k) node indices
    weights: np.ndarray | None   # (n, k), rows sum to 1
    colours: np.ndarray | None   # (n, 4) linear COLOR_0
    uv: np.ndarray | None
    texture: np.ndarray | None   # RGBA sRGB
    factor: np.ndarray           # linear RGBA
    alpha_cut: float | None      # drop samples below this alpha


def _triangles(indices: np.ndarray, mode: int) -> np.ndarray:
    if mode == 4:
        return indices.reshape(-1, 3)
    if mode == 5:   # strip
        tri = np.stack([indices[:-2], indices[1:-1], indices[2:]], 1)
        tri[1::2] = tri[1::2][:, [1, 0, 2]]
        return tri
    if mode == 6:   # fan
        return np.stack([np.full(len(indices) - 2, indices[0]), indices[1:-1], indices[2:]], 1)
    return np.zeros((0, 3), np.int64)


def clip_name(name: str | None, source: Path, count: int) -> str:
    """A tidy animation name: the glTF name, or the file name for Mixamo-style 'mixamo.com',
    'Take 001', 'Armature|...' names when the file holds one animation."""
    raw = name or ""
    if count == 1 and GENERIC_CLIP.search(raw.split("|")[-1] if "|" in raw else raw):
        raw = source.stem
    tidy = re.sub(r"[^0-9a-zA-Z]+", "_", raw.split("|")[-1] if "|" in raw else raw).strip("_")
    return tidy.lower() or source.stem.lower()


class Rig:
    """A glTF model that can be posed at any time of any of its animations."""

    def __init__(self, path):
        g = Gltf(path)
        self.gltf = g
        self.path = Path(path)
        n = len(g.nodes)
        self.names = [nd.get("name", f"node{i}") for i, nd in enumerate(g.nodes)]
        self.parent = np.full(n, -1)
        for i, nd in enumerate(g.nodes):
            for c in nd.get("children", []):
                self.parent[c] = i
        self.rest = []
        for nd in g.nodes:
            if "matrix" in nd:
                m = np.asarray(nd["matrix"], np.float64).reshape(4, 4).T
                t = m[:3, 3].copy()
                s = np.linalg.norm(m[:3, :3], axis=0)
                r = _matrix_quat(m[:3, :3] / np.maximum(s, 1e-12))
                self.rest.append((t, r, s))
            else:
                self.rest.append((np.asarray(nd.get("translation", [0, 0, 0]), np.float64),
                                  np.asarray(nd.get("rotation", [0, 0, 0, 1]), np.float64),
                                  np.asarray(nd.get("scale", [1, 1, 1]), np.float64)))
        scenes = g.doc.get("scenes")
        roots = (scenes[g.doc.get("scene", 0)].get("nodes", []) if scenes
                 else [i for i in range(n) if self.parent[i] < 0])
        self.order: list[int] = []            # parents before children, scene nodes only
        stack = list(reversed(roots))
        while stack:
            i = stack.pop()
            self.order.append(i)
            stack += list(reversed(g.nodes[i].get("children", [])))
        self.skins = []
        for sk in g.doc.get("skins", []):
            joints = np.asarray(sk["joints"])
            ibm = (g.accessor(sk["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
                   if "inverseBindMatrices" in sk else np.tile(np.eye(4), (len(joints), 1, 1)))
            self.skins.append((joints, ibm))
        self.primitives = [p for i in self.order if "mesh" in g.nodes[i]
                           for p in self._primitives(i)]
        if not self.primitives:
            raise ValueError(f"No triangles found in {self.path.name}")
        self.clips: list[Clip] = []
        self.add_animations(self.path, g)

    def _primitives(self, node: int):
        g = self.gltf
        nd = g.nodes[node]
        skin = nd.get("skin")
        for prim in g.doc["meshes"][nd["mesh"]]["primitives"]:
            at = prim["attributes"]
            mode = prim.get("mode", 4)
            if "POSITION" not in at or mode not in (4, 5, 6):
                continue
            pos = g.accessor(at["POSITION"])
            idx = (g.accessor(prim["indices"]).reshape(-1) if "indices" in prim
                   else np.arange(len(pos)))
            faces = _triangles(idx.astype(np.int64), mode)
            if not len(faces):
                continue
            joints = weights = None
            if skin is not None and "JOINTS_0" in at and "WEIGHTS_0" in at:
                sets = [k for k in range(8) if f"JOINTS_{k}" in at and f"WEIGHTS_{k}" in at]
                local = np.concatenate([g.accessor(at[f"JOINTS_{k}"]) for k in sets], 1).astype(np.int64)
                weights = np.concatenate([g.accessor(at[f"WEIGHTS_{k}"]) for k in sets], 1)
                total = weights.sum(1, keepdims=True)
                weights = np.where(total > 0, weights / np.maximum(total, 1e-12), 0)
                joints = np.clip(local, 0, len(self.skins[skin][0]) - 1)   # into the skin's joints
            colours = None
            if "COLOR_0" in at:
                c = g.accessor(at["COLOR_0"]).astype(np.float64)
                colours = np.concatenate([c, np.ones((len(c), 1))], 1) if c.shape[1] == 3 else c
            mat = g.doc.get("materials", [])[prim["material"]] if "material" in prim else {}
            pbr = mat.get("pbrMetallicRoughness", {})
            factor = np.asarray(pbr.get("baseColorFactor", [1, 1, 1, 1]), np.float64)
            texture = uv = None
            info = pbr.get("baseColorTexture")
            if info is not None:
                texture = g.image(info["index"])
                key = f"TEXCOORD_{info.get('texCoord', 0)}"
                if texture is not None and key in at:
                    uv = g.accessor(at[key])
                else:
                    texture = None
            mode_a = mat.get("alphaMode", "OPAQUE")
            cut = mat.get("alphaCutoff", 0.5) if mode_a == "MASK" else 0.5 if mode_a == "BLEND" else None
            yield _Primitive(node, skin if joints is not None else None, pos, faces, joints,
                             weights, colours, uv, texture, factor, cut)

    def add_animations(self, path, gltf: Gltf | None = None) -> list[Clip]:
        """Add the animations in a file; other files' channels go to bones with the same name."""
        g = gltf or Gltf(path)
        same = g is self.gltf
        by_name = {nm: i for i, nm in enumerate(self.names)}
        anims = g.doc.get("animations", [])
        added = []
        for a in anims:
            channels = []
            for ch in a.get("channels", []):
                target = ch.get("target", {})
                path_ = target.get("path")
                if "node" not in target or path_ not in ("translation", "rotation", "scale"):
                    continue
                node = target["node"]
                if not same:
                    node = by_name.get(g.nodes[node].get("name", f"node{node}"))
                    if node is None:
                        continue
                smp = a["samplers"][ch["sampler"]]
                interp = smp.get("interpolation", "LINEAR")
                times = g.accessor(smp["input"]).reshape(-1).astype(np.float64)
                vals = g.accessor(smp["output"]).astype(np.float64)
                if interp == "CUBICSPLINE":
                    vals = vals.reshape(len(times), 3, -1)
                channels.append(Channel(node, path_, times, vals, interp))
            if not channels:
                continue
            start = min(float(c.times[0]) for c in channels)
            end = max(float(c.times[-1]) for c in channels)
            clip = Clip(clip_name(a.get("name"), Path(path), len(anims)), start, end, channels,
                        source=a.get("name") or "")
            taken = {c.name for c in self.clips}
            base, k = clip.name, 2
            while clip.name in taken:
                clip.name, k = f"{base}_{k}", k + 1
            loop = a.get("extras", {}).get("loop") if isinstance(a.get("extras"), dict) else None
            clip.loop = bool(loop) if isinstance(loop, bool) else self._loops(clip)
            self.clips.append(clip)
            added.append(clip)
        return added

    # ---- posing

    def worlds(self, clip: Clip | None = None, t: float = 0.0) -> np.ndarray:
        """(nodes, 4, 4) world matrices at time t of a clip (None = the file's rest pose)."""
        local = [list(trs) for trs in self.rest]
        if clip is not None:
            slot = {"translation": 0, "rotation": 1, "scale": 2}
            for ch in clip.channels:
                local[ch.node][slot[ch.path]] = ch.at(t)
        out = np.tile(np.eye(4), (len(local), 1, 1))
        for i in self.order:
            m = _trs(*local[i])
            p = self.parent[i]
            out[i] = out[p] @ m if p >= 0 else m
        return out

    def vertices(self, worlds: np.ndarray) -> list[np.ndarray]:
        """Posed world positions of every primitive's vertices."""
        out = []
        for p in self.primitives:
            v = np.concatenate([p.positions, np.ones((len(p.positions), 1))], 1)
            if p.skin is None:
                out.append((v @ worlds[p.node].T)[:, :3])
                continue
            joints, ibm = self.skins[p.skin]
            mats = worlds[joints] @ ibm                                # (J, 4, 4)
            blend = np.einsum("nk,nkij->nij", p.weights, mats[p.joints])
            out.append(np.einsum("nij,nj->ni", blend, v)[:, :3])
        return out

    def centroid(self, clip: Clip | None = None, t: float = 0.0) -> np.ndarray:
        return np.concatenate(self.vertices(self.worlds(clip, t))).mean(0)

    def _loops(self, clip: Clip) -> bool:
        """Does the clip end in the pose it starts with (walk, run, idle)?"""
        a = np.concatenate(self.vertices(self.worlds(clip, clip.start)))
        b = np.concatenate(self.vertices(self.worlds(clip, clip.end)))
        size = max(float(np.ptp(np.concatenate(self.vertices(self.worlds())), 0).max()), 1e-9)
        moved = (b - a) - (b - a).mean(0)          # ignore root motion
        return float(np.linalg.norm(moved, axis=1).max()) < 0.05 * size


def _matrix_quat(m: np.ndarray) -> np.ndarray:
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return np.asarray(q, np.float64)


# --------------------------------------------------------------------------- surface points

@dataclass
class Surface:
    """Points spread over the rest pose, re-posed every frame (so colours never flicker)."""
    prim: list[tuple[np.ndarray, np.ndarray]]   # per primitive: (face ids, barycentrics)
    colours: np.ndarray                          # (points, 3) uint8 sRGB


def sample_surface(rig: Rig, points: int, seed: int = 0) -> Surface:
    rng = np.random.default_rng(seed)
    verts = rig.vertices(rig.worlds())
    areas = []
    for p, v in zip(rig.primitives, verts):
        tri = v[p.faces]
        areas.append(np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1))
    total = sum(float(a.sum()) for a in areas)
    if total <= 0:
        raise ValueError(f"{rig.path.name}: the model has no area")
    prim, colours = [], []
    for p, a in zip(rig.primitives, areas):
        count = int(round(points * float(a.sum()) / total))
        if count == 0:
            prim.append((np.zeros(0, np.int64), np.zeros((0, 3))))
            continue
        face = rng.choice(len(p.faces), size=count, p=a / a.sum())
        r1, r2 = rng.random(count), rng.random(count)
        s = np.sqrt(r1)
        bary = np.stack([1 - s, s * (1 - r2), s * r2], 1)
        col = np.tile(p.factor, (count, 1))
        if p.texture is not None:
            uv = (p.uv[p.faces[face]] * bary[..., None]).sum(1) % 1.0
            th, tw = p.texture.shape[:2]
            x = np.clip((uv[:, 0] * tw).astype(np.int64), 0, tw - 1)
            y = np.clip((uv[:, 1] * th).astype(np.int64), 0, th - 1)   # glTF uv: top-left origin
            texel = p.texture[y, x].astype(np.float64) / 255.0
            col = col * np.concatenate([_srgb_to_linear(texel[:, :3]), texel[:, 3:]], 1)
        if p.colours is not None:
            col = col * (p.colours[p.faces[face]] * bary[..., None]).sum(1)
        keep = col[:, 3] >= p.alpha_cut if p.alpha_cut is not None else slice(None)
        prim.append((face[keep], bary[keep]))
        colours.append((_linear_to_srgb(col[keep, :3]) * 255 + 0.5).astype(np.uint8))
    return Surface(prim, np.concatenate(colours) if colours else np.zeros((0, 3), np.uint8))


def surface_pose(rig: Rig, surface: Surface, worlds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(points, normals) of the sampled surface in one pose."""
    pts, nrm = [], []
    for p, v, (face, bary) in zip(rig.primitives, rig.vertices(worlds), surface.prim):
        if not len(face):
            continue
        corner = p.faces[face]
        pts.append(v[corner[:, 0]] * bary[:, :1] + v[corner[:, 1]] * bary[:, 1:2]
                   + v[corner[:, 2]] * bary[:, 2:])
        tri = v[p.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nrm.append((n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))[face])
    return np.concatenate(pts), np.concatenate(nrm)


def _normalised(pts: np.ndarray) -> np.ndarray:
    """Stand on y = 0, centred on the y axis, 1 unit tall."""
    lo, hi = pts.min(0), pts.max(0)
    centre = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    return (pts - centre) / max(float(hi[1] - lo[1]), 1e-9)


def transfer_colours(dst: np.ndarray, src: np.ndarray, src_colours: np.ndarray,
                     res: int = 128) -> np.ndarray:
    """Colour points `dst` from the nearest coloured points `src` of the same shape (both are
    normalised to stand 1 unit tall first, so scale and position don't matter). For a model
    rigged elsewhere (Mixamo drops vertex colours) coloured from the original."""
    a, b = _normalised(np.asarray(dst, np.float64)), _normalised(np.asarray(src, np.float64))
    lo = np.minimum(a.min(0), b.min(0))
    cell = float((np.maximum(a.max(0), b.max(0)) - lo).max()) / (res - 1) or 1.0

    def voxel(p):
        v = np.clip(((p - lo) / cell).astype(np.int64), 0, res - 1)
        return (v[:, 0] * res + v[:, 1]) * res + v[:, 2]
    total = np.zeros((res ** 3, 3))
    count = np.zeros(res ** 3)
    ix = voxel(b)
    np.add.at(total, ix, src_colours[:, :3].astype(np.float64))
    np.add.at(count, ix, 1)
    total, count = total.reshape(res, res, res, 3), count.reshape(res, res, res)
    want = np.zeros((res ** 3), bool)
    want[voxel(a)] = True
    want = want.reshape(res, res, res)
    for _ in range(24):                     # grow coloured voxels into empty ones nearby
        empty = want & (count == 0)
        if not empty.any():
            break
        grow_t, grow_c = np.zeros_like(total), np.zeros_like(count)
        for axis in range(3):
            for step in (1, -1):
                grow_t += np.roll(total * (count > 0)[..., None], step, axis)
                grow_c += np.roll((count > 0).astype(np.float64), step, axis)
        mean = np.where(count[..., None] > 0, total / np.maximum(count, 1)[..., None], 0)
        grown = np.zeros_like(total)
        for axis in range(3):
            for step in (1, -1):
                grown += np.roll(mean, step, axis)
        fill = (count == 0) & (grow_c > 0)
        total[fill] = grown[fill] / grow_c[fill][:, None]
        count[fill] = 1
    left = want & (count == 0)              # far from anything coloured: the usual colour
    total[left] = np.median(src_colours[:, :3], 0)
    count[left] = 1
    mean = total / np.maximum(count, 1)[..., None]
    return np.clip(mean.reshape(-1, 3)[voxel(a)] + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- writing

class GlbWriter:
    """Just enough of a glTF writer to build models in code (tests, the demo mannequin)."""

    def __init__(self):
        self.doc = {"asset": {"version": "2.0"}, "buffers": [{}], "bufferViews": [], "accessors": []}
        self.bin = bytearray()

    def view(self, data: bytes, stride=None) -> int:
        while len(self.bin) % 4:
            self.bin.append(0)
        v = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if stride:
            v["byteStride"] = stride
        self.bin += data
        self.doc["bufferViews"].append(v)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, arr, kind, ctype=5126, normalized=False) -> int:
        dt = {5126: "<f4", 5121: "u1", 5123: "<u2", 5125: "<u4"}[ctype]
        arr = np.ascontiguousarray(arr, dt)
        a = {"bufferView": self.view(arr.tobytes()), "componentType": ctype,
             "count": len(arr), "type": kind}
        if normalized:
            a["normalized"] = True
        if kind in ("SCALAR", "VEC3") and ctype == 5126:      # required for POSITION and times
            flat = arr.reshape(len(arr), -1)
            a["min"], a["max"] = flat.min(0).tolist(), flat.max(0).tolist()
        self.doc["accessors"].append(a)
        return len(self.doc["accessors"]) - 1

    def save(self, path):
        self.doc["buffers"][0]["byteLength"] = len(self.bin)
        js = json.dumps(self.doc).encode()
        js += b" " * (-len(js) % 4)
        binary = bytes(self.bin) + b"\0" * (-len(self.bin) % 4)
        body = (struct.pack("<II", len(js), 0x4E4F534A) + js
                + struct.pack("<II", len(binary), 0x004E4942) + binary)
        Path(path).write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
        return Path(path)


# --------------------------------------------------------------------------- animations

@dataclass
class Action:
    """One animation as sprites: `frames` poses, each seen from every direction."""
    name: str
    loop: bool
    frame_ms: int


def load_rig(paths) -> Rig:
    """The first file's model with the animations of every file."""
    paths = [Path(p) for p in paths]
    rig = Rig(paths[0])
    for p in paths[1:]:
        rig.add_animations(p)
    return rig


def select_clips(rig: Rig, animations: list[str] | None = None) -> list[Clip]:
    """Pick (and rename) clips: 'walk' or 'walk=Armature|mixamo.com|Layer0' or 'walk=2'."""
    if not rig.clips:
        raise ValueError(f"{rig.path.name} has no animations (it needs a skeleton and animations;"
                         " see docs: Animated directions)")
    if not animations:
        return list(rig.clips)
    out = []
    for item in animations:
        new, _, old = item.partition("=")
        old = old or new
        found = next((c for c in rig.clips if old in (c.name, c.source)), None)
        if found is None and old.isdigit() and int(old) < len(rig.clips):
            found = rig.clips[int(old)]
        if found is None:
            raise ValueError(f"No animation '{old}' (have: {', '.join(c.name for c in rig.clips)})")
        out.append(Clip(new if "=" in item else found.name, found.start, found.end,
                        found.channels, found.source, found.loop))
    return out


def clip_times(clip: Clip, frames: int) -> list[float]:
    """Evenly spaced times: a loop skips its last pose (it is the first one again)."""
    if frames < 1:
        raise ValueError("frames must be at least 1")
    if clip.duration <= 0 or frames == 1:
        return [clip.start] * frames if clip.duration <= 0 else [clip.start]
    steps = frames if clip.loop else frames - 1
    return [clip.start + clip.duration * i / steps for i in range(frames)]


def animated_poses(rig: Rig, clips: list[Clip], frames: int = 8, points: int = 1_000_000,
                   seed: int = 0, in_place: bool = True, colours_from=None):
    """Every clip's poses as (points, normals), centred like load_model_points.
    in_place: remove steady root motion (a walk that moves forward walks on the spot).
    colours_from: a coloured model of the same shape to take colours from."""
    surface = sample_surface(rig, points, seed)
    rest_pts, _ = surface_pose(rig, surface, rig.worlds())
    if colours_from is not None:
        from .turnaround import load_model_points
        src, src_cols, _ = load_model_points(colours_from, max(200_000, points // 2), seed=seed)
        surface.colours = transfer_colours(rest_pts, src, src_cols)
    lo, hi = rest_pts.min(0), rest_pts.max(0)
    centre = np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[2] + hi[2]) / 2])
    poses: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for clip in clips:
        times = clip_times(clip, frames)
        drift = np.zeros(3)
        if in_place and clip.duration > 0:
            moved = rig.centroid(clip, clip.end) - rig.centroid(clip, clip.start)
            drift = moved * np.array([1.0, 0.0, 1.0]) / clip.duration
        row = []
        for t in times:
            pts, nrm = surface_pose(rig, surface, rig.worlds(clip, t))
            row.append(((pts - centre - drift * (t - clip.start)).astype(np.float32),
                        nrm.astype(np.float32)))
        poses.append(row)
    ground = min(float(p[:, 1].min()) for row in poses for p, _ in row)
    poses = [[(p - np.float32([0.0, ground, 0.0]), n) for p, n in row] for row in poses]
    return poses, surface.colours


@dataclass
class AnimatedSet:
    """sprites[action][direction][frame], one palette and one size."""
    actions: list[Action]
    directions: list[str]
    sprites: list[list[list[Sprite]]] = field(repr=False)

    @property
    def frames(self) -> int:
        return len(self.sprites[0][0])

    def flat(self) -> list[Sprite]:
        return [s for a in self.sprites for d in a for s in d]


def animated_turnaround(paths, settings: Settings, directions: int = 8, frames: int = 8,
                        animations: list[str] | None = None, *, elevation: float = 30.0,
                        yaw: float = 0.0, light: float = 0.35, in_place: bool = True,
                        size: int | None = None, points: int | None = None,
                        seed: int = 0, colours_from=None,
                        normals: bool = False) -> tuple[AnimatedSet, str]:
    """Rigged + animated glTF file(s) -> AnimatedSet and a description of what was read.
    colours_from: the original coloured model, when rigging (Mixamo) dropped its colours.
    normals: also give every sprite a normal map (sprite.normal, see normals.py)."""
    rig = load_rig(paths)
    clips = select_clips(rig, animations)
    names = direction_names(directions)
    size = size or int(min(512, max(192, 4 * max(settings.width, settings.height or 0))))
    poses, colours = animated_poses(rig, clips, frames, points or 6 * size * size, seed, in_place,
                                    colours_from)
    flat = [p for row in poses for p in row]
    views = render_points(flat, colours, directions, elevation, size, yaw, light, normals)
    nviews = None
    if normals:
        views, nviews = views                                          # [dir][pose] each
    order = [(d, a * frames + f) for a in range(len(clips)) for d in range(len(names))
             for f in range(frames)]
    cropped = [views[d][i] for d, i in order]
    if normals:
        cropped, nviews = union_crop(cropped, settings.margin, settings.alpha_threshold,
                                     [nviews[d][i] for d, i in order])
    else:
        cropped = union_crop(cropped, settings.margin, settings.alpha_threshold)
    del views
    sprites = turnaround_sprites(cropped, settings, nviews)
    per = len(names) * frames
    nested = [[sprites[a * per + d * frames: a * per + (d + 1) * frames] for d in range(len(names))]
              for a in range(len(clips))]
    actions = [Action(c.name, c.loop, max(16, int(round(1000 * c.duration
                                                         / max(1, frames if c.loop else frames - 1)))))
               for c in clips]
    what = (f"{Path(paths[0]).name}: {', '.join(c.name for c in clips)} "
            f"({frames} frames each) rendered at {elevation:g} degrees")
    return AnimatedSet(actions, names, nested), what


# --------------------------------------------------------------------------- files

def godot_action_script(directions: list[str], actions: list[str], frames_name: str) -> str:
    dirs = ", ".join(f'&"{n}"' for n in directions)
    acts = ", ".join(f'&"{n}"' for n in actions)
    south = directions.index("S")
    return f"""extends AnimatedSprite2D
## Plays an action facing a direction. Set sprite_frames to {frames_name}, then call
## face(velocity) whenever the unit moves and act(&"walk") / act(&"idle") to switch.
## Directions run clockwise from facing right, matching Vector2.angle().

const DIRECTIONS: Array[StringName] = [{dirs}]
const ACTIONS: Array[StringName] = [{acts}]

@export var action: StringName = ACTIONS[0]
var facing := {south}


func _ready() -> void:
\t_show()


func face(direction: Vector2) -> void:
\tif not direction.is_zero_approx():
\t\tfacing = wrapi(roundi(direction.angle() / TAU * DIRECTIONS.size()), 0, DIRECTIONS.size())
\t_show()


func act(new_action: StringName) -> void:
\taction = new_action
\t_show()


func _show() -> void:
\tvar anim := StringName("%s_%s" % [action, DIRECTIONS[facing]])
\tif anim == animation and is_playing():
\t\treturn
\tvar same_action := String(animation).begins_with(String(action) + "_")
\tvar keep_frame := frame
\tvar keep_progress := frame_progress
\tplay(anim)
\tif same_action:  # turning mid-walk keeps the step
\t\tset_frame_and_progress(keep_frame, keep_progress)
"""


def write_animated(anim: AnimatedSet, out_png, aseprite: bool = False,
                   godot: bool = False, normals: bool = False) -> list[Path]:
    """Sheet (a row per action and direction, a column per frame) + JSON (a tag per
    action_direction); optionally .aseprite (same tags) and Godot SpriteFrames (an animation
    per action_direction, e.g. walk_SE) + a script that plays action + facing.
    normals: also <sheet>_n.png, the normal maps laid out exactly like the sheet."""
    out = Path(out_png)
    frames = [s.rgba for s in anim.flat()]
    f = anim.frames
    tags, names = [], []
    for a in anim.actions:
        for d in anim.directions:
            first = len(names)
            names += [f"{a.name}_{d} {i}" for i in range(f)]
            tags.append((f"{a.name}_{d}", first, first + f - 1, a))
    out_json = write_sheet(frames, out, f, anim.actions[0].frame_ms, None, names)
    meta = json.loads(out_json.read_text(encoding="utf-8"))
    for i, fr in enumerate(meta["frames"]):
        fr["duration"] = anim.actions[i // (f * len(anim.directions))].frame_ms
    meta["meta"]["frameTags"] = [{"name": n, "from": lo, "to": hi, "direction": "forward"}
                                 for n, lo, hi, _ in tags]
    out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written = [out, out_json]
    if normals:
        from .normals import write_normal_sheet
        written.append(write_normal_sheet(anim.flat(), out, f))
    if aseprite:
        ase = out.with_suffix(".aseprite")
        durations = [a.frame_ms for a in anim.actions for _ in range(f * len(anim.directions))]
        write_aseprite(ase, frames, anim.sprites[0][0][0].palette, durations=durations,
                       tags=[(n, lo, hi) for n, lo, hi, _ in tags])
        written.append(ase)
    if godot:
        _, rects = pack_sheet(frames, f)
        anims = {n: {"frames": list(range(lo, hi + 1)), "duration_ms": a.frame_ms, "loop": a.loop}
                 for n, lo, hi, a in tags}
        tres = write_sprite_frames(out.with_suffix(".tres"), out.name, rects, animations=anims)
        gd = out.with_name(out.stem + "_facing.gd")
        gd.write_text(godot_action_script(anim.directions, [a.name for a in anim.actions], tres.name),
                      encoding="utf-8")
        written += [tres, gd]
    return written


def write_animated_preview(anim: AnimatedSet, out_gif, scale: int = 3, columns: int = 8,
                           background=(28, 32, 46), loops: int = 2) -> Path:
    """GIF: every direction side by side (wrapping at `columns`), playing each action in turn."""
    out_gif = Path(out_gif)
    h, w = anim.sprites[0][0][0].rgba.shape[:2]
    n = len(anim.directions)
    cols = min(columns, n)
    rows = math.ceil(n / cols)
    images, durations = [], []
    for a, action in zip(anim.sprites, anim.actions):
        for _ in range(loops):
            for f in range(anim.frames):
                canvas = Image.new("RGBA", (cols * w, rows * h), background + (255,))
                for d in range(n):
                    canvas.alpha_composite(Image.fromarray(a[d][f].rgba, "RGBA"),
                                           ((d % cols) * w, (d // cols) * h))
                images.append(canvas.convert("RGB").resize((cols * w * scale, rows * h * scale),
                                                           Image.NEAREST))
                durations.append(action.frame_ms)
    images[0].save(out_gif, save_all=True, append_images=images[1:], duration=durations, loop=0)
    return out_gif
