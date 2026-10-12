"""A tactics battle map with height: the board shared by the three looks of the HD-2D demo
(real 3D, fake 3D pixel camera, isometric 2D).

    d = harbour(seed=0)        # stone tiers, stairs, a cobble quay, a wooden pier over water
    d.save("map.json")         # what the Godot scenes read
    tex = textures(d)          # {terrain: (top RGBA, side RGBA)} seamless pixel-art tiles

Cells: height in levels (0 = water level; one level = half a tile tall, like Triangle
Strategy's steps), a terrain name, and props (crates, barrels, torches, posts, fences).
Water cells are not walkable; a unit can climb `jump` levels per step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .maptiles import builtin_texture

TERRAINS = ("grass", "stone", "cobble", "planks", "dirt", "sand", "water")
WALKABLE = {"grass", "stone", "cobble", "planks", "dirt", "sand"}
LEVEL = 0.5          # one height level in tiles (world units: 1 tile = 1.0)


@dataclass
class Unit:
    name: str          # sprite set name, e.g. "dog", "knight"
    col: int
    row: int
    team: int = 0
    facing: str = "S"  # one of the 16 direction names (E, ESE, SE, ...)


@dataclass
class Diorama:
    cols: int
    rows: int
    height: np.ndarray                 # (rows, cols) int levels
    terrain: np.ndarray                # (rows, cols) str
    props: list = field(default_factory=list)    # [(kind, col, row)]
    units: list = field(default_factory=list)    # [Unit]
    water_level: float = 0.15          # world y of the water surface (levels * LEVEL)
    name: str = "map"

    def walkable(self, c: int, r: int) -> bool:
        return (0 <= c < self.cols and 0 <= r < self.rows
                and self.terrain[r, c] in WALKABLE
                and not any(p[1] == c and p[2] == r and p[0] in BLOCKING_PROPS for p in self.props))

    def to_dict(self) -> dict:
        return {
            "name": self.name, "cols": self.cols, "rows": self.rows, "level": LEVEL,
            "water_level": self.water_level,
            "height": self.height.astype(int).tolist(),
            "terrain": self.terrain.tolist(),
            "props": [{"kind": k, "col": c, "row": r} for k, c, r in self.props],
            "units": [vars(u) for u in self.units],
        }

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "Diorama":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["cols"], d["rows"], np.array(d["height"], int), np.array(d["terrain"], object),
                   [(p["kind"], p["col"], p["row"]) for p in d.get("props", [])],
                   [Unit(**u) for u in d.get("units", [])], d.get("water_level", 0.15),
                   d.get("name", "map"))


BLOCKING_PROPS = {"crate", "barrel", "post"}


def harbour(cols: int = 18, rows: int = 14, seed: int = 0) -> Diorama:
    """A harbour town like Triangle Strategy's pier map: a stone upper town (3-4 levels) with
    stairs down to a cobble quay (2), a wooden pier (1) out over the water, grass in the
    cracks, crates and barrels on the quay, torches and posts along the edges."""
    rng = np.random.default_rng(seed)
    h = np.zeros((rows, cols), int)
    t = np.full((rows, cols), "water", object)
    # upper town: the back rows, two tiers
    town = rows // 3
    h[:town, :] = 4
    t[:town, :] = "stone"
    h[:town - 2, cols // 2:] = 6
    # quay: a band in front of the town, on the left two thirds
    quay_end = int(cols * 0.62)
    h[town:town + 3, :quay_end] = 2
    t[town:town + 3, :quay_end] = "cobble"
    # stairs from the quay up to the town (2 -> 3 -> 4), three wide
    sx = quay_end // 2
    h[town - 1, sx:sx + 3] = 3
    t[town - 1, sx:sx + 3] = "stone"
    # stairs on the upper tier (4 -> 5 -> 6)
    ux = cols // 2 + 2
    h[town - 2, ux:ux + 2] = 5
    t[town - 2, ux:ux + 2] = "stone"
    # the pier: 3 wide, from the quay out over the water
    px = quay_end // 3
    h[town + 3:rows - 1, px:px + 3] = 1
    t[town + 3:rows - 1, px:px + 3] = "planks"
    # a side jetty off the pier
    jr = town + 3 + (rows - town - 4) // 2
    h[jr:jr + 2, px + 3:px + 8] = 1
    t[jr:jr + 2, px + 3:px + 8] = "planks"
    # grass in the cracks of the stone and quay
    for r in range(rows):
        for c in range(cols):
            if t[r, c] in ("stone", "cobble") and rng.random() < 0.12:
                t[r, c] = "grass"
    # a dirt yard in the upper-left town
    t[1:3, 1:4] = "dirt"

    props = []
    for c in range(0, quay_end, 3):                       # posts along the quay edge
        if not (px <= c < px + 3):
            props.append(("post", c, town + 2))
    for c in (2, 5, quay_end - 2):                        # crates and barrels on the quay
        props.append(("crate" if c % 2 else "barrel", c, town + 1))
    for r in range(town + 4, rows - 1, 2):                # torches on the pier sides
        props.append(("torch", px - 1, r))
        props.append(("torch", px + 3, r))
    props.append(("torch", sx - 1, town - 1))
    props.append(("torch", sx + 3, town - 1))
    props = [p for p in props if 0 <= p[1] < cols and 0 <= p[2] < rows]
    # torches hang over water off the pier: put them on the pier edge instead
    props = [(k, c, r) if t[r, c] != "water" else (k, min(max(c, px), px + 2), r) for k, c, r in props]

    units = [Unit("knight", px + 1, rows - 2, 0, "N"), Unit("dog", px, rows - 3, 0, "N"),
             Unit("knight", sx + 1, town - 3, 1, "S"), Unit("dog", sx + 4, 1, 1, "S")]
    return Diorama(cols, rows, h, t, props, units, name="harbour")


# --------------------------------------------------------------------------- textures

def _planks(size: int = 32, seed: int = 0) -> np.ndarray:
    """Weathered wooden boards running one way, with dark gaps and nail dots (seamless)."""
    rng = np.random.default_rng(seed + 7)
    ramp = np.array([[74, 48, 30], [102, 68, 42], [128, 88, 56], [152, 108, 70], [176, 132, 90]])
    board = max(4, size // 6)
    img = np.zeros((size, size, 3), np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    for b in range(0, size, board):
        tone = rng.uniform(1.2, 3.2)
        grain = np.sin((xx / size) * 2 * np.pi * rng.integers(2, 5) + rng.uniform(0, 6)) * 0.5
        rows = (yy >= b) & (yy < b + board)
        img[rows] = (tone + grain + rng.normal(0, 0.25, (size, size)))[rows][..., None]
    idx = np.clip(np.rint(img[..., 0]), 0, 4).astype(int)
    out = ramp[idx].astype(np.uint8)
    out[(yy % board) == board - 1] = ramp[0]                       # gaps between boards
    for b in range(0, size, board):                                # nails
        for x in (size // 4, 3 * size // 4):
            out[min(size - 1, b + board // 2), x] = (52, 40, 34)
    return np.dstack([out, np.full((size, size), 255, np.uint8)])


def _cobble(size: int = 32, seed: int = 0) -> np.ndarray:
    """Rounded paving stones with dark mortar (seamless)."""
    rng = np.random.default_rng(seed + 11)
    n = 9
    pts = rng.random((n, 2)) * size
    pts = np.concatenate([pts + [dx, dy] for dx in (-size, 0, size) for dy in (-size, 0, size)])
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.sqrt((xx[..., None] - pts[:, 0]) ** 2 + (yy[..., None] - pts[:, 1]) ** 2)
    srt = np.sort(d, -1)
    cell = np.argmin(d, -1) % n
    edge = (srt[..., 1] - srt[..., 0]) < 1.6
    ramp = np.array([[66, 64, 70], [96, 94, 102], [122, 120, 128], [148, 146, 152], [176, 174, 180]])
    tone = (rng.integers(1, 4, n)[cell] + (srt[..., 0] < size / 8)).clip(0, 4)
    out = ramp[tone].astype(np.uint8)
    out[edge] = ramp[0]
    return np.dstack([out, np.full((size, size), 255, np.uint8)])


def _side(top: np.ndarray, name: str, size: int) -> np.ndarray:
    """The texture for a block's sides: stone walls under stone/cobble/grass, dirt under dirt,
    wooden beams under planks."""
    if name == "planks":
        return np.ascontiguousarray(np.rot90(_planks(size, 3)))
    if name in ("dirt", "sand"):
        return builtin_texture("dirt", size, 1)
    wall = _cobble(size, 5)
    wall[..., :3] = (wall[..., :3].astype(np.float32) * 0.82).astype(np.uint8)
    return wall


def textures(d: Diorama | None = None, size: int = 32, seed: int = 0) -> dict:
    """{terrain: (top RGBA, side RGBA)} for every terrain the map uses (all, if no map)."""
    names = sorted(set(d.terrain.ravel())) if d is not None else list(TERRAINS)
    out = {}
    for name in names:
        if name == "planks":
            top = _planks(size, seed)
        elif name == "cobble":
            top = _cobble(size, seed)
        else:
            top = builtin_texture(name, size, seed)
        out[name] = (top, _side(top, name, size))
    return out


def write_textures(d: Diorama, folder, size: int = 32, seed: int = 0) -> dict:
    """Save <terrain>_top.png / <terrain>_side.png; returns {terrain: {"top": name, "side": name}}."""
    from PIL import Image
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = {}
    for name, (top, side) in textures(d, size, seed).items():
        Image.fromarray(top).save(folder / f"{name}_top.png")
        Image.fromarray(side).save(folder / f"{name}_side.png")
        names[name] = {"top": f"{name}_top.png", "side": f"{name}_side.png"}
    return names


def path_cost(d: Diorama, start: tuple[int, int], move: int = 4, jump: int = 2) -> dict:
    """Cells reachable from start within `move` steps, climbing at most `jump` levels per step
    (dropping any distance up to 2*jump): {(col, row): steps}. The movement-range overlay."""
    best = {start: 0}
    frontier = [start]
    while frontier:
        nxt = []
        for c, r in frontier:
            for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c + dc, r + dr)
                if not d.walkable(*n):
                    continue
                dh = d.height[n[1], n[0]] - d.height[r, c]
                if dh > jump or -dh > 2 * jump:
                    continue
                cost = best[(c, r)] + 1
                if cost <= move and cost < best.get(n, 99):
                    best[n] = cost
                    nxt.append(n)
        frontier = nxt
    return best
