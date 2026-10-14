"""Map assets: battle maps with height you make once and reuse in many games.

    doc = new_map("forest", 20, 16, seed=3)      # a MapDoc: Diorama + team start cells
    asset = save_map(doc, "Old Forest")          # Projects/<project>/maps/old-forest/
    export_map(asset, "godot-3d", "out/forest")  # also godot-tilemap, godot-iso, tiled, json
    d = load_diorama(asset, blue=["knight"], red=["wolf"])   # for the Game tab / make_hd2d

A map asset folder holds asset.json (extra: template, size, seed), map.json (Diorama.to_dict +
"spawns" + "template" + "seed"), tex/ (the project's tile size), preview.png (top-down),
thumb.png and iso_preview.png (a quick painter's-order isometric render).

Templates are procedural and seeded: flat, harbour, forest, castle, snowfield, village. Their
props are named kinds (tree, rock, house, wall, well, fence + the harbour's crate, barrel, post,
torch); every renderer here draws unknown kinds as a plain box.

The HD-2D board (hd2d.board.gd) draws the new prop kinds through res://map/props3d.gd
(`install_board_extras(project)` writes it); without that file it draws boxes.

Generated tilesets (tilegen.py): `map_look(doc, project)` = the project's saved top-down sets
(MapDoc.tiles picks one per terrain / cliff / fence / wall: {key: set folder, "" = built-in},
newest otherwise). The editor, previews, godot-tilemap, tiled, json (colour.png) and godot-3d
(top textures) all draw with it; terrains no set covers keep the built-in textures.
"""
from __future__ import annotations

import json
import shutil
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import assets
from .diorama import (BLOCKING_PROPS, LEVEL, TERRAINS, WALKABLE, Diorama, Unit, harbour,
                      path_cost, place_units, textures, write_textures)

TEMPLATES = ("flat", "harbour", "forest", "castle", "snowfield", "village")
DEFAULT_SIZES = {"flat": (16, 12), "harbour": (18, 14), "forest": (20, 16), "castle": (22, 18),
                 "snowfield": (20, 16), "village": (20, 16)}
MIN_SIZE = {"castle": (14, 12), "harbour": (12, 10)}
PROP_KINDS = ("tree", "rock", "house", "wall", "well", "fence", "battlement", "crate", "barrel", "post", "torch",
              "brazier", "lantern", "crystal")   # the last four light the HD-2D board
TARGETS = ("godot-3d", "godot-tilemap", "godot-iso", "tiled", "json")
MIN_LEVEL, MAX_LEVEL = 0, 12
JUMP = 2
TEAM_RGB = ((70, 140, 255), (255, 80, 70))


# --------------------------------------------------------------------------- the document

@dataclass
class MapDoc:
    """A map being made: the Diorama (height, terrain, props; no units) + team start cells."""
    diorama: Diorama
    spawns: list = field(default_factory=list)       # [(team, col, row)]
    template: str = "flat"
    seed: int = 0
    tiles: dict = field(default_factory=dict)        # generated tileset per key (map_look)

    @property
    def cols(self) -> int:
        return self.diorama.cols

    @property
    def rows(self) -> int:
        return self.diorama.rows

    def copy(self) -> "MapDoc":
        d = self.diorama
        nd = Diorama(d.cols, d.rows, d.height.copy(), d.terrain.copy(), list(d.props), [],
                     d.water_level, d.name, d.sea)
        return MapDoc(nd, list(self.spawns), self.template, self.seed, dict(self.tiles))

    def to_dict(self) -> dict:
        out = self.diorama.to_dict()
        out["spawns"] = [{"team": t, "col": c, "row": r} for t, c, r in self.spawns]
        out["template"] = self.template
        out["seed"] = self.seed
        if self.tiles:
            out["tiles"] = dict(self.tiles)
        return out

    @classmethod
    def from_dict(cls, data: dict) -> "MapDoc":
        d = Diorama(data["cols"], data["rows"], np.array(data["height"], int),
                    np.array(data["terrain"], object),
                    [(p["kind"], p["col"], p["row"]) for p in data.get("props", [])], [],
                    data.get("water_level", 0.15), data.get("name", "map"), bool(data.get("sea", True)))
        spawns = [(int(s["team"]), int(s["col"]), int(s["row"])) for s in data.get("spawns", [])]
        if not spawns:      # an old map.json with units: their cells are the starts
            spawns = [(int(u.get("team", 0)), int(u["col"]), int(u["row"])) for u in data.get("units", [])]
        return cls(d, spawns, data.get("template", "flat"), int(data.get("seed", 0)),
                   {str(k): str(v) for k, v in (data.get("tiles") or {}).items()})

    def prop_at(self, c: int, r: int):
        for p in self.diorama.props:
            if p[1] == c and p[2] == r:
                return p[0]
        return None

    def free(self, c: int, r: int) -> bool:
        """Walkable and nothing standing there."""
        return self.diorama.walkable(c, r) and self.prop_at(c, r) is None


# --------------------------------------------------------------------------- template helpers

def _noise(rows: int, cols: int, cells: int, rng) -> np.ndarray:
    from .maptiles import _periodic_noise
    n = max(rows, cols)
    return _periodic_noise(n, max(2, cells), rng)[:rows, :cols]


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def _limit_slopes(h: np.ndarray, keep: np.ndarray | None = None, step: int = JUMP) -> np.ndarray:
    """Lower cells until no neighbour is more than `step` levels below (gentle, walkable land).
    Cells in `keep` (e.g. water) are never changed but still count as neighbours."""
    h = h.copy()
    for _ in range(64):
        p = np.pad(h, 1, mode="edge")
        lowest = np.minimum.reduce([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        new = np.minimum(h, lowest + step)
        if keep is not None:
            new[keep] = h[keep]
        if (new == h).all():
            break
        h = new
    return h


def _components(doc: MapDoc) -> list[set]:
    """Groups of free cells connected by steps of at most JUMP levels (either way)."""
    d = doc.diorama
    seen, out = set(), []
    for r in range(d.rows):
        for c in range(d.cols):
            if (c, r) in seen or not doc.free(c, r):
                continue
            comp, stack = set(), [(c, r)]
            seen.add((c, r))
            while stack:
                x, y = stack.pop()
                comp.add((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (x + dx, y + dy)
                    if n in seen or not doc.free(*n):
                        continue
                    if abs(int(d.height[n[1], n[0]]) - int(d.height[y, x])) > JUMP:
                        continue
                    seen.add(n)
                    stack.append(n)
            out.append(comp)
    return sorted(out, key=len, reverse=True)


def auto_spawns(doc: MapDoc, per_team: int = 4) -> list:
    """Blue starts at the front (high rows), red at the back, on one connected area, spread
    out (no two starts side by side)."""
    comps = _components(doc)
    if not comps:
        return []
    cells = comps[0]
    mid = doc.cols / 2
    out, taken = [], []

    def pick(order, team):
        n = 0
        for c, r in order:
            if n >= per_team:
                break
            if all(abs(c - tc) + abs(r - tr) > 1 for tc, tr in taken):
                taken.append((c, r))
                out.append((team, c, r))
                n += 1
    pick(sorted(cells, key=lambda p: (-p[1], abs(p[0] - mid))), 0)
    pick(sorted(cells, key=lambda p: (p[1], abs(p[0] - mid))), 1)
    return out


def _doc(cols, rows, h, t, props, template, seed, sea=False, name=None, spawns=None, per_team=4):
    d = Diorama(cols, rows, h.astype(int), t, props, [], name=name or template, sea=sea)
    doc = MapDoc(d, [], template, seed)
    # nothing may stand on water, and only one prop per cell
    seen, keep = set(), []
    for k, c, r in d.props:
        if 0 <= c < cols and 0 <= r < rows and (c, r) not in seen and t[r, c] != "water":
            seen.add((c, r))
            keep.append((k, int(c), int(r)))
    d.props = keep
    doc.spawns = spawns if spawns is not None else auto_spawns(doc, per_team)
    return doc


def _clear_around(props: list, cells, radius: int = 1) -> list:
    near = {(c + dx, r + dy) for c, r in cells for dx in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)}
    return [p for p in props if (p[1], p[2]) not in near]


def _respawn_clear(doc: MapDoc, per_team: int = 4) -> MapDoc:
    """Spawns, then no blocking prop right next to a start (units must get out)."""
    doc.spawns = auto_spawns(doc, per_team)
    doc.diorama.props = _clear_around(doc.diorama.props, [(c, r) for _, c, r in doc.spawns], 0)
    return doc


# --------------------------------------------------------------------------- templates

def flat_map(cols: int = 16, rows: int = 12, seed: int = 0, terrain: str = "grass", level: int = 2) -> MapDoc:
    """An empty map: one terrain at one height (above the water)."""
    h = np.full((rows, cols), level, int)
    t = np.full((rows, cols), terrain, object)
    return _doc(cols, rows, h, t, [], "flat", seed)


def harbour_map(cols: int = 18, rows: int = 14, seed: int = 0) -> MapDoc:
    d = harbour(cols, rows, seed)
    spawns = [(u.team, u.col, u.row) for u in d.units]
    doc = _doc(cols, rows, d.height, d.terrain, d.props, "harbour", seed, sea=True, spawns=spawns)
    doc.diorama.water_level = d.water_level
    # the harbour has 2 starts per team: add a couple more, near the first ones
    more = auto_spawns(doc, 4)
    have = {(c, r) for _, c, r in doc.spawns}
    for team in (0, 1):
        n = sum(1 for s in doc.spawns if s[0] == team)
        for s in more:
            if n >= 4:
                break
            if s[0] == team and (s[1], s[2]) not in have:
                doc.spawns.append(s)
                have.add((s[1], s[2]))
                n += 1
    return doc


def _meander(n: int, length: int, centre: float, amp: float, rng) -> np.ndarray:
    ph, f = rng.uniform(0, 6.28), rng.uniform(1.0, 1.8)
    t = np.arange(length) / max(1, length - 1)
    return np.clip(np.round(centre + amp * np.sin(t * np.pi * f + ph)), 0, n - 1).astype(int)


def forest_map(cols: int = 20, rows: int = 16, seed: int = 0) -> MapDoc:
    """Grass on gentle hills (base 1, +0..3 levels), a stream in a little valley with a wooden
    bridge, dirt trails (east-west and north-south), clustered trees and a few rocks."""
    rng = np.random.default_rng(seed)
    hills = _norm(0.7 * _noise(rows, cols, 3, rng) + 0.3 * _noise(rows, cols, 6, rng))
    h = 1 + np.round(hills * 3).astype(int)
    t = np.full((rows, cols), "grass", object)
    yy, xx = np.mgrid[0:rows, 0:cols]
    # the stream: top to bottom, 2 wide, carves a valley
    sx = _meander(cols, rows, cols * rng.uniform(0.3, 0.45), cols * 0.12, rng)
    dist = np.abs(xx - sx[:, None]).astype(float)
    water = (xx >= sx[:, None]) & (xx <= sx[:, None] + 1)
    dist = np.where(xx > sx[:, None], np.maximum(0, xx - sx[:, None] - 1), dist)
    h = np.minimum(h, 1 + (dist // 2).astype(int))
    h[water] = 0
    t[water] = "water"
    bank = ~water & (dist <= 1)
    t[bank & (rng.random((rows, cols)) < 0.6)] = "sand"
    # trails: east-west across the stream (bridge), north-south on the far bank
    py = _meander(rows, cols, rows * 0.55, rows * 0.12, rng)
    trail = np.zeros((rows, cols), bool)
    trail[py, np.arange(cols)] = True
    tx = _meander(cols, rows, cols * 0.72, cols * 0.08, rng)
    trail[np.arange(rows), tx] = True
    for r in range(rows - 1):                           # keep the north-south trail connected
        a, b = sorted((tx[r], tx[r + 1]))
        trail[r, a:b + 1] = True
    for c in range(cols - 1):
        a, b = sorted((py[c], py[c + 1]))
        trail[a:b + 1, c] = True
    bridge = trail & water
    t[trail & ~water] = "dirt"
    t[bridge] = "planks"
    h[bridge] = 1
    h = _limit_slopes(h, keep=water)
    # trails are easy walking: no step over one level along them
    h = np.where(trail & ~water, np.maximum(1, h), h)
    h = _limit_slopes(h, keep=water, step=JUMP)
    # trees: clustered by noise, not on trails or banks
    dense = _norm(_noise(rows, cols, 4, rng))
    near_trail = np.zeros_like(trail)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            near_trail |= np.roll(np.roll(trail, dy, 0), dx, 1)
    ok = (t == "grass") & ~near_trail
    trees = ok & (rng.random((rows, cols)) < 0.1 + 0.62 * dense ** 2)
    props = [("tree", int(c), int(r)) for r, c in zip(*np.nonzero(trees))]
    rocks = np.argwhere(ok & ~trees)
    for i in rng.choice(len(rocks), size=min(len(rocks), max(2, cols * rows // 70)), replace=False):
        r, c = rocks[i]
        props.append(("rock", int(c), int(r)))
    doc = _doc(cols, rows, h, t, props, "forest", seed)
    return _respawn_clear(doc)


def castle_map(cols: int = 22, rows: int = 18, seed: int = 0) -> MapDoc:
    """A stone castle at the back: walls (6 levels) with towers (8) at the corners and either
    side of the gate, a cobble courtyard (2) with stairs up to the wall walk, a moat with a
    wooden bridge to the gate, and a grass field in front with a dirt road, trees and rocks."""
    cols, rows = max(cols, MIN_SIZE["castle"][0]), max(rows, MIN_SIZE["castle"][1])
    rng = np.random.default_rng(seed)
    h = 1 + (_norm(_noise(rows, cols, 4, rng)) > 0.62).astype(int)     # low grassy mounds
    t = np.full((rows, cols), "grass", object)
    x0, x1 = 3, cols - 4                       # castle rectangle (walls on its border)
    y0, y1 = 2, int(rows * 0.62)
    # moat: a ring of water one cell outside the walls
    t[y0 - 1:y1 + 2, x0 - 1:x1 + 2] = "water"
    h[y0 - 1:y1 + 2, x0 - 1:x1 + 2] = 0
    # courtyard
    t[y0:y1 + 1, x0:x1 + 1] = "cobble"
    h[y0:y1 + 1, x0:x1 + 1] = 2
    for r in range(y0 + 1, y1):                # grass in the cracks
        for c in range(x0 + 1, x1):
            if rng.random() < 0.12:
                t[r, c] = "grass"
    # walls
    for r in range(y0, y1 + 1):
        for c in range(x0, x1 + 1):
            if r in (y0, y1) or c in (x0, x1):
                h[r, c] = 6
                t[r, c] = "stone"
    # corner towers 2x2
    for cx, cy in ((x0, y0), (x1 - 1, y0), (x0, y1 - 1), (x1 - 1, y1 - 1)):
        h[cy:cy + 2, cx:cx + 2] = 8
        t[cy:cy + 2, cx:cx + 2] = "stone"
    # gate in the front wall, towers either side
    gx = (x0 + x1) // 2
    h[y1, gx - 1:gx + 2] = 2
    t[y1, gx - 1:gx + 2] = "cobble"
    for c in (gx - 2, gx + 2):
        h[y1 - 1:y1 + 1, c] = 8
        t[y1 - 1:y1 + 1, c] = "stone"
    # bridge over the moat, a road to the front edge
    h[y1 + 1, gx - 1:gx + 2] = 1
    t[y1 + 1, gx - 1:gx + 2] = "planks"
    road_x = _meander(cols, rows - y1 - 2, gx, 1.5, rng)
    for i, r in enumerate(range(y1 + 2, rows)):
        c = int(road_x[i])
        t[r, max(0, c - 1):c + 2] = "dirt"
        h[r, max(0, c - 1):c + 2] = 1
    # stairs from the courtyard up to the wall walk: 3, 4, 5 then the wall (6), both sides
    for side, c in ((0, x0 + 1), (1, x1 - 1)):
        for k, lvl in enumerate((5, 4, 3)):
            r = y0 + 2 + k
            h[r, c] = lvl
            t[r, c] = "stone"
    # a keep at the back middle of the courtyard: a raised stone hall (4) with a step
    kx0, kx1 = gx - 2, gx + 2
    h[y0 + 1:y0 + 3, kx0:kx1 + 1] = 4
    t[y0 + 1:y0 + 3, kx0:kx1 + 1] = "stone"
    h[y0 + 3, gx - 1:gx + 2] = 3
    t[y0 + 3, gx - 1:gx + 2] = "stone"
    props = [("battlement", c, r) for r in range(y0, y1 + 1) for c in range(x0, x1 + 1)
             if h[r, c] >= 6 and battlement_sides(h, c, r)]
    props += [("well", gx + 3, y1 - 3), ("crate", x0 + 2, y1 - 1), ("barrel", x0 + 3, y1 - 1),
             ("barrel", x1 - 2, y1 - 1), ("torch", gx - 2, y1 + 2), ("torch", gx + 2, y1 + 2),
             ("torch", gx - 2, y0 + 3), ("torch", gx + 2, y0 + 3)]
    field_rows = range(y1 + 3, rows)
    for _ in range(max(4, cols * len(field_rows) // 8)):
        r, c = int(rng.choice(list(field_rows) or [rows - 1])), int(rng.integers(0, cols))
        if t[r, c] == "grass":
            props.append(("tree" if rng.random() < 0.7 else "rock", c, r))
    for c in (0, cols - 1):                    # trees at the sides of the moat
        for r in range(y0, y1, 3):
            if t[r, c] == "grass":
                props.append(("tree", c, r))
    doc = _doc(cols, rows, h, t, props, "castle", seed)
    # blue arrives along the road; red holds the courtyard
    front = sorted([(c, r) for r in range(rows) for c in range(cols) if doc.free(c, r) and r > y1 + 1],
                   key=lambda p: (-p[1], abs(p[0] - gx)))
    inside = sorted([(c, r) for r in range(y0 + 1, y1) for c in range(x0 + 1, x1) if doc.free(c, r)
                     and h[r, c] == 2], key=lambda p: (abs(p[0] - gx) + abs(p[1] - (y0 + y1) / 2)))
    doc.spawns = _spread(front, 0, 4) + _spread(inside, 1, 4)
    return doc


def _spread(cells, team, n):
    out, taken = [], []
    for c, r in cells:
        if len(out) >= n:
            break
        if all(abs(c - tc) + abs(r - tr) > 1 for tc, tr in taken):
            taken.append((c, r))
            out.append((team, c, r))
    return out


def snowfield_map(cols: int = 20, rows: int = 16, seed: int = 0) -> MapDoc:
    """Snow drifts (1-3 levels), rocky stone ridges up to 5, a frozen-over pond (water), snowy
    pines and rocks, and a trodden dirt trail from front to back."""
    rng = np.random.default_rng(seed)
    drift = _norm(_noise(rows, cols, 3, rng) * 0.6 + _noise(rows, cols, 7, rng) * 0.4)
    h = 1 + np.round(drift * 2).astype(int)
    t = np.full((rows, cols), "snow", object)
    ridge = _norm(_noise(rows, cols, 4, rng))
    rocky = ridge > 0.78
    t[rocky] = "stone"
    h[rocky] += 2
    yy, xx = np.mgrid[0:rows, 0:cols]
    cx, cy = cols * rng.uniform(0.25, 0.4), rows * rng.uniform(0.3, 0.6)
    pond = ((xx - cx) / max(2, cols * 0.14)) ** 2 + ((yy - cy) / max(2, rows * 0.16)) ** 2 < 1
    t[pond] = "water"
    h[pond] = 0
    shore = ~pond & (((xx - cx) / max(2, cols * 0.14 + 1)) ** 2 + ((yy - cy) / max(2, rows * 0.16 + 1)) ** 2 < 1)
    h[shore] = np.minimum(h[shore], 1)
    tx = _meander(cols, rows, cols * 0.62, cols * 0.1, rng)
    trail = np.zeros((rows, cols), bool)
    for r in range(rows):
        a, b = sorted((tx[r], tx[min(rows - 1, r + 1)]))
        trail[r, a:b + 1] = True
    trail &= ~pond
    t[trail] = "dirt"
    h[trail] = np.minimum(h[trail], 2)
    h = _limit_slopes(h, keep=pond)
    ok = (t == "snow") & ~trail & ~shore
    near = np.zeros_like(trail)
    for dx in (-1, 0, 1):
        near |= np.roll(trail, dx, 1)
    ok &= ~near
    dense = _norm(_noise(rows, cols, 3, rng))
    trees = ok & (rng.random((rows, cols)) < 0.04 + 0.3 * dense ** 3)
    props = [("tree", int(c), int(r)) for r, c in zip(*np.nonzero(trees))]
    stones = np.argwhere((ok | (t == "stone")) & ~trees)
    for i in rng.choice(len(stones), size=min(len(stones), max(3, cols * rows // 50)), replace=False):
        r, c = stones[i]
        props.append(("rock", int(c), int(r)))
    doc = _doc(cols, rows, h, t, props, "snowfield", seed)
    return _respawn_clear(doc)


def village_map(cols: int = 20, rows: int = 16, seed: int = 0) -> MapDoc:
    """A village: a dirt square with a well in the middle, cobble roads to the edges, cottages
    along the roads (blocked cells), fenced gardens, a terraced hill corner and trees."""
    rng = np.random.default_rng(seed)
    h = np.full((rows, cols), 2, int)
    t = np.full((rows, cols), "grass", object)
    # a terraced hill in the back-right corner
    hx, hy = int(cols * 0.68), int(rows * 0.34)
    h[:hy, hx:] = 4
    h[:max(1, hy - 2), min(cols - 1, hx + 2):] = 5
    cx, cy = cols // 2, rows // 2
    sq = (slice(cy - 2, cy + 3), slice(cx - 3, cx + 4))
    t[sq] = "dirt"
    h[sq] = 2
    road = np.zeros((rows, cols), bool)
    road[:, cx] = True
    road[cy, :] = True
    t[road] = "cobble"
    h[road] = np.where(h[road] > 2, 3, 2)          # roads climb the hill in one step
    t[sq] = np.where(road[sq], "cobble", "dirt")
    h = _limit_slopes(h)
    props = [("well", cx + 1, cy - 1)]
    # cottages: one cell off the road, every other cell, outside the square
    in_sq = np.zeros((rows, cols), bool)
    in_sq[cy - 3:cy + 4, cx - 4:cx + 5] = True
    houses = []
    for r in range(rows):
        for off in (-2, 2):
            c = cx + off
            if 0 <= c < cols and not in_sq[r, c] and r % 2 == 0:
                houses.append((c, r))
    for c in range(cols):
        for off in (-2, 2):
            r = cy + off
            if 0 <= r < rows and not in_sq[r, c] and c % 2 == 1:
                houses.append((c, r))
    houses = [(c, r) for c, r in houses if not road[r, c] and rng.random() < 0.8]
    props += [("house", c, r) for c, r in houses]
    # fenced gardens (3x2 rings of fence, dirt inside) in free grass
    for _ in range(3):
        for _try in range(20):
            gx, gy = int(rng.integers(1, cols - 4)), int(rng.integers(1, rows - 3))
            cells = [(gx + i, gy + j) for i in range(4) for j in range(3)]
            if all(t[y, x] == "grass" and not in_sq[y, x] and (x, y) not in houses
                   and abs(int(h[y, x]) - int(h[gy, gx])) == 0 for x, y in cells):
                for x, y in cells:
                    if x in (gx, gx + 3) or y in (gy, gy + 2):
                        props.append(("fence", x, y))
                    else:
                        t[y, x] = "dirt"
                break
    taken = {(p[1], p[2]) for p in props}
    for r, c in np.argwhere((t == "grass") & (rng.random((rows, cols)) < 0.07)):
        if (int(c), int(r)) not in taken and not in_sq[r, c] and not road[r, c]:
            props.append(("tree", int(c), int(r)))
    for c, r in houses[:4]:                        # crates and barrels by a few doors
        for dc in (-1, 1):
            if 0 <= c + dc < cols and t[r, c + dc] == "grass" and (c + dc, r) not in taken:
                props.append(("barrel" if (c + r) % 2 else "crate", c + dc, r))
                taken.add((c + dc, r))
                break
    # nothing on the roads: they are how everyone gets around
    props = [p for p in props if not road[p[2], p[1]]]
    doc = _doc(cols, rows, h, t, props, "village", seed)
    return _respawn_clear(doc)


_BUILDERS = {"flat": flat_map, "harbour": harbour_map, "forest": forest_map, "castle": castle_map,
             "snowfield": snowfield_map, "village": village_map}


def new_map(template: str = "flat", cols: int | None = None, rows: int | None = None,
            seed: int = 0) -> MapDoc:
    """A new map from a template (sizes default per template; at least 8 x 8)."""
    if template not in _BUILDERS:
        raise ValueError(f"Unknown map template '{template}' (use {', '.join(TEMPLATES)})")
    dc, dr = DEFAULT_SIZES[template]
    mc, mr = MIN_SIZE.get(template, (8, 8))
    cols, rows = max(mc, int(cols or dc)), max(mr, int(rows or dr))
    doc = _BUILDERS[template](cols, rows, seed)
    doc.template, doc.seed = template, seed
    doc.diorama.name = template
    return doc


def validate(doc: MapDoc) -> list[str]:
    """Problems that would break a game on this map (empty list = fine)."""
    out = []
    d = doc.diorama
    if d.height.shape != (d.rows, d.cols) or d.terrain.shape != (d.rows, d.cols):
        out.append("height/terrain arrays don't match the size")
    for name in set(d.terrain.ravel()):
        if name not in TERRAINS:
            out.append(f"unknown terrain {name}")
    teams = {t for t, _, _ in doc.spawns}
    for team, c, r in doc.spawns:
        if not doc.free(c, r):
            out.append(f"start {c},{r} (team {team}) is not a free walkable cell")
    if len(doc.spawns) != len({(c, r) for _, c, r in doc.spawns}):
        out.append("two starts on one cell")
    if teams >= {0, 1}:
        b = next((c, r) for t, c, r in doc.spawns if t == 0)
        reach = path_cost(d, b, move=10 ** 6, jump=JUMP)
        for team, c, r in doc.spawns:
            if (c, r) not in reach:
                out.append(f"start {c},{r} (team {team}) can't be reached from blue's start")
    else:
        out.append("needs starts for both teams")
    return out


# --------------------------------------------------------------------------- editing

class MapEdit:
    """Edit operations with undo/redo (one undo step per begin()). Used by the map editor;
    plain Python so tests drive it directly."""

    def __init__(self, doc: MapDoc, limit: int = 100):
        self.doc = doc
        self.undo_stack: list[MapDoc] = []
        self.redo_stack: list[MapDoc] = []
        self.limit = limit

    def begin(self) -> None:
        """Remember the map before a change (call once per click/stroke)."""
        self.undo_stack.append(self.doc.copy())
        del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.doc.copy())
        self._set(self.undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.doc.copy())
        self._set(self.redo_stack.pop())
        return True

    def _set(self, doc: MapDoc) -> None:
        # keep the same MapDoc object (views hold it): copy the state in
        self.doc.diorama, self.doc.spawns = doc.diorama, doc.spawns
        self.doc.template, self.doc.seed = doc.template, doc.seed

    def cells(self, c: int, r: int, brush: int = 1) -> list[tuple[int, int]]:
        k = max(1, int(brush)) - 1
        d = self.doc.diorama
        return [(x, y) for y in range(r - k, r + k + 1) for x in range(c - k, c + k + 1)
                if 0 <= x < d.cols and 0 <= y < d.rows]

    def raise_cells(self, c: int, r: int, amount: int = 1, brush: int = 1) -> None:
        h = self.doc.diorama.height
        for x, y in self.cells(c, r, brush):
            h[y, x] = int(np.clip(h[y, x] + amount, MIN_LEVEL, MAX_LEVEL))

    def lower_cells(self, c: int, r: int, amount: int = 1, brush: int = 1) -> None:
        self.raise_cells(c, r, -amount, brush)

    def set_height(self, c: int, r: int, level: int, brush: int = 1) -> None:
        h = self.doc.diorama.height
        for x, y in self.cells(c, r, brush):
            h[y, x] = int(np.clip(level, MIN_LEVEL, MAX_LEVEL))

    def paint(self, c: int, r: int, terrain: str, brush: int = 1) -> None:
        if terrain not in TERRAINS:
            raise ValueError(f"Unknown terrain '{terrain}'")
        for x, y in self.cells(c, r, brush):
            self.doc.diorama.terrain[y, x] = terrain
        if terrain == "water":
            self._drop_on_water()

    def fill(self, c: int, r: int, terrain: str) -> int:
        """Flood-fill the connected area of the same terrain (and height); returns cells."""
        d = self.doc.diorama
        if not (0 <= c < d.cols and 0 <= r < d.rows):
            return 0
        old, lvl = d.terrain[r, c], d.height[r, c]
        if old == terrain:
            return 0
        stack, n = [(c, r)], 0
        seen = {(c, r)}
        while stack:
            x, y = stack.pop()
            d.terrain[y, x] = terrain
            n += 1
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if (0 <= nx < d.cols and 0 <= ny < d.rows and (nx, ny) not in seen
                        and d.terrain[ny, nx] == old and d.height[ny, nx] == lvl):
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        if terrain == "water":
            self._drop_on_water()
        return n

    def _drop_on_water(self) -> None:
        d = self.doc.diorama
        d.props = [p for p in d.props if d.terrain[p[2], p[1]] != "water"]
        self.doc.spawns = [s for s in self.doc.spawns if d.terrain[s[2], s[1]] != "water"]

    def put_prop(self, c: int, r: int, kind: str) -> None:
        d = self.doc.diorama
        if not (0 <= c < d.cols and 0 <= r < d.rows) or d.terrain[r, c] == "water":
            return
        d.props = [p for p in d.props if (p[1], p[2]) != (c, r)] + [(kind, c, r)]
        self.doc.spawns = [s for s in self.doc.spawns if (s[1], s[2]) != (c, r)]

    def erase_prop(self, c: int, r: int) -> None:
        d = self.doc.diorama
        d.props = [p for p in d.props if (p[1], p[2]) != (c, r)]

    def set_spawn(self, c: int, r: int, team: int) -> None:
        d = self.doc.diorama
        if not (0 <= c < d.cols and 0 <= r < d.rows):
            return
        self.doc.spawns = [s for s in self.doc.spawns if (s[1], s[2]) != (c, r)] + [(int(team), c, r)]
        self.erase_prop(c, r)

    def clear_spawn(self, c: int, r: int) -> None:
        self.doc.spawns = [s for s in self.doc.spawns if (s[1], s[2]) != (c, r)]

    def resize(self, cols: int, rows: int, terrain: str = "grass", level: int = 2) -> None:
        d = self.doc.diorama
        h = np.full((rows, cols), level, int)
        t = np.full((rows, cols), terrain, object)
        rc, cc = min(rows, d.rows), min(cols, d.cols)
        h[:rc, :cc] = d.height[:rc, :cc]
        t[:rc, :cc] = d.terrain[:rc, :cc]
        d.cols, d.rows, d.height, d.terrain = cols, rows, h, t
        d.props = [p for p in d.props if p[1] < cols and p[2] < rows]
        self.doc.spawns = [s for s in self.doc.spawns if s[1] < cols and s[2] < rows]


# --------------------------------------------------------------------------- textures + sprites

_TEX: dict = {}


def terrain_textures(size: int = 16, seed: int = 0) -> dict:
    """{terrain: (top, side)} for every terrain at `size` px (cached)."""
    key = (size, seed)
    if key not in _TEX:
        tex = textures(None, size, seed)
        if "water" in tex:
            top = tex["water"][0]
            side = top.copy()
            side[..., :3] = (side[..., :3].astype(np.float32) * 0.6).astype(np.uint8)
            tex["water"] = (top, side)
        _TEX[key] = tex
    return _TEX[key]


def _ramp_shade(mask: np.ndarray, ramp, cx: float, cy: float, rad: float, bias: float = 0.0) -> np.ndarray:
    """Fill mask with a colour ramp lit from the upper left (pixel-art banding)."""
    hgt, w = mask.shape
    y, x = np.mgrid[0:hgt, 0:w]
    light = 0.55 - ((x - cx) + (y - cy)) / (2.2 * max(1.0, rad)) + bias
    idx = np.clip((light * len(ramp)).astype(int), 0, len(ramp) - 1)
    out = np.zeros((hgt, w, 4), np.uint8)
    out[mask, :3] = np.asarray(ramp)[idx[mask]]
    out[mask, 3] = 255
    return out


def _outline_rgba(img: np.ndarray, colour=(28, 22, 30)) -> np.ndarray:
    """A 1px dark outline around the opaque pixels (in the transparent ones)."""
    m = img[..., 3] > 0
    p = np.pad(m, 1)
    ring = (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]) & ~m
    out = img.copy()
    out[ring] = (*colour, 255)
    return out


def _ell(w, h, cx, cy, rx, ry):
    y, x = np.mgrid[0:h, 0:w]
    return ((x + 0.5 - cx) / max(0.5, rx)) ** 2 + ((y + 0.5 - cy) / max(0.5, ry)) ** 2 <= 1


GREENS = [(30, 62, 38), (44, 92, 46), (66, 124, 54), (98, 156, 66), (140, 186, 84)]
PINES = [(22, 52, 46), (30, 74, 58), (44, 98, 70), (66, 124, 84)]
GREYS = [(64, 62, 70), (92, 90, 98), (122, 120, 128), (156, 154, 160), (188, 186, 192)]


def tree_sprite(w: int, pine: bool = False, snow: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """An isometric tree (round or pine, optionally snowy): (RGBA, anchor = the cell centre)."""
    sw, sh = max(10, int(w * 0.8)) // 2 * 2, max(14, int(w * 1.35))
    img = np.zeros((sh, sw, 4), np.uint8)
    cx = sw / 2
    tw = max(2, w // 10)
    trunk_h = max(3, int(w * 0.3))
    tx0 = int(cx - tw / 2)
    img[sh - trunk_h - 1:sh - 1, tx0:tx0 + tw] = (92, 60, 36, 255)
    img[sh - trunk_h - 1:sh - 1, tx0 + tw - 1] = (64, 42, 28, 255)
    if pine:
        tiers = 3
        top = 1
        bottom = sh - trunk_h + 1
        y, x = np.mgrid[0:sh, 0:sw]
        mask = np.zeros((sh, sw), bool)
        for k in range(tiers):
            t0 = top + (bottom - top) * k / (tiers + 0.6)
            t1 = t0 + (bottom - top) / (tiers - 0.4)
            half = (y - t0) / (t1 - t0) * (sw / 2 - 1) * (0.7 + 0.3 * k / tiers)
            mask |= (y >= t0) & (y < t1) & (np.abs(x + 0.5 - cx) <= half)
        can = _ramp_shade(mask, PINES, cx, sh * 0.4, sw * 0.5)
        if snow:
            caps = mask & ~np.roll(mask, 2, 0)
            can[caps] = (236, 242, 250, 255)
            can[caps & (x > cx)] = (196, 208, 226, 255)
    else:
        mask = (_ell(sw, sh, cx, sh * 0.42, sw * 0.48, sh * 0.3)
                | _ell(sw, sh, cx - sw * 0.18, sh * 0.55, sw * 0.3, sh * 0.2)
                | _ell(sw, sh, cx + sw * 0.18, sh * 0.55, sw * 0.3, sh * 0.2))
        mask &= np.arange(sh)[:, None] < sh - trunk_h + 2
        can = _ramp_shade(mask, GREENS, cx, sh * 0.45, sw * 0.5)
        rng = np.random.default_rng(w)
        spots = mask & (rng.random(mask.shape) < 0.06)
        can[spots, :3] = GREENS[0]
        if snow:
            caps = mask & ~np.roll(mask, 2, 0)
            can[caps] = (236, 242, 250, 255)
    on = can[..., 3] > 0
    img[on] = can[on]
    return _outline_rgba(img), (sw // 2, sh - 2)


def rock_sprite(w: int, snow: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    sw, sh = max(8, int(w * 0.55)) // 2 * 2, max(6, int(w * 0.4))
    m = _ell(sw, sh, sw / 2, sh * 0.58, sw * 0.48, sh * 0.42) | _ell(sw, sh, sw * 0.4, sh * 0.42, sw * 0.3, sh * 0.36)
    img = _ramp_shade(m, GREYS, sw * 0.45, sh * 0.5, sw * 0.4)
    if snow:
        caps = m & ~np.roll(m, 2, 0)
        img[caps] = (236, 242, 250, 255)
    img[m & (np.arange(sw)[None, :] == sw // 2) & (np.arange(sh)[:, None] > sh // 2)] = (*GREYS[0], 255)
    return _outline_rgba(img), (sw // 2, sh - 2)


def house_sprite(w: int) -> tuple[np.ndarray, tuple[int, int]]:
    """A cottage: plaster walls with timber edges, a door, a lit window, a red pyramid roof."""
    fw = int(w * 0.84) // 2 * 2
    wall_h, roof_h = int(w * 0.42), int(w * 0.42)
    sw, sh = fw + 2, fw // 4 + wall_h + roof_h + fw // 4 + 4
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    g = ImageDraw.Draw(img)
    ax, ay = sw // 2, sh - fw // 4 - 2           # cell centre (bottom diamond's centre)
    L, F, R = (ax - fw // 2, ay), (ax, ay + fw // 4), (ax + fw // 2, ay)
    up = lambda p, k: (p[0], p[1] - k)
    g.polygon([L, F, up(F, wall_h), up(L, wall_h)], fill=(214, 196, 160))
    g.polygon([F, R, up(R, wall_h), up(F, wall_h)], fill=(168, 146, 112))
    for p in (L, F, R):
        g.line([p, up(p, wall_h)], fill=(92, 62, 40))
    g.line([up(L, wall_h), up(F, wall_h), up(R, wall_h)], fill=(92, 62, 40))
    # door on the left face, window on the right face
    dx = (L[0] + F[0]) // 2
    dy = (L[1] + F[1]) // 2
    dh = max(3, wall_h * 2 // 3)
    g.polygon([(dx - 2, dy - 1), (dx + 2, dy + 1), (dx + 2, dy + 1 - dh), (dx - 2, dy - 1 - dh)], fill=(84, 54, 34))
    wx = (F[0] + R[0]) // 2
    wy = (F[1] + R[1]) // 2 - wall_h // 2
    g.polygon([(wx - 2, wy + 1), (wx + 2, wy - 1), (wx + 2, wy - 4), (wx - 2, wy - 2)], fill=(255, 210, 120))
    # roof: a pyramid over the walls, eaves one pixel out
    top = (ax, ay - wall_h - roof_h)
    eL, eF, eR = (L[0] - 1, L[1] - wall_h), (F[0], F[1] - wall_h + 1), (R[0] + 1, R[1] - wall_h)
    eB = (ax, ay - fw // 4 - wall_h - 1)
    g.polygon([eB, eL, top], fill=(150, 62, 44))
    g.polygon([eB, eR, top], fill=(118, 46, 36))
    g.polygon([eL, eF, top], fill=(190, 84, 56))
    g.polygon([eF, eR, top], fill=(140, 58, 42))
    g.line([eF, top], fill=(110, 44, 34))
    out = np.asarray(img).copy()
    out[out[..., 3] > 0, 3] = 255
    return _outline_rgba(out), (ax, ay)


def well_sprite(w: int) -> tuple[np.ndarray, tuple[int, int]]:
    sw = int(w * 0.6) // 2 * 2
    ry = sw // 4
    side = max(3, w // 6)
    post = max(6, int(w * 0.45))
    sh = ry * 2 + side + post + 4
    img = np.zeros((sh, sw, 4), np.uint8)
    cy = sh - ry - 1
    body = np.zeros((sh, sw), bool)
    body[cy - side:cy] = True
    body &= _ell(sw, sh, sw / 2, cy - side / 2, sw / 2, side + ry)
    body |= _ell(sw, sh, sw / 2, cy, sw / 2, ry)
    img[body] = (*GREYS[2], 255)
    img[body & (np.arange(sw)[None, :] > sw / 2)] = (*GREYS[1], 255)
    rim = _ell(sw, sh, sw / 2, cy - side, sw / 2, ry)
    img[rim] = (*GREYS[3], 255)
    img[_ell(sw, sh, sw / 2, cy - side, sw / 2 - 2, ry - 1)] = (34, 60, 96, 255)
    for x in (1, sw - 3):                                    # posts and a little roof
        img[cy - side - post:cy - side, x:x + 2] = (110, 74, 44, 255)
    ry0 = cy - side - post
    for k in range(4):
        img[ry0 - k, k:sw - k] = (150, 62, 44, 255)
    img[cy - side - post + 2, 2:sw - 2] = (80, 54, 34, 255)      # crossbar
    return _outline_rgba(img), (sw // 2, cy)


def fence_sprite(w: int, along_col: bool = True) -> tuple[np.ndarray, tuple[int, int]]:
    """Two posts and two rails along the cell (+col runs to the lower right, +row lower left)."""
    sw, ph = w // 2 + 2, max(5, w // 4)
    sh = w // 4 + ph + 3
    img = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    g = ImageDraw.Draw(img)
    ax, ay = sw // 2, sh - w // 8 - 2
    dx, dy = w // 4 - 1, w // 8
    a = (ax - dx, ay - dy) if along_col else (ax + dx, ay - dy)
    b = (ax + dx, ay + dy) if along_col else (ax - dx, ay + dy)
    for k in (ph - 1, ph // 2):
        g.line([(a[0], a[1] - k), (b[0], b[1] - k)], fill=(150, 108, 66))
    for p in (a, b):
        g.line([p, (p[0], p[1] - ph)], fill=(104, 70, 42), width=2)
    out = np.asarray(img).copy()
    out[out[..., 3] > 0, 3] = 255
    return _outline_rgba(out), (ax, ay)


def box_sprite(w: int, colour=(170, 120, 190), height: int | None = None) -> tuple[np.ndarray, tuple[int, int]]:
    """A plain block: unknown prop kinds (and the wall prop, in stone colours)."""
    from .tilesets import iso_block
    bw = max(8, int(w * 0.6)) // 4 * 4
    t = np.zeros((8, 8, 4), np.uint8)
    t[...] = (*colour, 255)
    t[0, :, :3] = t[:, 0, :3] = np.clip(np.array(colour) * 0.75, 0, 255)
    img = iso_block(t, t, bw, height)
    return img, (bw // 2, img.shape[0] - bw // 4)


def battlement_sprite(w: int, sides) -> tuple[np.ndarray, tuple[int, int]]:
    """Merlons along the cell edges in `sides` ((dc, dr) toward lower neighbours)."""
    from .tilesets import iso_block
    mw = max(4, w // 5) // 2 * 2
    t = np.zeros((8, 8, 4), np.uint8)
    t[...] = (*GREYS[2], 255)
    t[0, :, :3] = GREYS[1]
    merlon = iso_block(t, t, mw, max(3, w // 6))
    size = w + mw * 2
    canvas = np.zeros((size, size, 4), np.uint8)
    ax, ay = size // 2, size // 2 + mw
    vc, vr = np.array([w / 2, w / 4]), np.array([-w / 2, w / 4])
    order = sorted(sides, key=lambda d: d[0] + d[1])          # back edges first
    for dc, dr in order:
        mid = 0.42 * (dc * vc + dr * vr)
        perp = dr * vc + dc * vr
        for k in (-0.28, 0.28):
            x, y = mid + k * perp
            _paste(canvas, merlon, int(ax + x - mw / 2), int(ay + y - merlon.shape[0] + mw / 4))
    return canvas, (ax, ay)


def battlement_sides(height: np.ndarray, c: int, r: int) -> list[tuple[int, int]]:
    rows, cols = height.shape
    out = []
    for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        n = (c + dc, r + dr)
        if not (0 <= n[0] < cols and 0 <= n[1] < rows) or height[n[1], n[0]] < height[r, c] - 1:
            out.append((dc, dr))
    return out


def prop_sprite(kind: str, w: int, snow: bool = False, along_col: bool = True):
    """(RGBA, anchor) for a prop kind at tile width w; unknown kinds = a box."""
    from .iso_scene import barrel_sprite, crate_sprite, post_sprite, torch_frames
    if kind == "tree":
        return tree_sprite(w, pine=snow, snow=snow)
    if kind == "pine":
        return tree_sprite(w, pine=True, snow=snow)
    if kind == "rock":
        return rock_sprite(w, snow)
    if kind == "house":
        return house_sprite(w)
    if kind == "well":
        return well_sprite(w)
    if kind == "fence":
        return fence_sprite(w, along_col)
    if kind == "wall":
        return box_sprite(w, (124, 122, 130), max(4, w // 3))
    if kind == "crate":
        cw = max(8, int(w * 0.6) // 4 * 4)
        img = crate_sprite(cw)
        return img, (cw // 2, img.shape[0] - cw // 4)
    if kind == "barrel":
        bw = max(8, int(w * 0.4) // 2 * 2)
        img = barrel_sprite(bw)
        return img, (bw // 2, img.shape[0] - 2)
    if kind == "post":
        img = post_sprite(4, max(6, int(w * 0.6)))
        return img, (2, img.shape[0] - 1)
    if kind == "crystal":
        return box_sprite(w, (90, 190, 240), max(4, w // 3))
    if kind in ("torch", "brazier", "lantern"):
        strip = torch_frames(w)
        tw = strip.shape[1] // 4
        img = np.ascontiguousarray(strip[:, :tw])
        return img, (tw // 2, img.shape[0] - 1)
    return box_sprite(w)


def prop_icon(kind: str, cell: int) -> np.ndarray:
    """A top-down icon for a prop at `cell` px (the editor canvas, the top-down preview)."""
    img = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    g = ImageDraw.Draw(img)
    m = max(1, cell // 8)
    full = (m, m, cell - 1 - m, cell - 1 - m)
    dark = (24, 20, 26)
    if kind in ("tree", "pine"):
        g.ellipse(full, fill=(52, 110, 50), outline=dark)
        g.ellipse((cell * 0.25, cell * 0.2, cell * 0.55, cell * 0.5), fill=(110, 170, 78))
    elif kind == "rock":
        g.ellipse((m * 2, m * 3, cell - 1 - m * 2, cell - 1 - m), fill=(132, 130, 138), outline=dark)
    elif kind == "house":
        g.rectangle(full, fill=(170, 70, 48), outline=dark)
        g.line((m, cell // 2, cell - 1 - m, cell // 2), fill=(110, 44, 34))
    elif kind == "wall":
        g.rectangle(full, fill=(126, 124, 132), outline=dark)
    elif kind == "battlement":
        q = max(2, cell // 4)
        for x in (m, cell - 1 - m - q):
            for y in (m, cell - 1 - m - q):
                g.rectangle((x, y, x + q, y + q), fill=(150, 148, 156), outline=dark)
    elif kind == "well":
        g.ellipse(full, fill=(140, 138, 146), outline=dark)
        g.ellipse((cell * 0.3, cell * 0.3, cell * 0.7, cell * 0.7), fill=(40, 70, 110))
    elif kind == "fence":
        g.rectangle((m, cell // 2 - 1, cell - 1 - m, cell // 2 + 1), fill=(150, 108, 66))
        g.rectangle((cell // 2 - 1, m, cell // 2 + 1, cell - 1 - m), fill=(150, 108, 66))
    elif kind == "crate":
        g.rectangle((m * 2, m * 2, cell - 1 - m * 2, cell - 1 - m * 2), fill=(150, 104, 62), outline=dark)
    elif kind == "barrel":
        g.ellipse((m * 2, m * 2, cell - 1 - m * 2, cell - 1 - m * 2), fill=(128, 84, 48), outline=dark)
    elif kind == "post":
        g.rectangle((cell // 2 - m, cell // 2 - m, cell // 2 + m, cell // 2 + m), fill=(104, 70, 42), outline=dark)
    elif kind in ("torch", "brazier", "lantern"):
        r = 0.35 if kind == "torch" else (0.22 if kind == "brazier" else 0.4)
        g.ellipse((cell * r, cell * r, cell * (1 - r), cell * (1 - r)), fill=(255, 170, 40), outline=(200, 70, 20))
    elif kind == "crystal":
        g.polygon([(cell / 2, m), (cell - 1 - m * 2, cell / 2), (cell / 2, cell - 1 - m), (m * 2, cell / 2)],
                  fill=(110, 200, 245), outline=(30, 80, 140))
    else:
        g.rectangle((m * 2, m * 2, cell - 1 - m * 2, cell - 1 - m * 2), fill=(170, 120, 190), outline=dark)
    return np.asarray(img).copy()


# --------------------------------------------------------------------------- previews

def _shade(rgb: np.ndarray, k) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) * k, 0, 255).astype(np.uint8)


def map_look(doc: MapDoc | None = None, project: assets.Project | None = None, tile: int | None = None):
    """The project's generated tilesets as this map uses them (tilegen.MapLook), or None."""
    from .tilegen import MapLook
    try:
        return MapLook.for_project(project, doc.tiles if doc is not None else None, tile)
    except Exception:  # noqa: BLE001 - no project / unreadable sets: built-in textures
        return None


def render_topdown(doc: MapDoc, cell: int = 16, *, spawns: bool = True, props: bool = True,
                   grid: bool = False, look=None) -> np.ndarray:
    """Top-down colour picture: terrain textures (or a tilegen.MapLook's tiles: shores, paths,
    cliffs, fences), lighter when higher, cliff shadows, props and team starts.
    (rows*cell, cols*cell, 4)."""
    d = doc.diorama
    tex = terrain_textures(cell)
    img = np.zeros((d.rows * cell, d.cols * cell, 4), np.uint8)
    h = d.height.astype(int)
    tiled = None
    if look is not None:
        tiled = look.render(d, doc.seed)
        if look.tile != cell:
            tiled = np.asarray(Image.fromarray(tiled).resize((d.cols * cell, d.rows * cell), Image.NEAREST))
    cliffs = look is not None and "cliff" in look.sets
    for r in range(d.rows):
        for c in range(d.cols):
            k = 0.74 + 0.06 * min(h[r, c], 8)
            if tiled is not None:
                tile = tiled[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].copy()
            else:
                tile = tex.get(d.terrain[r, c], tex["grass"])[0].copy()
            tile[..., :3] = _shade(tile[..., :3], k)
            # shadow from a higher neighbour above/left (light from the upper left)
            for dc, dr, sl in ((0, -1, (slice(0, None), slice(None))), (-1, 0, (slice(None), slice(0, None)))):
                nc, nr = c + dc, r + dr
                if cliffs and 0 <= nc < d.cols and 0 <= nr < d.rows and h[nr, nc] - h[r, c] >= 2:
                    continue                                  # the cliff tiles draw that drop
                if 0 <= nc < d.cols and 0 <= nr < d.rows and h[nr, nc] > h[r, c]:
                    band = min(cell // 2, (h[nr, nc] - h[r, c]) * max(1, cell // 8))
                    if dr:
                        tile[:band, :, :3] = _shade(tile[:band, :, :3], 0.62)
                    else:
                        tile[:, :band, :3] = _shade(tile[:, :band, :3], 0.68)
            # a lit edge where this cell drops toward the viewer (lower neighbour below/right)
            for dc, dr in ((0, 1), (1, 0)):
                nc, nr = c + dc, r + dr
                if cliffs and 0 <= nc < d.cols and 0 <= nr < d.rows and h[r, c] - h[nr, nc] >= 2:
                    continue
                if 0 <= nc < d.cols and 0 <= nr < d.rows and h[nr, nc] < h[r, c]:
                    if dr:
                        tile[-1:, :, :3] = _shade(tile[-1:, :, :3], 0.45)
                    else:
                        tile[:, -1:, :3] = _shade(tile[:, -1:, :3], 0.5)
            if grid:
                tile[0, :, :3] = _shade(tile[0, :, :3], 0.85)
                tile[:, 0, :3] = _shade(tile[:, 0, :3], 0.85)
            img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = tile
    if props:
        icons: dict = {}
        drawn = look.props if look is not None else set()
        for kind, c, r in d.props:
            if not (0 <= c < d.cols and 0 <= r < d.rows) or kind in drawn:
                continue
            ic = icons.get(kind)
            if ic is None:
                ic = icons[kind] = prop_icon(kind, cell)
            dst = img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            on = ic[..., 3] > 0
            dst[on] = ic[on]
    if spawns:
        for team, c, r in doc.spawns:
            if not (0 <= c < d.cols and 0 <= r < d.rows):
                continue
            col = TEAM_RGB[team % 2]
            dst = img[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell]
            dst[..., :3] = (dst[..., :3] * 0.45 + np.array(col) * 0.55).astype(np.uint8)
            for s in (slice(0, 2), slice(cell - 2, cell)):
                dst[s, :, :3] = col
                dst[:, s, :3] = col
    return img


def rotate_doc(doc: MapDoc, quarter_turns: int = 0) -> MapDoc:
    """The map turned by 90° steps (counter-clockwise on the grid = the view turns clockwise)."""
    k = quarter_turns % 4
    out = doc.copy()
    d = out.diorama
    for _ in range(k):
        cols = d.cols
        d.height = np.rot90(d.height).copy()
        d.terrain = np.rot90(d.terrain).copy()
        d.props = [(kind, r, cols - 1 - c) for kind, c, r in d.props]
        out.spawns = [(t, r, cols - 1 - c) for t, c, r in out.spawns]
        d.rows, d.cols = d.height.shape
    return out


_COLS: dict = {}


def _column(terrain: str, level: int, base: int, tile: int) -> np.ndarray:
    from .iso_scene import iso_column
    key = (terrain, level, base, tile)
    if key not in _COLS:
        tex = terrain_textures(tile)
        top, side = tex.get(terrain, tex["grass"])
        _COLS[key] = iso_column(top, side, tile, level - base, tile // 4, level)
    return _COLS[key]


_PROPS: dict = {}


def _prop(kind: str, tile: int, snow: bool, along: bool):
    key = (kind, tile, snow, along)
    if key not in _PROPS:
        _PROPS[key] = prop_sprite(kind, tile, snow, along)
    return _PROPS[key]


def _paste(canvas: np.ndarray, img: np.ndarray, x: int, y: int) -> None:
    """Alpha-over (binary alpha is enough for these sprites) with clipping."""
    hh, ww = img.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + ww), min(H, y + hh)
    if x0 >= x1 or y0 >= y1:
        return
    src = img[y0 - y:y1 - y, x0 - x:x1 - x]
    dst = canvas[y0:y1, x0:x1]
    a = src[..., 3:4].astype(np.float32) / 255.0
    dst[..., :3] = (src[..., :3] * a + dst[..., :3] * (1 - a)).astype(np.uint8)
    dst[..., 3] = np.maximum(dst[..., 3], src[..., 3])


def iso_geometry(doc: MapDoc, tile: int):
    """(width, height, origin_x, origin_y, base level) of the iso picture: cell (c, r) at
    level h has its top-diamond centre at (ox + (c - r) * tile/2, oy + (c + r) * tile/4 - h * tile/4)."""
    d = doc.diorama
    lp = tile // 4
    base = min(0, int(d.height.min())) - 1
    hi = int(d.height.max())
    width = (d.cols + d.rows) * tile // 2
    top_pad = hi * lp + tile * 3 // 2                      # tall columns + trees at the back
    height = top_pad + (d.cols + d.rows) * tile // 4 + (-base) * lp + tile // 2
    ox = d.rows * tile // 2
    oy = top_pad + tile // 4
    return width, height, ox, oy, base


def render_iso(doc: MapDoc, rotation: int = 0, tile: int = 16, *, spawns: bool = True,
               background=(0, 0, 0, 0)) -> np.ndarray:
    """Quick isometric picture of the map (painter's order: back diagonals first), turned by
    `rotation` quarter turns. Unknown prop kinds are drawn as boxes."""
    if tile % 8:
        raise ValueError("tile must be a multiple of 8")
    doc = rotate_doc(doc, rotation)
    d = doc.diorama
    lp = tile // 4
    width, height, ox, oy, base = iso_geometry(doc, tile)
    canvas = np.zeros((height, width, 4), np.uint8)
    canvas[...] = background
    props = {(c, r): k for k, c, r in d.props}
    starts = {(c, r): t for t, c, r in doc.spawns} if spawns else {}
    fences = {(c, r) for k, c, r in d.props if k == "fence"}
    from .iso_scene import cursor_sprite
    ring = cursor_sprite(tile)
    for s in range(d.cols + d.rows - 1):
        cells = [(c, s - c) for c in range(max(0, s - d.rows + 1), min(d.cols, s + 1))]
        for c, r in cells:
            lvl = int(d.height[r, c])
            col = _column(d.terrain[r, c], lvl, base, tile)
            x = ox + (c - r) * tile // 2 - tile // 2
            y = oy + (c + r) * tile // 4 - lvl * lp - tile // 4
            _paste(canvas, col, x, y)
            if (c, r) in starts:
                rgb = TEAM_RGB[starts[(c, r)] % 2]
                rr = ring.copy()
                rr[rr[..., 3] > 0, :3] = rgb
                fill = iso_floor_mask(tile)
                over = np.zeros_like(rr)
                over[fill] = (*rgb, 150)
                _paste(canvas, over, x, y)
                _paste(canvas, rr, x, y)
        for c, r in cells:
            kind = props.get((c, r))
            if kind is None:
                continue
            lvl = int(d.height[r, c])
            snow = d.terrain[r, c] == "snow"
            along = (c - 1, r) in fences or (c + 1, r) in fences or not ((c, r - 1) in fences or (c, r + 1) in fences)
            if kind == "battlement":
                img, (ax, ay) = battlement_sprite(tile, battlement_sides(d.height, c, r))
            else:
                img, (ax, ay) = _prop(kind, tile, snow, along)
            cx = ox + (c - r) * tile // 2
            cy = oy + (c + r) * tile // 4 - lvl * lp
            _paste(canvas, img, cx - ax, cy - ay)
    return canvas


def iso_floor_mask(tile: int) -> np.ndarray:
    from .tilesets import iso_floor
    return iso_floor(np.full((8, 8, 4), 255, np.uint8), tile)[..., 3] > 0


def _fit(img: np.ndarray, w: int, h: int, bg=(30, 26, 38, 255)) -> np.ndarray:
    """Scale (nearest, integer steps when growing) into a w x h box on a background."""
    ih, iw = img.shape[:2]
    k = min(w / iw, h / ih)
    if k >= 1:
        k = int(k)
    nw, nh = max(1, int(iw * k)), max(1, int(ih * k))
    im = Image.fromarray(img).resize((nw, nh), Image.NEAREST if k >= 1 else Image.BOX)
    out = Image.new("RGBA", (w, h), bg)
    out.alpha_composite(im, ((w - nw) // 2, (h - nh) // 2))
    return np.asarray(out).copy()


# --------------------------------------------------------------------------- the asset store

def _project_of(asset: assets.Asset) -> assets.Project:
    folder = asset.folder.parent.parent
    return assets.Project(folder.name, folder)


def save_map(doc: MapDoc, name: str | None = None, *, project: assets.Project | None = None,
             asset: assets.Asset | None = None, tags=None) -> assets.Asset:
    """Save as a map asset (a new one, or over `asset`): map.json, tex/, preview.png,
    thumb.png, iso_preview.png. Returns the Asset."""
    name = (name or (asset.name if asset else doc.diorama.name) or "map").strip() or "map"
    extra = {"template": doc.template, "size": [doc.cols, doc.rows], "seed": doc.seed}
    if asset is None:
        asset = assets.new_asset("map", name, project, **extra)
    else:
        asset.name = name
        asset.extra.update(extra)
    if tags is not None:
        asset.tags = sorted({t.strip() for t in tags if t.strip()})
    elif not asset.tags:
        asset.tags = [doc.template]
    proj = project or _project_of(asset)
    tile = proj.style.tile
    doc.diorama.name = name
    folder = asset.folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "map.json").write_text(json.dumps(doc.to_dict(), indent=1), encoding="utf-8")
    tex = folder / "tex"
    names = write_textures(doc.diorama, tex, tile)
    keep = {f for pair in names.values() for f in pair.values()}
    for p in tex.glob("*.png"):                  # terrains the map no longer uses
        if p.name not in keep:
            try:
                p.unlink()
            except OSError:
                pass
    look = map_look(doc, proj)
    Image.fromarray(render_topdown(doc, look.tile if look else 16, look=look)).save(folder / "preview.png")
    iso = render_iso(doc, 0, 32 if tile >= 32 else 16)
    Image.fromarray(iso).save(folder / "iso_preview.png")
    Image.fromarray(_fit(iso, 160, 120)).save(folder / "thumb.png")
    asset.save()
    return asset


def _map_json(src) -> Path:
    if isinstance(src, assets.Asset):
        return src.folder / "map.json"
    p = Path(src)
    return p / "map.json" if p.is_dir() else p


def load_map(src) -> MapDoc:
    """A MapDoc from a map Asset, its folder or a map.json."""
    if isinstance(src, MapDoc):
        return src.copy()
    data = json.loads(_map_json(src).read_text(encoding="utf-8"))
    return MapDoc.from_dict(data)


def list_maps(project: assets.Project | None = None, query: str = "", tag: str | None = None):
    return assets.list_assets("map", query, tag, project=project)


def load_diorama(src, blue: list[str] | None = None, red: list[str] | None = None) -> Diorama:
    """The map as a Diorama for a game (hd2d.make_hd2d / iso_scene): with rosters, units stand
    on the team starts (more units than starts: the nearest free cells around them)."""
    doc = load_map(src)
    d = doc.diorama
    d.units = []
    if not blue and not red:
        return d
    if not doc.spawns:
        return place_units(d, list(blue or []), list(red or []))
    taken: set = set()
    for team, names, facing in ((0, blue or [], "N"), (1, red or [], "S")):
        starts = [(c, r) for t, c, r in doc.spawns if t == team]
        order = list(starts)
        # then the cells around the starts, nearest first
        if starts:
            ring = sorted({(c + dx, r + dy) for c, r in starts for dx in range(-3, 4) for dy in range(-3, 4)},
                          key=lambda p: min(abs(p[0] - c) + abs(p[1] - r) for c, r in starts))
            order += [p for p in ring if p not in starts]
        for name in names:
            for c, r in order:
                if (c, r) not in taken and doc.free(c, r):
                    taken.add((c, r))
                    d.units.append(Unit(name, c, r, team, facing))
                    break
    return d


# --------------------------------------------------------------------------- Godot 3D extras

PROPS3D_GD = r'''extends RefCounted
# Prop kinds the HD-2D board (board.gd) doesn't build itself: trees, rocks, cottages, walls,
# wells, fences. board.gd calls build() for every prop; returning false = not ours.
# Written by pixel_pipeline.maplib.

const KINDS := ["tree", "rock", "house", "wall", "well", "fence", "battlement"]
const BLOCKING := ["tree", "rock", "house", "wall", "well", "fence"]
var mats := {}
var fences := {}

func _mat(key: String, col: Color, tex := "") -> StandardMaterial3D:
	if mats.has(key):
		return mats[key]
	var m := StandardMaterial3D.new()
	m.albedo_color = col
	m.roughness = 0.92
	if tex != "" and FileAccess.file_exists(tex):
		var img := Image.load_from_file(ProjectSettings.globalize_path(tex))
		if img != null:
			img.generate_mipmaps()
			m.albedo_texture = ImageTexture.create_from_image(img)
			m.texture_filter = BaseMaterial3D.TEXTURE_FILTER_NEAREST_WITH_MIPMAPS
	mats[key] = m
	return m

func _add(board: Node3D, mesh: Mesh, mat: Material, pos: Vector3, yaw := 0.0, scale := Vector3.ONE) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	mi.mesh = mesh
	mi.material_override = mat
	mi.position = pos
	mi.rotation.y = yaw
	mi.scale = scale
	board.add_child(mi)
	return mi

func _cone(r: float, h: float) -> CylinderMesh:
	var cm := CylinderMesh.new()
	cm.top_radius = 0.0
	cm.bottom_radius = r
	cm.height = h
	cm.radial_segments = 8
	return cm

func _box(x: float, y: float, z: float) -> BoxMesh:
	var bm := BoxMesh.new()
	bm.size = Vector3(x, y, z)
	return bm

func build(board: Node3D, kind: String, c: int, r: int, base: Vector3, i: int) -> bool:
	if not kind in KINDS:
		return false
	if fences.is_empty():
		for p in board.map.get("props", []):
			if p["kind"] == "fence":
				fences[Vector2i(int(p["col"]), int(p["row"]))] = true
		fences[Vector2i(-99, -99)] = true
	var rng := RandomNumberGenerator.new()
	rng.seed = c * 7919 + r * 104729 + 17
	var snowy: bool = board.terrain[r][c] == "snow"
	var bark := _mat("bark", Color(0.36, 0.24, 0.15))
	var stone := _mat("stone", Color(0.62, 0.61, 0.66), "res://map/tex/stone_top.png")
	var snow := _mat("snow", Color(0.93, 0.96, 1.0))
	var j := Vector3(rng.randf_range(-0.12, 0.12), 0, rng.randf_range(-0.12, 0.12))
	match kind:
		"tree":
			var k := rng.randf_range(0.85, 1.2)
			var trunk := CylinderMesh.new()
			trunk.top_radius = 0.06; trunk.bottom_radius = 0.1; trunk.height = 0.7 * k
			_add(board, trunk, bark, base + j + Vector3(0, 0.35 * k, 0))
			if snowy or (c * 3 + r * 5) % 4 == 0:
				var green := _mat("pine", Color(0.13, 0.32, 0.25))
				for t in 3:
					var y := (0.55 + t * 0.38) * k
					_add(board, _cone((0.5 - t * 0.12) * k, 0.6 * k), green, base + j + Vector3(0, y, 0), rng.randf() * TAU)
					if snowy:
						_add(board, _cone((0.3 - t * 0.07) * k, 0.3 * k), snow, base + j + Vector3(0, y + 0.18 * k, 0))
			else:
				var leaf := _mat("leaf%d" % (i % 3), Color(0.2 + 0.04 * (i % 3), 0.42 + 0.05 * (i % 2), 0.17))
				var sm := SphereMesh.new(); sm.radius = 0.42 * k; sm.height = 0.72 * k
				sm.radial_segments = 10; sm.rings = 5
				_add(board, sm, leaf, base + j + Vector3(0, 0.95 * k, 0))
				_add(board, sm, leaf, base + j + Vector3(0.2, 0.72 * k, 0.12), 0.0, Vector3(0.65, 0.65, 0.65))
				_add(board, sm, leaf, base + j + Vector3(-0.16, 0.78 * k, -0.14), 0.0, Vector3(0.6, 0.6, 0.6))
		"rock":
			var sm := SphereMesh.new(); sm.radius = 0.3; sm.height = 0.42; sm.radial_segments = 7; sm.rings = 4
			_add(board, sm, stone, base + j + Vector3(0, 0.12, 0), rng.randf() * TAU, Vector3(rng.randf_range(0.9, 1.4), 1, 1))
			if snowy:
				var cap := SphereMesh.new(); cap.radius = 0.24; cap.height = 0.16
				_add(board, cap, snow, base + j + Vector3(0, 0.3, 0))
		"house":
			var yaw := (PI / 2.0) * float(rng.randi_range(0, 3))
			var plaster := _mat("plaster", Color(0.86, 0.79, 0.64))
			var roof := _mat("roof", Color(0.62, 0.25, 0.17))
			_add(board, _box(0.86, 0.62, 0.8), plaster, base + Vector3(0, 0.31, 0), yaw)
			var pm := PrismMesh.new(); pm.size = Vector3(1.0, 0.5, 0.94)
			_add(board, pm, roof, base + Vector3(0, 0.87, 0), yaw)
			var door := _add(board, _box(0.22, 0.36, 0.04), bark, base + Vector3(0, 0.18, 0.41).rotated(Vector3.UP, yaw), yaw)
			var win := _mat("window", Color(1.0, 0.78, 0.4))
			win.emission_enabled = true
			win.emission = Color(1.0, 0.62, 0.25)
			win.emission_energy_multiplier = 1.4
			_add(board, _box(0.04, 0.16, 0.18), win, base + Vector3(0.44, 0.38, 0.0).rotated(Vector3.UP, yaw), yaw)
		"wall":
			_add(board, _box(0.96, 0.7, 0.96), stone, base + Vector3(0, 0.35, 0))
		"well":
			var cm := CylinderMesh.new(); cm.top_radius = 0.34; cm.bottom_radius = 0.36; cm.height = 0.42
			_add(board, cm, stone, base + Vector3(0, 0.21, 0))
			var water := StandardMaterial3D.new(); water.albedo_color = Color(0.1, 0.2, 0.35); water.roughness = 0.1
			var disc := CylinderMesh.new(); disc.top_radius = 0.26; disc.bottom_radius = 0.26; disc.height = 0.02
			_add(board, disc, water, base + Vector3(0, 0.4, 0))
			for x in [-0.3, 0.3]:
				_add(board, _box(0.06, 0.7, 0.06), bark, base + Vector3(x, 0.6, 0))
			var pm := PrismMesh.new(); pm.size = Vector3(0.8, 0.26, 0.5)
			_add(board, pm, _mat("roof", Color(0.62, 0.25, 0.17)), base + Vector3(0, 1.05, 0))
		"fence":
			var along_x := fences.has(Vector2i(c - 1, r)) or fences.has(Vector2i(c + 1, r)) \
				or not (fences.has(Vector2i(c, r - 1)) or fences.has(Vector2i(c, r + 1)))
			var wood := _mat("fence", Color(0.55, 0.4, 0.25))
			var yaw := 0.0 if along_x else PI / 2.0
			for x in [-0.42, 0.0, 0.42]:
				_add(board, _box(0.08, 0.5, 0.08), wood, base + Vector3(x, 0.25, 0).rotated(Vector3.UP, yaw), yaw)
			for y in [0.18, 0.38]:
				_add(board, _box(1.0, 0.06, 0.05), wood, base + Vector3(0, y, 0), yaw)
		"battlement":
			# merlons along every edge that drops to a lower neighbour
			var h0 := float(board.heights[r][c])
			for d: Vector2i in [Vector2i(1, 0), Vector2i(-1, 0), Vector2i(0, 1), Vector2i(0, -1)]:
				var n: Vector2i = Vector2i(c, r) + d
				var low: bool = n.x < 0 or n.y < 0 or n.x >= board.cols or n.y >= board.rows
				if not low:
					low = float(board.heights[n.y][n.x]) < h0 - 1.0
				if not low:
					continue
				var edge: Vector3 = Vector3(d.x, 0, d.y) * 0.42
				var side: Vector3 = Vector3(d.y, 0, d.x)
				for k in [-0.3, 0.3]:
					_add(board, _box(0.24 if d.x == 0 else 0.16, 0.3, 0.24 if d.x != 0 else 0.16), stone,
						base + edge + side * k + Vector3(0, 0.15, 0))
	if kind in BLOCKING:
		board.blocked[Vector2i(c, r)] = true
	return true
'''

VIEW_TSCN = """[gd_scene load_steps=2 format=3]

[ext_resource type="Script" path="res://view.gd" id="1"]

[node name="MapView" type="Node3D"]
script = ExtResource("1")
mode = "real"
"""

VIEW_GD = r'''extends "res://board.gd"
# View only: the map with its team starts, no units or rules.
# Q/E turn, WASD pan, wheel zoom, Esc quits.
# Screenshot: godot --path . -- --shot=C:/out.png [--yaw=45] [--dist=30]

var _shot := ""
var _frames := 0

func _ready() -> void:
	super._ready()
	cam_dist = clamp(max(cols, rows) * 1.9, 20.0, 60.0)
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--shot="):
			_shot = a.substr(7)
		elif a.begins_with("--yaw="):
			cam_yaw = float(a.substr(6))
			cam_yaw_target = cam_yaw
		elif a.begins_with("--dist="):
			cam_dist = float(a.substr(7))
	pivot.position = Vector3(cols / 2.0, 1.0, rows / 2.0)
	if cam.attributes is CameraAttributesPractical:
		cam.attributes.dof_blur_amount = 0.07          # a viewer: keep more of the map sharp
	_spawn_markers()

func _spawn_markers() -> void:
	for s in map.get("spawns", []):
		var c := int(s["col"]); var r := int(s["row"])
		var q := MeshInstance3D.new()
		var qm := QuadMesh.new(); qm.size = Vector2(0.86, 0.86); qm.orientation = PlaneMesh.FACE_Y
		q.mesh = qm
		var m := StandardMaterial3D.new()
		m.albedo_color = Color(0.3, 0.55, 1.0, 0.55) if int(s["team"]) == 0 else Color(1.0, 0.3, 0.25, 0.55)
		m.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		m.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		q.material_override = m
		q.position = cell_pos(c, r) + Vector3(0, 0.03, 0)
		add_child(q)

func _update_hud(extra: String = "") -> void:
	if hud != null:
		hud.text = "%s   Q/E: turn   WASD: pan   Wheel: zoom   Esc: quit" % map.get("name", "map")

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and event.keycode == KEY_ESCAPE:
		get_tree().quit()
		return
	if event is InputEventKey and event.keycode in [KEY_SPACE, KEY_TAB]:
		return
	if event is InputEventMouseButton and event.button_index == MOUSE_BUTTON_LEFT:
		return
	super._unhandled_input(event)

func _process(delta: float) -> void:
	super._process(delta)
	if _shot != "":
		_frames += 1
		if _frames == 50:
			get_viewport().get_texture().get_image().save_png(_shot)
			get_tree().quit()
'''

VIEW_PROJECT = """config_version=5

[application]

config/name="{name}"
run/main_scene="res://view.tscn"
config/features=PackedStringArray("4.6", "Forward Plus")

[display]

window/size/viewport_width=1920
window/size/viewport_height=1080
window/size/window_width_override=1280
window/size/window_height_override=720
window/stretch/mode="canvas_items"

[rendering]

textures/canvas_textures/default_texture_filter=0
anti_aliasing/quality/msaa_3d=0
"""


def install_board_extras(project_dir, tile: int = 32) -> list[Path]:
    """Write res://map/props3d.gd so hd2d's board.gd builds trees, rocks, cottages, walls,
    wells, fences and battlements (call after hd2d.make_hd2d for a game on a saved map). Also
    the planks and stone textures if the map has none: the board's crates, barrels and posts
    and the props here use them."""
    root = Path(project_dir) / "map"
    p = root / "props3d.gd"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(PROPS3D_GD, encoding="utf-8")
    out = [p]
    tex = textures(None, tile)
    for name in ("planks", "stone"):
        for part, img in zip(("top", "side"), tex[name]):
            f = root / "tex" / f"{name}_{part}.png"
            if not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(img).save(f)
                out.append(f)
    return out


# --------------------------------------------------------------------------- exports

def _export_godot3d(doc: MapDoc, out: Path, style: assets.Style, look=None) -> list[Path]:
    from .hd2d import make_hd2d
    d = doc.copy().diorama
    d.units = []
    files = make_hd2d(out, {}, d, name=d.name, iso=False, tile_px=style.tile, ppu=style.ppu)
    if look is not None:                  # generated sets: their plain tile tops the terrain's blocks
        for name in set(d.terrain.ravel()):
            p = out / "map" / "tex" / f"{name}_top.png"
            if p.exists() and (name in look.sets or any(s.base == name for s in look.sets.values())):
                Image.fromarray(np.ascontiguousarray(look.plain(name)[0][0])).save(p)
    (out / "map" / "map.json").write_text(json.dumps(doc.to_dict(), indent=1), encoding="utf-8")
    files += install_board_extras(out, style.tile)
    for rel, text in (("view.tscn", VIEW_TSCN), ("view.gd", VIEW_GD),
                      ("project.godot", VIEW_PROJECT.format(name=d.name))):
        p = out / rel
        p.write_text(text, encoding="utf-8")
        files.append(p)
    return sorted(set(files))


def _tilemap_data(cells: list[tuple[int, int, int, int, int, int]]) -> str:
    """TileMapLayer.tile_map_data: format 0, then per cell int16 x, y, uint16 source,
    atlas x, atlas y, alternative."""
    raw = struct.pack("<H", 0) + b"".join(struct.pack("<hhHHHH", *c) for c in cells)
    return "PackedByteArray(" + ", ".join(str(b) for b in raw) + ")"


def _tres_atlas(path_png: str, tile_w: int, tile_h: int, tiles: list[tuple[int, int, dict]],
                custom: list[tuple[str, int]] = (), extra_resource: list[str] = (),
                ext_id: str = "1_tex", res_id: str = "TileSetAtlasSource_0") -> list[str]:
    """Lines of a one-source TileSet .tres; tiles = [(x, y, {prop: value})]."""
    lines = ['[gd_resource type="TileSet" load_steps=3 format=3]', "",
             f'[ext_resource type="Texture2D" path="{path_png}" id="{ext_id}"]', "",
             f'[sub_resource type="TileSetAtlasSource" id="{res_id}"]',
             f'texture = ExtResource("{ext_id}")', f"texture_region_size = Vector2i({tile_w}, {tile_h})"]
    for x, y, props in tiles:
        lines.append(f"{x}:{y}/0 = 0")
        for k, v in props.items():
            lines.append(f"{x}:{y}/0/{k} = {v}")
    if not any(x.startswith("tile_size") for x in extra_resource):
        extra_resource = [*extra_resource, f"tile_size = Vector2i({tile_w}, {tile_h})"]
    lines += ["", "[resource]", *extra_resource]
    for i, (name, typ) in enumerate(custom):
        lines += [f'custom_data_layer_{i}/name = "{name}"', f"custom_data_layer_{i}/type = {typ}"]
    lines += [f'sources/0 = SubResource("{res_id}")', ""]
    return lines


def _icon_strip(kinds, cell) -> np.ndarray:
    return np.concatenate([prop_icon(k, cell) for k in kinds], 1)


def _height_tiles(levels: int, cell: int) -> np.ndarray:
    """One tile per level: translucent shade, darker low / lighter high, a tick per level."""
    out = np.zeros((cell, cell * levels, 4), np.uint8)
    for k in range(levels):
        t = out[:, k * cell:(k + 1) * cell]
        v = int(40 + 215 * k / max(1, levels - 1))
        t[...] = (v, v, v, 90)
        for i in range(min(k, cell // 2)):
            t[1, 1 + i * 2, :] = (255, 255, 255, 255) if v < 160 else (0, 0, 0, 255)
    return out


def _look_files(look, out: Path) -> dict:
    """Copy the look's sets to out/tilesets/<name>/ (+ overlay.png / overlay.tres: the set as
    drawn over the ground layer, same atlas coords, peering bits and probabilities)."""
    res = {}
    for s in look.sets.values():
        src, dst = s.asset.folder, out / "tilesets" / s.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("tileset.png", "tileset.tres", "tileset.tsx", "tileset.json"):
            if (src / f).exists():
                shutil.copy2(src / f, dst / f)
        Image.fromarray(s.overlay).save(dst / "overlay.png")
        tres = dst / "overlay.tres"
        if (dst / "tileset.tres").exists():
            text = (dst / "tileset.tres").read_text(encoding="utf-8")
            tres.write_text(text.replace('path="tileset.png"', 'path="overlay.png"'), encoding="utf-8")
        else:
            tres.write_text("\n".join(_tres_atlas("overlay.png", s.tile, s.tile, [])), encoding="utf-8")
        res[s.key] = [dst / f for f in ("tileset.png", "tileset.tres", "tileset.tsx", "tileset.json",
                                         "overlay.png", "overlay.tres") if (dst / f).exists()]
    return res


def _export_tilemap(doc: MapDoc, out: Path, style: assets.Style,
                    project: assets.Project | None = None, look=None) -> list[Path]:
    """Top-down TileMap: the maptiles terrain TileSet (terrain painting works in the editor),
    a Height layer (custom data "level"), a Props layer (custom data "kind"), Marker2D starts.
    Terrains the project has a generated tileset for (tilegen.py) paint with its texture; those
    tilesets are copied to tilesets/<name>/ too. With a look (map_look) the scene is what the
    map editor shows: Terrain = ground.tres (plain tiles by weight), then a layer per set
    (shores, paths, cliffs, fences) on its tilesets/<name>/overlay.tres."""
    from .maptiles import Terrain, build_terrain_set, write_godot_terrain_tileset
    from .tilegen import project_textures, project_tilesets
    d = doc.diorama
    T = style.tile
    tex = terrain_textures(T)
    generated = project_textures(project, T)
    used = sorted(set(d.terrain.ravel()), key=lambda n: -int((d.terrain == n).sum()))
    terrains = []
    for name in used:
        if name in generated:
            terrains.append(Terrain(name, *generated[name]))
            continue
        top = tex[name][0]
        dark = top[..., :3].reshape(-1, 3).astype(int)
        edge = tuple(int(v) for v in dark[dark.sum(1).argmin()])
        terrains.append(Terrain(name, top, edge))
    ts = build_terrain_set(terrains, variants=4, seed=doc.seed)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    Image.fromarray(ts.atlas).save(out / "terrain.png")
    files += [out / "terrain.png", write_godot_terrain_tileset(ts, out / "terrain.tres", out / "terrain.png")]
    rng = np.random.default_rng(doc.seed)
    tcells = []
    for r in range(d.rows):
        for c in range(d.cols):
            k = used.index(d.terrain[r, c])
            ax, ay = (int(rng.integers(4)), 0) if k == 0 else (14, k)
            tcells.append((c, r, 0, ax, ay, 0))
    look_layers, ext = [], [("TileSet", "res://terrain.tres", "1_terrain")]
    if look is not None:
        layers = look.layers(d, doc.seed)
        Image.fromarray(look.ground_atlas).save(out / "ground.png")
        gtiles = [(j, i, {"probability": f"{w:g}"} if w != 1 else {})
                  for i, name in enumerate(look.terrains) for j, w in enumerate(look._plain[name][1])]
        (out / "ground.tres").write_text("\n".join(_tres_atlas("ground.png", T, T, gtiles)), encoding="utf-8")
        files += [out / "ground.png", out / "ground.tres"]
        ext = [("TileSet", "res://ground.tres", "1_terrain")]
        tcells = [(c, r, 0, *map(int, layers[0].cells[r, c]), 0) for r in range(d.rows) for c in range(d.cols)]
        copied = _look_files(look, out)
        for i, layer in enumerate(layers[1:]):
            cells = [(c, r, 0, int(x), int(y), 0) for (r, c), (x, y) in
                     ((rc, layer.cells[rc]) for rc in np.ndindex(d.rows, d.cols)) if x >= 0]
            rid = f"{i + 4}_{layer.key}"
            ext.append(("TileSet", f"res://tilesets/{layer.set.name}/overlay.tres", rid))
            look_layers.append((layer.key.title(), rid, cells))
            files += copied.get(layer.key, [])
    levels = max(MAX_LEVEL + 1, int(d.height.max()) + 1)
    Image.fromarray(_height_tiles(levels, T)).save(out / "heights.png")
    (out / "heights.tres").write_text("\n".join(_tres_atlas(
        "heights.png", T, T, [(k, 0, {"custom_data_0": k}) for k in range(levels)], [("level", 2)])),
        encoding="utf-8")
    hcells = [(c, r, 0, max(0, int(d.height[r, c])), 0, 0) for r in range(d.rows) for c in range(d.cols)]
    kinds = list(PROP_KINDS) + sorted({k for k, _, _ in d.props} - set(PROP_KINDS))
    strip = _icon_strip(kinds, T)
    for k in (look.props if look is not None else ()):  # drawn by their own layer: data only
        strip[:, kinds.index(k) * T:(kinds.index(k) + 1) * T] = 0
    Image.fromarray(strip).save(out / "props.png")
    (out / "props.tres").write_text("\n".join(_tres_atlas(
        "props.png", T, T, [(i, 0, {"custom_data_0": f'"{k}"', "custom_data_1": str(k in BLOCKING_PROPS).lower()})
                            for i, k in enumerate(kinds)], [("kind", 4), ("blocking", 1)])), encoding="utf-8")
    pcells = [(c, r, 0, kinds.index(k), 0, 0) for k, c, r in d.props]
    files += [out / "heights.png", out / "heights.tres", out / "props.png", out / "props.tres"]
    ext += [("TileSet", "res://heights.tres", "2_heights"), ("TileSet", "res://props.tres", "3_props")]
    lines = [f'[gd_scene load_steps={len(ext) + 1} format=3]', ""]
    lines += [f'[ext_resource type="{t}" path="{p}" id="{i}"]' for t, p, i in ext]
    lines += ["", f'[node name="{_node_name(d.name)}" type="Node2D"]',
              f'metadata/cols = {d.cols}', f'metadata/rows = {d.rows}', f'metadata/level = {LEVEL}', "",
              '[node name="Terrain" type="TileMapLayer" parent="."]',
              f"tile_map_data = {_tilemap_data(tcells)}", 'tile_set = ExtResource("1_terrain")', ""]
    for name, rid, cells in look_layers:
        lines += [f'[node name="{name}" type="TileMapLayer" parent="."]',
                  f"tile_map_data = {_tilemap_data(cells)}", f'tile_set = ExtResource("{rid}")', ""]
    lines += ['[node name="Height" type="TileMapLayer" parent="."]',
             f"tile_map_data = {_tilemap_data(hcells)}", 'tile_set = ExtResource("2_heights")', "",
             '[node name="Props" type="TileMapLayer" parent="."]',
             f"tile_map_data = {_tilemap_data(pcells)}", 'tile_set = ExtResource("3_props")', "",
             '[node name="Spawns" type="Node2D" parent="."]', ""]
    for i, (team, c, r) in enumerate(doc.spawns):
        lines += [f'[node name="{"Blue" if team == 0 else "Red"}{i}" type="Marker2D" parent="Spawns"]',
                  f"position = Vector2({c * T + T / 2}, {r * T + T / 2})", f"metadata/team = {team}",
                  f"metadata/cell = Vector2i({c}, {r})", ""]
    scene = out / "map.tscn"
    scene.write_text("\n".join(lines), encoding="utf-8")
    files.append(scene)
    files += _write_project(out, d.name, "res://map.tscn")
    for a in project_tilesets(project):                 # fences, walls, shores... ready to paint
        if a.extra.get("camera", "topdown") == "topdown" and (a.folder / "tileset.tres").exists():
            dst = out / "tilesets" / a.folder.name
            dst.mkdir(parents=True, exist_ok=True)
            for f in ("tileset.png", "tileset.tres", "tileset.tsx", "tileset.json"):
                if (a.folder / f).exists():
                    files.append(Path(shutil.copy2(a.folder / f, dst / f)))
    return sorted(set(files))


def _node_name(name: str) -> str:
    s = "".join(ch for ch in name.title() if ch.isalnum())
    return s or "Map"


def _write_project(out: Path, name: str, main: str) -> list[Path]:
    p = out / "project.godot"
    if p.exists():
        return []
    p.write_text("config_version=5\n\n[application]\n\n"
                 f'config/name="{name}"\nrun/main_scene="{main}"\n'
                 'config/features=PackedStringArray("4.6", "GL Compatibility")\n\n'
                 "[rendering]\n\nrenderer/rendering_method=\"gl_compatibility\"\n"
                 "textures/canvas_textures/default_texture_filter=0\n", encoding="utf-8")
    return [p]


def _export_iso(doc: MapDoc, out: Path, style: assets.Style) -> list[Path]:
    """Isometric TileMap: one pre-rendered column per (terrain, level) in one y-sorted
    TileMapLayer (diamond-down layout, like tilesets' iso TileSet), props as sprites."""
    d = doc.diorama
    W = style.tile // 8 * 8 or 32
    h, lp = W // 2, W // 4
    base = min(0, int(d.height.min())) - 1
    hi = int(d.height.max())
    used = sorted(set(d.terrain.ravel()))
    levels = list(range(min(0, int(d.height.min())), hi + 1))
    S = h + (hi - base) * lp
    atlas = np.zeros((S * len(used), W * len(levels), 4), np.uint8)
    for ti, name in enumerate(used):
        for li, lvl in enumerate(levels):
            col = _column(name, lvl, base, W)
            atlas[ti * S + S - col.shape[0]:(ti + 1) * S, li * W:(li + 1) * W] = col
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(out / "iso_columns.png")
    y_level0 = S - h // 2 + base * lp                   # the level-0 diamond centre in a slot
    origin = y_level0 - S // 2                         # Godot: +y lifts the texture
    tiles = [(li, ti, {"texture_origin": f"Vector2i(0, {origin})"})
             for ti in range(len(used)) for li in range(len(levels))]
    lines = _tres_atlas("iso_columns.png", W, S, tiles,
                        extra_resource=["tile_shape = 1", "tile_layout = 5", f"tile_size = Vector2i({W}, {h})"])
    (out / "iso_columns.tres").write_text("\n".join(lines), encoding="utf-8")
    files = [out / "iso_columns.png", out / "iso_columns.tres"]
    cells = [(c, r, 0, levels.index(int(d.height[r, c])), used.index(d.terrain[r, c]), 0)
             for r in range(d.rows) for c in range(d.cols)]
    (out / "props").mkdir(exist_ok=True)
    fences = {(c, r) for k, c, r in d.props if k == "fence"}
    ext, sprites = {}, []
    for kind, c, r in d.props:
        snow = d.terrain[r, c] == "snow"
        along = (c - 1, r) in fences or (c + 1, r) in fences or not ((c, r - 1) in fences or (c, r + 1) in fences)
        key = f"{kind}{'_snow' if snow else ''}{'' if along or kind != 'fence' else '_r'}"
        if kind == "battlement":
            sides = battlement_sides(d.height, c, r)
            key = "battlement_" + "".join(f"{'pmz'[dc + 1]}{'pmz'[dr + 1]}" for dc, dr in sides)
            img, anchor = battlement_sprite(W, sides)
        else:
            img, anchor = prop_sprite(kind, W, snow, along)
        if key not in ext:
            p = out / "props" / f"{key}.png"
            Image.fromarray(img).save(p)
            files.append(p)
            ext[key] = (f"{len(ext) + 2}_{key}", img.shape, anchor)
        sprites.append((key, c, r))
    lines = [f'[gd_scene load_steps={len(ext) + 2} format=3]', "",
             '[ext_resource type="TileSet" path="res://iso_columns.tres" id="1_cols"]']
    lines += [f'[ext_resource type="Texture2D" path="res://props/{k}.png" id="{v[0]}"]' for k, v in ext.items()]
    cx = (d.cols - d.rows) * W / 4 + W / 2
    cy = (d.cols + d.rows) * h / 4 - hi * lp / 2
    lines += ["", f'[node name="{_node_name(d.name)}" type="Node2D"]', "y_sort_enabled = true",
              f"metadata/cols = {d.cols}", f"metadata/rows = {d.rows}", f"metadata/level_px = {lp}", "",
              '[node name="Blocks" type="TileMapLayer" parent="."]', "y_sort_enabled = true",
              f"tile_map_data = {_tilemap_data(cells)}", 'tile_set = ExtResource("1_cols")', "",
              '[node name="Props" type="Node2D" parent="."]', "y_sort_enabled = true", ""]
    for i, (key, c, r) in enumerate(sprites):
        rid, shape, (ax, ay) = ext[key]
        gx, gy = (c - r) * W / 2 + W / 2, (c + r) * h / 2 + h / 2      # map_to_local(c, r)
        lvl = int(d.height[r, c])
        lines += [f'[node name="P{i}" type="Node2D" parent="Props"]', f"position = Vector2({gx}, {gy + 1})", "",
                  f'[node name="Sprite" type="Sprite2D" parent="Props/P{i}"]',
                  f'texture = ExtResource("{rid}")', "centered = false",
                  f"offset = Vector2({-ax}, {-ay - lvl * lp - 1})", ""]
    lines += ['[node name="Spawns" type="Node2D" parent="."]', ""]
    for i, (team, c, r) in enumerate(doc.spawns):
        lvl = int(d.height[r, c])
        lines += [f'[node name="{"Blue" if team == 0 else "Red"}{i}" type="Marker2D" parent="Spawns"]',
                  f"position = Vector2({(c - r) * W / 2 + W / 2}, {(c + r) * h / 2 + h / 2 - lvl * lp})",
                  f"metadata/team = {team}", f"metadata/cell = Vector2i({c}, {r})", ""]
    lines += ['[node name="Camera" type="Camera2D" parent="."]', f"position = Vector2({cx:.1f}, {cy:.1f})",
              "zoom = Vector2(2, 2)", ""]
    scene = out / "iso_map.tscn"
    scene.write_text("\n".join(lines), encoding="utf-8")
    files.append(scene)
    files += _write_project(out, d.name, "res://iso_map.tscn")
    return files


def _tsj(name: str, png: str, img: np.ndarray, T: int, props: list[list[dict]]) -> dict:
    n = img.shape[1] // T
    return {"type": "tileset", "version": "1.10", "tiledversion": "1.10.2", "name": name,
            "image": png, "imagewidth": int(img.shape[1]), "imageheight": int(img.shape[0]),
            "tilewidth": T, "tileheight": T, "tilecount": n, "columns": n, "margin": 0, "spacing": 0,
            "tiles": [{"id": i, "properties": p} for i, p in enumerate(props)]}


def _export_tiled(doc: MapDoc, out: Path, style: assets.Style, look=None) -> list[Path]:
    """Tiled .tmj (orthogonal): terrain layer, height layer (gid = level tile, + "level"
    property per tile), props and spawns object layers; tilesets as .tsj + .png. With a look
    (map_look) the terrain layer is data only (hidden) and the tiles the map editor shows come
    above it: "ground" (look_ground.tsj, tile probabilities) + a layer per set (look_<key>)."""
    d = doc.diorama
    T = style.tile
    out.mkdir(parents=True, exist_ok=True)
    tex = terrain_textures(T)
    timg = np.concatenate([tex[n][0] for n in TERRAINS], 1)
    levels = max(MAX_LEVEL + 1, int(d.height.max()) + 1)
    himg = _height_tiles(levels, T)
    kinds = list(PROP_KINDS) + sorted({k for k, _, _ in d.props} - set(PROP_KINDS))
    pimg = _icon_strip(kinds, T)
    sets = [("terrain", timg, [[{"name": "terrain", "type": "string", "value": n},
                                {"name": "walkable", "type": "bool", "value": n in WALKABLE}] for n in TERRAINS]),
            ("heights", himg, [[{"name": "level", "type": "int", "value": k}] for k in range(levels)]),
            ("props", pimg, [[{"name": "kind", "type": "string", "value": k},
                              {"name": "blocking", "type": "bool", "value": k in BLOCKING_PROPS}] for k in kinds])]
    files, first, gid = [], {}, 1
    for name, img, props in sets:
        Image.fromarray(img).save(out / f"{name}.png")
        p = out / f"{name}.tsj"
        p.write_text(json.dumps(_tsj(name, f"{name}.png", img, T, props), indent=1), encoding="utf-8")
        files += [out / f"{name}.png", p]
        first[name] = gid
        gid += img.shape[1] // T
    look_layers = []
    if look is not None:
        for layer in look.layers(d, doc.seed):
            name = f"look_{layer.key}"
            img = layer.atlas
            cols_ = img.shape[1] // T
            ts = _tsj(name, f"{name}.png", img, T, [])
            ts.update(tilecount=cols_ * (img.shape[0] // T), columns=cols_, tiles=[])
            weights = ([(j, i, w) for i, n in enumerate(look.terrains) for j, w in enumerate(look._plain[n][1])]
                       if layer.set is None else
                       [(c, r, w) for xys, ws in layer.set.choices.values() for (c, r), w in zip(xys, ws)])
            ts["tiles"] = [{"id": r * cols_ + c, "probability": w} for c, r, w in weights if w != 1]
            Image.fromarray(img).save(out / f"{name}.png")
            (out / f"{name}.tsj").write_text(json.dumps(ts, indent=1), encoding="utf-8")
            files += [out / f"{name}.png", out / f"{name}.tsj"]
            sets.append((name, img, []))
            first[name] = gid
            data = [0 if x < 0 else gid + int(y) * cols_ + int(x)
                    for x, y in layer.cells.reshape(-1, 2).tolist()]
            gid += ts["tilecount"]
            look_layers.append({"name": "ground" if layer.set is None else layer.key, "data": data})
    tdata = [first["terrain"] + TERRAINS.index(d.terrain[r, c]) for r in range(d.rows) for c in range(d.cols)]
    hdata = [first["heights"] + max(0, int(d.height[r, c])) for r in range(d.rows) for c in range(d.cols)]
    oid = 1
    pobjs = []
    for kind, c, r in d.props:
        pobjs.append({"id": oid, "name": kind, "type": "prop", "gid": first["props"] + kinds.index(kind),
                      "x": c * T, "y": (r + 1) * T, "width": T, "height": T, "rotation": 0, "visible": True,
                      "properties": [{"name": "kind", "type": "string", "value": kind},
                                     {"name": "level", "type": "int", "value": int(d.height[r, c])}]})
        oid += 1
    sobjs = []
    for team, c, r in doc.spawns:
        sobjs.append({"id": oid, "name": "blue" if team == 0 else "red", "type": "spawn", "point": True,
                      "x": c * T + T / 2, "y": r * T + T / 2, "width": 0, "height": 0, "rotation": 0,
                      "visible": True, "properties": [{"name": "team", "type": "int", "value": team},
                                                      {"name": "col", "type": "int", "value": c},
                                                      {"name": "row", "type": "int", "value": r}]})
        oid += 1
    base = {"x": 0, "y": 0, "opacity": 1, "visible": True}
    tmj = {"type": "map", "version": "1.10", "tiledversion": "1.10.2", "orientation": "orthogonal",
           "renderorder": "right-down", "width": d.cols, "height": d.rows, "tilewidth": T, "tileheight": T,
           "infinite": False, "compressionlevel": -1, "nextlayerid": 5 + len(look_layers),
           "nextobjectid": oid,
           "properties": [{"name": "name", "type": "string", "value": d.name},
                          {"name": "level_height", "type": "float", "value": LEVEL},
                          {"name": "water_level", "type": "float", "value": d.water_level},
                          {"name": "template", "type": "string", "value": doc.template}],
           "tilesets": [{"firstgid": first[n], "source": f"{n}.tsj"} for n, _, _ in sets],
           "layers": [
               {"id": 1, "name": "terrain", "type": "tilelayer", "width": d.cols, "height": d.rows,
                "data": tdata, **base, **({"visible": False} if look_layers else {})},
               *[{"id": 5 + i, "name": ll["name"], "type": "tilelayer", "width": d.cols,
                  "height": d.rows, "data": ll["data"], **base} for i, ll in enumerate(look_layers)],
               {"id": 2, "name": "height", "type": "tilelayer", "width": d.cols, "height": d.rows,
                "data": hdata, **base, "opacity": 0.5,
                "properties": [{"name": "meaning", "type": "string",
                                "value": "tile = height level (property 'level'); 1 level = half a tile"}]},
               {"id": 3, "name": "props", "type": "objectgroup", "draworder": "topdown", "objects": pobjs, **base},
               {"id": 4, "name": "spawns", "type": "objectgroup", "draworder": "topdown", "objects": sobjs, **base},
           ]}
    p = out / "map.tmj"
    p.write_text(json.dumps(tmj, indent=1), encoding="utf-8")
    files.append(p)
    return files


HEIGHT_SCALE = 4096


def _export_json(doc: MapDoc, out: Path, style: assets.Style, look=None) -> list[Path]:
    """map.json + height.png (16-bit: value = (level - offset) * scale) + terrain.png (8-bit
    index into the terrains list) + colour.png (top-down) + props.json + legend.json."""
    d = doc.diorama
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = min(0, int(d.height.min())), int(d.height.max())
    scale = HEIGHT_SCALE if (hi - lo) * HEIGHT_SCALE <= 65535 else 65535 // max(1, hi - lo)
    h16 = ((d.height.astype(np.int64) - lo) * scale).astype(np.uint16)
    Image.fromarray(h16).save(out / "height.png")
    idx = np.vectorize(TERRAINS.index)(d.terrain).astype(np.uint8)
    Image.fromarray(idx, "L").save(out / "terrain.png")
    Image.fromarray(render_topdown(doc, style.tile, look=look)).save(out / "colour.png")
    (out / "map.json").write_text(json.dumps(doc.to_dict(), indent=1), encoding="utf-8")
    (out / "props.json").write_text(json.dumps(
        [{"kind": k, "col": c, "row": r, "level": int(d.height[r, c]), "blocking": k in BLOCKING_PROPS}
         for k, c, r in d.props], indent=1), encoding="utf-8")
    (out / "legend.json").write_text(json.dumps({
        "cols": d.cols, "rows": d.rows, "terrains": list(TERRAINS), "walkable": sorted(WALKABLE),
        "height_scale": scale, "height_offset": lo, "level_tiles": LEVEL, "tile_px": style.tile,
        "spawns": [{"team": t, "col": c, "row": r} for t, c, r in doc.spawns],
        "tilesets": {k: s.name for k, s in look.sets.items()} if look is not None else {},
        "files": {"height.png": "16-bit grey, level = value / height_scale + height_offset",
                  "terrain.png": "8-bit, terrain = terrains[value]", "colour.png": "top-down picture",
                  "props.json": "props with their cell and level"}}, indent=1), encoding="utf-8")
    return [out / n for n in ("map.json", "height.png", "terrain.png", "colour.png", "props.json", "legend.json")]


_EXPORTERS = {"godot-3d": _export_godot3d, "godot-tilemap": _export_tilemap, "godot-iso": _export_iso,
              "tiled": _export_tiled, "json": _export_json}


def export_map(asset, target: str, out_dir) -> list[Path]:
    """Export a map (Asset, its folder, a map.json or a MapDoc) to `target` (one of TARGETS)
    into out_dir. Sizes come from the asset's project Style (tile, ppu)."""
    if target not in _EXPORTERS:
        raise ValueError(f"Unknown export target '{target}' (use {', '.join(TARGETS)})")
    doc = load_map(asset)
    if isinstance(asset, assets.Asset):
        style = _project_of(asset).style
        doc.diorama.name = asset.name
    else:
        style = assets.current_project().style
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proj = _project_of(asset) if isinstance(asset, assets.Asset) else None
    look = map_look(doc, proj, style.tile) if target != "godot-iso" else None
    if target == "godot-tilemap":
        return _export_tilemap(doc, out, style, proj, look)
    if target == "godot-iso":
        return _export_iso(doc, out, style)
    return _EXPORTERS[target](doc, out, style, look)


# --------------------------------------------------------------------------- project types

# assets.Style.tile_style -> (default map template, default map export) for a project type
TILE_STYLE_DEFAULTS = {"hd2d": ("harbour", "godot-3d"), "topdown": ("village", "godot-tilemap"),
                       "iso": ("castle", "godot-iso"), "side": ("flat", "godot-tilemap"),
                       "none": ("flat", "json")}


def defaults_for(style=None) -> tuple[str, str]:
    """(template, export target) a new map in this project starts with (project types). The
    type's own template wins when assets.PROJECT_TYPES names one; Unity projects export Tiled
    maps (SuperTiled2Unity imports them), or plain json for a 3D (HD-2D) map."""
    style = style or assets.current_project().style
    template, target = TILE_STYLE_DEFAULTS.get(getattr(style, "tile_style", "hd2d"),
                                               TILE_STYLE_DEFAULTS["hd2d"])
    t = assets.PROJECT_TYPES.get(getattr(style, "project_type", "custom"), {})
    if t.get("map") in _BUILDERS and getattr(style, "project_type", "custom") != "custom":
        template = t["map"]
    if getattr(style, "engine", "godot") == "unity":
        target = "json" if target in ("godot-3d", "json") else "tiled"
    return template, target
