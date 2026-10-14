"""A tactics battle map with height: the board shared by the three looks of the HD-2D demo
(real 3D, fake 3D pixel camera, isometric 2D).

    d = harbour(seed=0)        # stone tiers, stairs, a cobble quay, a wooden pier over water
    d.save("map.json")         # what the Godot scenes read
    tex = textures(d)          # {terrain: (top RGBA, side RGBA)} seamless neo 16-bit tiles

Cells: height in levels (0 = water level; one level = half a tile tall, like Triangle
Strategy's steps), a terrain name, and props (crates, barrels, torches, posts, fences).
Water cells are not walkable; a unit can climb `jump` levels per step.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


TERRAINS = ("grass", "stone", "cobble", "planks", "dirt", "sand", "water", "snow")
WALKABLE = {"grass", "stone", "cobble", "planks", "dirt", "sand", "snow"}
LEVEL = 0.5          # one height level in tiles (world units: 1 tile = 1.0)


# Class stats: hp, attack, defence, move, jump, reach (attack distance in cells).
CLASSES = {
    "knight":    (34, 12, 8, 4, 2, 1), "swordsman": (30, 13, 6, 4, 2, 1),
    "paladin":   (40, 11, 11, 3, 1, 1), "spearman": (30, 12, 7, 4, 2, 2),
    "archer":    (24, 11, 4, 4, 2, 4), "mage":      (22, 14, 3, 3, 2, 3),
    "physician": (24, 6, 4, 4, 2, 1), "cleric":    (26, 8, 6, 4, 2, 1),
    "rogue":     (24, 12, 4, 5, 3, 1), "dog":       (22, 9, 3, 5, 2, 1),
    "wolf":      (26, 11, 4, 5, 2, 1), "bear":      (42, 15, 7, 3, 1, 1),
}
DEFAULT_STATS = (28, 10, 5, 4, 2, 1)


@dataclass
class Unit:
    name: str          # sprite set name, e.g. "dog", "knight"
    col: int
    row: int
    team: int = 0
    facing: str = "S"  # one of the 16 direction names (E, ESE, SE, ...)
    hp: int = 0        # 0 = from CLASSES by name
    attack: int = 0
    defence: int = 0
    move: int = 0
    jump: int = 0
    reach: int = 0

    def __post_init__(self):
        stats = CLASSES.get(self.name, DEFAULT_STATS)
        for field_name, value in zip(("hp", "attack", "defence", "move", "jump", "reach"), stats):
            if not getattr(self, field_name):
                setattr(self, field_name, value)


def damage(attacker: Unit, target: Unit, attacker_level: int = 0, target_level: int = 0) -> int:
    """attack - defence/2, +25% from higher ground, at least 1 (no dice: easy to plan)."""
    dmg = attacker.attack - target.defence // 2
    if attacker_level > target_level:
        dmg = dmg * 5 // 4
    return max(1, dmg)


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
    sea: bool = True                   # False: water only inside the map (no sea around it)

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
            "sea": bool(self.sea),
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
                   d.get("name", "map"), bool(d.get("sea", True)))


BLOCKING_PROPS = {"crate", "barrel", "post", "tree", "rock", "house", "wall", "well", "fence", "brazier", "crystal"}


def place_units(d: Diorama, blue: list[str], red: list[str]) -> Diorama:
    """Replace the map's units: blue starts at the front (high rows: pier and quay), red at
    the back (low rows: the town), each on free walkable cells spread across the width."""
    free = [(c, r) for r in range(d.rows) for c in range(d.cols) if d.walkable(c, r)]
    front = sorted(free, key=lambda p: (-p[1], abs(p[0] - d.cols / 3)))
    back = sorted(free, key=lambda p: (p[1], abs(p[0] - d.cols / 2)))
    units, taken = [], set()

    def put(names, cells, team, facing):
        for name in names:
            for c, r in cells:
                if (c, r) not in taken and all(abs(c - tc) + abs(r - tr) > 1 for tc, tr in taken):
                    taken.add((c, r))
                    units.append(Unit(name, c, r, team, facing))
                    break
    put(blue, front, 0, "N")
    put(red, back, 1, "S")
    d.units = units
    return d


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

def textures(d: Diorama | None = None, size: int = 32, seed: int = 0) -> dict:
    """{terrain: (top RGBA, side RGBA)} for every terrain the map uses (all, if no map): the
    neo 16-bit procedural look (neo_terrain: hue-shifted ramps, lit stones, boards and tufts;
    rock strata under soft ground, ashlar under stone, boards under planks)."""
    from .neo_terrain import paint_side, paint_top
    names = sorted(set(d.terrain.ravel())) if d is not None else list(TERRAINS)
    return {name: (paint_top(name, size, seed)[0], paint_side(name, size, seed)[0]) for name in names}


def write_textures(d: Diorama, folder, size: int = 32, seed: int = 0, look=None) -> dict:
    """Save the board's textures (neo_terrain.write_board_textures: <terrain>_top.png /
    _side.png as before, plus per-cell top variants, normal maps, wall-top overhangs and
    terrain.json). look: a tilegen.MapLook whose saved tilesets texture the terrain (None =
    procedural). Returns {terrain: {"top": name, "side": name}}."""
    from .neo_terrain import write_board_textures
    names = sorted(set(d.terrain.ravel()))
    write_board_textures(names, folder, size, seed, look)
    return {name: {"top": f"{name}_top.png", "side": f"{name}_side.png"} for name in names}


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
