"""Words that shape drawn ground tiles (2026-10-16, Kevin: "words should shape soft terrains, not
just colours"). The drawn method (tilepaint.drawn_tile) draws grass, dirt, sand, snow and stone by
code; SDXL's swatch only gives the colours. Here the user's words become shape parameters:

    d = details("dirt", "mossy dirt path")      # Details(moss=1.0, words=("mossy",))
    idx = tilepaint.dirt_idx(32, 0, d)           # the index-level details (pebbles, cracks...)
    tex = stamp_details(tex, idx, "dirt", d, 0)  # coloured stamps (moss clumps, flowers, puddles)

Vocabulary (small on purpose; phrases first, whole words only):
  mossy / moss                      -> moss clumps (in joints and low spots)
  rocky / pebbly / gravel / stony   -> more and larger stones
  flowers / meadow / wildflowers    -> flower stamps (meadow: taller grass too)
  cracked / dry / parched / arid    -> crack networks (dry grass: fewer tufts)
  muddy / puddles / wet / boggy     -> puddles with a wet rim and a sky glint
  dune / ripples / windswept        -> longer, denser sand ripples (dune: big crest bands)
  fresh / powder                    -> snow with fewer drifts, more sparkle
  footprints / tracks / trodden     -> a trail of prints (snow, sand, dirt)
  tall grass / overgrown / lush     -> taller, denser blades
  autumn leaves / fallen leaves     -> leaf stamps in autumn colours
  short / mown / sparse / patchy    -> fewer tufts
"""
from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field, replace

import numpy as np


@dataclass(frozen=True)
class Details:
    """Shape parameters of a drawn ground tile (1.0 / 0 / False = the plain tile)."""
    pebbles: float = 1.0          # stones / pebbles / grains: count x
    big_pebbles: bool = False     # larger stones (rocky)
    moss: float = 0.0             # moss clumps
    flowers: float = 0.0          # flower stamps
    cracks: float = 0.0           # crack network
    puddles: float = 0.0          # puddles
    ripples: float = 1.0          # sand / snow ripples: count and length x
    dunes: bool = False           # big crest bands over the ripples
    footprints: bool = False      # a trail of prints
    fresh: bool = False           # fresh snow: few drifts, more sparkle
    tall: float = 0.0             # taller grass blades
    leaves: float = 0.0           # fallen leaves
    tufts: float = 1.0            # grass tuft count x
    words: tuple = field(default=(), compare=False)    # the vocabulary words found

    def plain(self) -> bool:
        return self == Details()


PLAIN = Details()

# (phrases, changes); a later match adds to / overrides an earlier one (max for amounts)
VOCAB: list[tuple[tuple[str, ...], dict]] = [
    (("tall grass", "long grass", "overgrown", "lush", "wild grass", "high grass"),
     {"tall": 1.0, "tufts": 1.4}),
    (("autumn leaves", "fallen leaves", "fall leaves", "dead leaves", "autumn", "autumnal", "leaves",
      "leafy"), {"leaves": 1.0}),
    (("mossy", "moss", "moss covered", "moss-covered", "overgrown"), {"moss": 1.0}),
    (("rocky", "rocks", "boulders", "stony"), {"pebbles": 2.4, "big_pebbles": True}),
    (("pebbly", "pebbles", "pebble", "gravel", "gravelly", "shingle"), {"pebbles": 2.2}),
    (("flower meadow", "flowers", "flower", "flowery", "flowering", "wildflowers", "wildflower",
      "blossoms", "daisies", "poppies"), {"flowers": 1.0}),
    (("meadow",), {"flowers": 0.7, "tall": 0.5, "tufts": 1.2}),
    (("cracked", "cracks", "cracking", "parched", "arid", "drought", "baked", "dried"),
     {"cracks": 1.0}),
    (("dry",), {"cracks": 0.8, "tufts": 0.75}),
    (("muddy", "mud", "puddles", "puddle", "puddled", "wet", "rainy", "boggy", "soggy", "marshy"),
     {"puddles": 1.0}),
    (("dune", "dunes", "rippled", "ripples", "ripple", "windswept", "wind-swept", "wind swept",
      "wavy"), {"ripples": 2.0}),
    (("dune", "dunes"), {"dunes": True}),
    (("fresh", "powder", "powdery", "pristine", "untouched", "freshly fallen"), {"fresh": True}),
    (("footprints", "footprint", "tracks", "trodden", "trampled", "footsteps", "prints"),
     {"footprints": True}),
    (("short", "mown", "trimmed", "manicured", "sparse", "patchy", "thin"), {"tufts": 0.55}),
]

# what each terrain can show (others are ignored for it: "mossy snow" has no moss)
USES = {
    "grass": {"tall", "tufts", "flowers", "leaves", "pebbles", "big_pebbles", "puddles", "moss",
              "cracks"},
    "dirt": {"pebbles", "big_pebbles", "moss", "cracks", "puddles", "leaves", "flowers",
             "footprints"},
    "sand": {"ripples", "dunes", "pebbles", "big_pebbles", "cracks", "footprints"},
    "snow": {"fresh", "footprints", "ripples", "pebbles", "big_pebbles"},
    "stone": {"moss", "cracks", "puddles", "leaves"},
}


def _has(text: str, phrase: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text) is not None


def details(terrain: str, words: str = "") -> Details:
    """The user's words -> Details for a drawn terrain (unknown words and terrains: plain)."""
    text = " " + re.sub(r"\s+", " ", (words or "").lower()) + " "
    uses = USES.get(terrain)
    if not uses or not text.strip():
        return PLAIN
    vals: dict = {}
    found: list[str] = []
    for phrases, changes in VOCAB:
        hit = next((p for p in phrases if _has(text, p)), None)
        if hit is None:
            continue
        # "tall grass" as a phrase is consumed, so its "grass" isn't counted twice
        took = False
        for k, v in changes.items():
            if k not in uses:
                continue
            took = True
            if isinstance(v, bool):
                vals[k] = vals.get(k, False) or v
            elif k == "tufts":
                vals[k] = vals.get(k, 1.0) * v
            else:
                vals[k] = max(vals.get(k, 0.0 if k not in ("pebbles", "ripples") else 1.0), v)
        if took and hit not in found:
            found.append(hit)
    if not vals:
        return PLAIN
    return replace(PLAIN, **vals, words=tuple(found))


def _rng(*key) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32("/".join(str(k) for k in key).encode()))


# --------------------------------------------------------------------------- index-level details

def crack_lines(t: int, rng, cell: float, keep: float = 0.8) -> np.ndarray:
    """A wrapping crack network (dried mud): the borders of a jittered Voronoi of ~cell px
    plates, 1 px wide, broken in places (keep = share kept)."""
    from .neo_terrain import _cells, pnoise
    g = max(2, int(round(t / cell)))
    _, _, edge = _cells(t, t, g, g, rng, 0.85)
    line = edge < 0.5
    # thin to 1 px where the border came out 2 px: keep the pixel with the smaller edge
    thick = line & np.roll(line, 1, 0) & np.roll(line, 1, 1)
    line &= ~thick
    f = pnoise(t, t, max(2, t // 6), max(2, t // 6), rng)
    return line & (f < np.quantile(f, keep))


def add_cracks(idx: np.ndarray, rng, amount: float, dark: int = 0, lit: int | None = None,
               cell: float | None = None) -> np.ndarray:
    """Crack lines in an index map: the line at `dark`, a lit lip on the plate below / right of
    it (lit = that index; None = one step up)."""
    t = idx.shape[0]
    line = crack_lines(t, rng, cell or max(5.0, t / 3.2), keep=0.5 + 0.3 * min(1.0, amount))
    out = idx.copy()
    lip = np.roll(line, 1, 0) & ~line                            # the plate edge just below
    hi = int(idx.max())
    out[lip] = np.minimum(hi, (idx[lip] + 1) if lit is None else lit)
    out[line] = dark
    return out


def footprint_trail(t: int, rng) -> list[tuple[int, int, int]]:
    """(y, x, side) of prints walking across a wrapping tile (a straight-ish diagonal, left /
    right feet alternating), so the trail runs on into the next tile."""
    steps = max(4, t // 4)
    dy, dx = t / steps, t / steps * float(rng.choice((-1, 1))) * 0.5   # wraps after `steps`
    y0, x0 = float(rng.integers(t)), float(rng.integers(t))
    out = []
    for k in range(steps):
        side = 1 if k % 2 else -1
        y = y0 + k * dy
        x = x0 + k * dx + side * max(1.0, t / 24)
        out.append((int(round(y)) % t, int(round(x)) % t, side))
    return out


def add_footprints(idx: np.ndarray, rng, deep: int, rim: int | None = None,
                   mid: int | None = None) -> np.ndarray:
    """A trail of prints: each a 2 x 3 px dent (2 x 2 below 24 px): its upper rows in shadow
    (deep), its heel a step lighter (mid), a lit rim of pushed-up snow / sand below it (rim)."""
    t = idx.shape[0]
    out = idx.copy()
    w, h = (2, 3) if t >= 24 else (1, 2)
    for y, x, _ in footprint_trail(t, rng):
        for j in range(h):
            for i in range(w):
                out[(y + j) % t, (x + i) % t] = deep if j < h - 1 or mid is None else mid
        if rim is not None:
            for i in range(w):
                out[(y + h) % t, (x + i) % t] = rim
    return out


def dune_bands(t: int, rng) -> np.ndarray:
    """-1 / 0 / +1 wide wavy crest bands across the tile (wrapping: whole periods in y, a whole
    number of wiggles along x)."""
    from .neo_terrain import _majority
    yy, xx = np.mgrid[0:t, 0:t].astype(float)
    ph = float(rng.random())
    wig = 0.12 * t * np.sin(2 * np.pi * (xx / t + ph))
    f = np.sin(2 * np.pi * (2 * (yy + wig) / t + ph))
    lit = _majority(f > 0.55)
    shade = _majority(f < -0.7)
    return lit.astype(int) - shade.astype(int)


# --------------------------------------------------------------------------- coloured stamps

MOSS = "16302a 22482f 37683a 5a8c3e 8cb24c"
FLOWERS = [((240, 238, 226), (242, 200, 64)),        # daisy: white petals, yellow eye
           ((246, 208, 70), (196, 116, 40)),          # buttercup
           ((236, 122, 164), (250, 230, 120)),        # pink
           ((152, 122, 226), (250, 232, 130)),        # violet
           ((222, 70, 66), (40, 30, 40))]             # poppy: red, dark eye
LEAVES = [(176, 64, 36), (212, 98, 34), (232, 148, 44), (240, 196, 76), (150, 48, 44), (128, 84, 44)]
PUDDLE = [(46, 60, 98), (78, 102, 146), (122, 152, 196), (206, 226, 246)]   # deep, body, sky, glint
PEBBLE = "3a3b4c 5d6274 8790a0 b4bcc4 dde2e2"


def _hex(keys: str) -> np.ndarray:
    from .neo_terrain import hex_ramp
    return hex_ramp(keys)


def stamp_details(tex: np.ndarray, idx: np.ndarray, terrain: str, d: Details, seed: int = 0,
                  ramp: np.ndarray | None = None) -> np.ndarray:
    """Coloured details over a drawn tile (wrapping, lit from the upper left): moss clumps,
    flowers, fallen leaves, puddles and grey pebbles in grass / sand / snow. idx: the tile's ramp
    index map (shadows step it down); ramp: its colours."""
    from .tilepaint import scatter
    if d.plain():
        return tex
    out = tex.copy()
    t = tex.shape[0]
    rng = _rng("stamps", terrain, t, seed, d.words)
    if ramp is None:
        ramp = np.array([out[idx == k][0, :3] if (idx == k).any() else (0, 0, 0)
                         for k in range(int(idx.max()) + 1)], np.uint8)
    taken = np.zeros((t, t), bool)

    def put(y, x, col):
        y, x = y % t, x % t
        out[y, x, :3] = col
        taken[y, x] = True

    def shade(ys, xs, steps=1):                     # ground under / beside a stamp, a step darker
        for y, x in zip(ys, xs):
            y, x = int(y) % t, int(x) % t
            if not taken[y, x]:
                out[y, x, :3] = ramp[max(0, int(idx[y, x]) - steps)]

    if d.puddles > 0 and terrain in ("dirt", "grass", "stone"):
        cols = np.array(PUDDLE, np.uint8)
        n = max(1, int(round(d.puddles * t * t / 420)))
        yy, xx = np.mgrid[0:t, 0:t]
        for cy, cx in scatter(t, n, rng, t / 2.2):
            rx = float(rng.uniform(max(2.0, t / 8), max(2.5, t / 5)))
            ry = rx * float(rng.uniform(0.5, 0.7))
            dy = (yy - cy + t / 2) % t - t / 2
            dx = (xx - cx + t / 2) % t - t / 2
            r = (dy / ry) ** 2 + (dx / rx) ** 2
            wob = 0.18 * np.sin(3 * np.arctan2(dy, dx) + float(rng.random()) * 6)
            inside = r < 1 + wob
            ring = (r < (1 + wob) * 1.55) & ~inside
            ys, xs = np.nonzero(ring & ~taken)
            shade(ys, xs, 1)                    # wet, darker rim
            body = np.where(dy < -ry * 0.35, 0, 1)                # the far bank's shadow
            glint = (np.abs(dy - ry * 0.15) < 0.6) & (dx > -rx * 0.55) & (dx < -rx * 0.05)
            level = np.where(glint, 3, np.where(dy > ry * 0.3, 2, body))
            for y, x in zip(*np.nonzero(inside)):
                put(y, x, cols[level[y, x]])
    if d.moss > 0 and terrain in ("dirt", "stone", "grass"):
        cols = _hex(MOSS)
        n = int(round(d.moss * t * t / (70 if terrain != "grass" else 110)))
        # moss likes the low spots: the darker half of the tile, joints first
        low = idx <= np.quantile(idx, 0.4)
        pts = scatter(t, n * 3, rng, max(2.5, t / 8))
        pts = sorted(pts, key=lambda p: (not low[p[0], p[1]], rng.random()))[:n]
        for y, x in pts:
            w, h = int(rng.integers(2, 5)), int(rng.integers(2, 4))
            for j in range(h):
                for i in range(w):
                    if (i in (0, w - 1) and j in (0, h - 1)) and rng.random() < 0.6:
                        continue                                  # rounded clump corners
                    lv = 2 + (1 if j == 0 else 0) + (1 if j == 0 and i == 1 else 0) - (1 if j == h - 1 else 0)
                    put(y + j, x + i, cols[min(4, max(0, lv))])
            shade([y + h] * w, [x + i for i in range(w)], 1)
    if d.pebbles > 1.0 and terrain in ("grass", "sand", "snow"):
        cols = _hex(PEBBLE)
        n = int(round((d.pebbles - 1.0) * t * t / 150))
        for y, x in scatter(t, n, rng, max(3.0, t / 6)):
            w = int(rng.integers(2, 4 if d.big_pebbles else 3))
            h = 2 if w >= 2 else 1
            for j in range(h):
                for i in range(w):
                    lv = 3 if (j == 0 and i == 0) else (1 if (j == h - 1 and i == w - 1) else 2)
                    put(y + j, x + i, cols[lv])
            shade([y + h] * w, [x + i + 1 for i in range(w)], 2)
    if d.flowers > 0 and terrain in ("grass", "dirt"):
        n = max(2, int(round(d.flowers * t * t / 60)))
        kinds = rng.choice(len(FLOWERS), size=min(3, len(FLOWERS)), replace=False)
        for k, (y, x) in enumerate(scatter(t, n, rng, max(2.5, t / 7))):
            petal, eye = FLOWERS[int(kinds[k % len(kinds)])]
            if t >= 24 and rng.random() < 0.7:                    # a 5-px flower
                for oy, ox in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    put(y + oy, x + ox, petal)
                put(y, x, eye)
                shade([y + 2], [x], 2)          # its shadow / stem
            else:
                put(y, x, petal)
                shade([y + 1], [x], 2)
    if d.leaves > 0 and terrain in ("grass", "dirt", "stone"):
        cols = np.array(LEAVES, np.uint8)
        n = int(round(d.leaves * t * t / 45))
        for y, x in scatter(t, n, rng, max(2.0, t / 9)):
            c = cols[int(rng.integers(len(cols)))]
            lit = np.clip(c.astype(int) + 34, 0, 255)
            if rng.random() < 0.5:                                # lying flat: 2 px across
                put(y, x, lit)
                put(y, x + 1, c)
                shade([y + 1, y + 1], [x + 1, x + 2], 1)
            else:                                                 # diagonal
                put(y, x, lit)
                put(y + 1, x + 1, c)
                shade([y + 2], [x + 1], 1)
    return out
