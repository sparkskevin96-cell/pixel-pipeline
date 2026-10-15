"""Is this picture ONE subject? (QA 2026-10-15: for single-subject requests SDXL drew sprite sheets,
several objects or a whole scene in ~60% of pictures.)

`assess(picture)` -> Verdict, on the CPU in ~20 ms, no models: the background is the colour of
the picture's border; everything not connected to the border in that colour is foreground; the
foreground is split into separate pieces (a few px of closing glues one object's parts).

- pieces: separate things bigger than a speck. The main piece + pieces touching its box (a fence's
  posts, a tree's loose leaves) are the subject; the others are extra things.
- sheet: many similar pieces (a sprite sheet / an item set / a character sheet).
- scene: the border isn't one plain colour, or the foreground fills the picture / runs off it.
- framing: main piece centred, not cropped, a sensible size; sharpness (mean edge strength
  inside the subject) only ranks good pictures.

`ok` = single subject, plain background, not cropped. `rank` orders pictures best first, and
`generate_single` (generate.py wrapper) redraws the failures with new seeds; every picture
stage (item icons, drawn props, prop/character 3D pictures, portrait subjects) uses it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .color import rgb_to_lab

WORK = 256                 # analysis size (longest side)
BG_TOL = 9.0               # Lab distance that still counts as background colour
SPECK = 0.0015             # pieces smaller than this share of the picture are specks
BIG = 0.08                 # an extra piece this share of the main one is "another thing"
EXTRA_SHARE = 0.10         # extra things may hold at most this share of the foreground
REACH = 0.03               # pieces touching the main box grown by this much belong to it
SHEET_PIECES = 5           # this many real pieces (besides the subject) = a sheet / a set
PLAIN = 0.80               # share of the border in the background colour for a plain background
FLAT = 0.45                # more of the subject than this in one colour = a flat silhouette
GRAD_STEP = 3              # per-pixel colour step a smooth background gradient may take


@dataclass
class Verdict:
    ok: bool
    score: float
    pieces: int = 0            # real pieces (not specks), the subject's own included
    extras: int = 0            # separate big things besides the subject
    extra_share: float = 0.0   # their share of the foreground
    sheet: bool = False
    scene: bool = False
    cropped: bool = False
    plain: float = 1.0         # border share in the background colour
    fill: float = 0.0          # foreground share of the picture
    size: float = 0.0          # subject box, longest side / picture side
    offset: float = 0.0        # subject centre from the picture centre (0 = centred, 1 = edge)
    sharp: float = 0.0
    flat: float = 0.0          # share of the subject in one colour
    grey: float = 0.0          # chroma of that colour (the start blob is grey)
    why: list = field(default_factory=list)

    def reason(self) -> str:
        return ", ".join(self.why) if self.why else "one subject"

    def summary(self) -> dict:
        """What asset.json keeps per picture (asset.extra["checks"])."""
        return {"ok": bool(self.ok), "why": self.reason(), "score": round(float(self.score), 3)}


def _rgb(picture) -> np.ndarray:
    if hasattr(picture, "convert"):                       # PIL
        picture = np.asarray(picture.convert("RGBA"))
    a = np.asarray(picture)
    if a.ndim == 2:
        a = np.dstack([a] * 3)
    if a.shape[2] == 4:                                   # transparent -> on white
        al = a[..., 3:4].astype(np.float32) / 255
        a = (a[..., :3].astype(np.float32) * al + 255 * (1 - al)).astype(np.uint8)
    return np.ascontiguousarray(a[..., :3])


def foreground(rgb: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """(mask of the subject pixels, border plainness, background Lab) at the given size."""
    h, w = rgb.shape[:2]
    lab = rgb_to_lab(rgb)
    b = max(2, round(0.02 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    plain = float((np.linalg.norm(border - bg, axis=1) < BG_TOL * 1.5).mean())
    near = (np.linalg.norm(lab - bg, axis=2) < BG_TOL).astype(np.uint8)
    n, lab_ids = cv2.connectedComponents(near, connectivity=4)
    edge = np.unique(np.concatenate([lab_ids[0], lab_ids[-1], lab_ids[:, 0], lab_ids[:, -1]]))
    edge = edge[edge > 0]
    background = np.isin(lab_ids, edge) & (near > 0)
    # smooth studio gradients / vignettes (grey to white) and soft floor shadows are background
    # too: flood from the border where neighbouring pixels change only a little
    smooth = cv2.GaussianBlur(rgb, (3, 3), 0)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (1 << 8)
    ring = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] + \
        [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
    for x, y in ring[::2]:            # seeds: border pixels in the background colour only (a
        if background[y, x] and not mask[y + 1, x + 1]:      # flat subject at the edge isn't)
            cv2.floodFill(smooth, mask, (x, y), 0, (GRAD_STEP,) * 3, (GRAD_STEP,) * 3, flags)
    background = background | (mask[1:-1, 1:-1] > 0)
    edge_px = np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]])
    plain = max(plain, float(edge_px.mean()))
    return ~background, plain, bg


def assess(picture, silhouette: bool = False) -> Verdict:
    """How single-subject a generated picture is (see the module text). silhouette: also fail a
    flat grey one-colour subject (a picture drawn from the start blob that kept the blob)."""
    rgb = _rgb(picture)
    h, w = rgb.shape[:2]
    k = WORK / max(h, w)
    if k < 1:
        rgb = cv2.resize(rgb, (max(1, round(w * k)), max(1, round(h * k))), interpolation=cv2.INTER_AREA)
    h, w = rgb.shape[:2]
    fg, plain, _ = foreground(rgb)
    m = fg.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    total = float(h * w)
    fill = float(m.sum()) / total
    v = Verdict(ok=False, score=0.0, plain=plain, fill=fill)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
    real = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= SPECK * total]
    if not real:
        v.why.append("empty")
        return v
    main = max(real, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    area = int(stats[main, cv2.CC_STAT_AREA])
    # pieces within REACH of the main piece's pixels belong to it (loose leaves, a staff tip);
    # a box test grouped a tall figure with the sketches drawn beside it
    r = max(1, round(REACH * max(w, h)))
    near_main = cv2.dilate((labels == main).astype(np.uint8),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    touching = set(np.unique(labels[near_main > 0]).tolist())
    group, extra = [main], []
    for i in real:
        if i != main:
            (group if i in touching else extra).append(i)
    fg_area = float(sum(stats[i, cv2.CC_STAT_AREA] for i in real))
    big = [i for i in extra if stats[i, cv2.CC_STAT_AREA] >= BIG * area]
    v.pieces = len(real)
    v.extras = len(big)
    v.extra_share = float(sum(stats[i, cv2.CC_STAT_AREA] for i in extra)) / fg_area
    # sheets: many real pieces, or several extra things of about one size
    sizes = np.array([stats[i, cv2.CC_STAT_AREA] for i in extra], float)
    similar = int(((sizes >= 0.25 * np.median(sizes)) & (sizes <= 4 * np.median(sizes))).sum()) \
        if len(sizes) else 0
    v.sheet = len(extra) >= SHEET_PIECES or (similar >= 3 and v.extra_share > 0.15)
    # the subject's box (main + its own pieces)
    gx0 = min(stats[i, 0] for i in group)
    gy0 = min(stats[i, 1] for i in group)
    gx1 = max(stats[i, 0] + stats[i, 2] for i in group)
    gy1 = max(stats[i, 1] + stats[i, 3] for i in group)
    v.size = max((gx1 - gx0) / w, (gy1 - gy0) / h)
    cx, cy = (gx0 + gx1) / 2 / w - 0.5, (gy0 + gy1) / 2 / h - 0.5
    v.offset = float(2 * max(abs(cx), abs(cy)))
    edge_px = 1
    touch = sum(int(t) for t in (gx0 < edge_px, gy0 < edge_px, gx1 > w - edge_px, gy1 > h - edge_px))
    v.cropped = touch >= 2 or (touch == 1 and v.size > 0.9)
    v.scene = plain < PLAIN or fill > 0.70
    # sharpness: mean edge strength inside the subject (ranking only)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
    inside = labels == main
    v.sharp = float(np.hypot(gx, gy)[inside].mean() / 255) if inside.any() else 0.0
    # a flat one-colour silhouette (the img2img start blob left unpainted) is no picture
    if inside.sum() > 50:
        labs = rgb_to_lab(rgb)[inside]
        med = np.median(labs, axis=0)
        v.flat = float((np.linalg.norm(labs - med, axis=1) < 7).mean())
        v.grey = float(np.hypot(med[1], med[2]))
    if v.sheet:
        v.why.append(f"sheet of {len(real)} things")
    elif v.extras:
        v.why.append(f"{v.extras + 1} separate things")
    elif v.extra_share > EXTRA_SHARE:
        v.why.append("loose bits around it")
    if v.scene:
        v.why.append("scene / busy background" if plain < PLAIN else "fills the picture")
    if v.cropped:
        v.why.append("cut off at the edge")
    if v.size < 0.2:
        v.why.append("tiny")
    if silhouette and v.flat > FLAT and v.grey < 14:
        v.why.append("flat silhouette")
    v.ok = not v.why
    # score: 1 = perfect; good pictures always score above bad ones
    s = 1.0 - 1.5 * v.extra_share - 0.3 * v.offset - 0.3 * max(0.0, 0.35 - v.size) \
        - 0.2 * max(0.0, v.size - 0.95) + 0.2 * min(1.0, v.sharp * 4)
    s -= 0.15 * max(0.0, PLAIN - plain) / PLAIN
    v.score = float(s + (1.0 if v.ok else 0.0) - (0.5 if v.sheet else 0.0)
                    - 0.2 * min(v.extras, 5) - (0.4 if v.scene else 0.0) - (0.3 if v.cropped else 0.0))
    return v


def rank(pictures) -> list[int]:
    """Indexes of the pictures, best first (single subject, centred, sharp)."""
    vs = [assess(p) for p in pictures]
    return sorted(range(len(vs)), key=lambda i: (-vs[i].score, i))


# --------------------------------------------------------------------------- generation

CHECK_ENV = "PIXEL_SINGLE"         # 0 = no check / redraws (pictures in the order drawn)


def check_on() -> bool:
    import os
    return os.environ.get(CHECK_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


@dataclass
class Pick:
    image: object              # PIL image as generated
    seed: int
    verdict: Verdict


def generate_single(positive: str, negative: str = "", *, count: int = 1, redraws: int | None = None,
                    seed: int | None = None, client=None, check=None, say=None,
                    init: str | None = None, init_denoise: float = 0.9, init_noise: float = 0.0,
                    init_words: str = "",
                    **gen_kw) -> list[Pick]:
    """generate.generate_images for ONE-subject pictures: draw `count`, assess each, redraw the
    failures with new seeds (up to `redraws` extra pictures; default = count, at least 2), and
    return the best `count`, best first (so "picture 1" / a batch's blind pick is the best one).

    check(image) -> reason or None: an extra test (e.g. the 3D picture's cut-out has 2 pieces).
    init: an INIT_SHAPES name: start from one soft blob there (img2img at init_denoise, blob
    filled with colour noise of strength init_noise), which sets the layout to one centred thing
    (the 2026-10-15 A/B: 21/21 single vs 8/21 from an empty canvas).
    PIXEL_SINGLE=0 turns the check, the redraws and the start blob off."""
    from .generate import generate_images
    say = say or (lambda text: None)
    if init and check_on() and gen_kw.get("init_image") is None:
        size = max(int(gen_kw.get("width") or 1024), int(gen_kw.get("height") or 1024))
        gen_kw.update(init_image=single_init(init, size=size, noise=init_noise,
                                             colour=init_colour(init_words)),
                      denoise=init_denoise)
    pics = generate_images(positive, negative, client=client, seed=seed, count=count, **gen_kw)
    if not check_on():
        return [Pick(img, s, Verdict(ok=True, score=0.0)) for img, s in pics]

    def judge(img):
        v = assess(img, silhouette=gen_kw.get("init_image") is not None)
        if v.ok and check is not None:
            why = check(img)
            if why:
                v.ok, v.score = False, v.score - 1.0
                v.why.append(why)
        return v

    picks = [Pick(img, int(s), judge(img)) for img, s in pics]
    budget = max(2, count) if redraws is None else max(0, int(redraws))
    next_seed = max(p.seed for p in picks) + 1 if picks else (seed or 0)
    while budget > 0:
        good = sum(p.verdict.ok for p in picks)
        if good >= count:
            break
        n = min(count - good, budget)
        bad = [p.verdict.reason() for p in picks if not p.verdict.ok]
        say(f"{len(bad)} of the pictures {'were' if len(bad) > 1 else 'was'} not one clean subject "
            f"({bad[0]}); drawing {n} more…")
        more = generate_images(positive, negative, client=client, seed=next_seed % 2**64, count=n,
                               **gen_kw)
        picks += [Pick(img, int(s), judge(img)) for img, s in more]
        next_seed += n
        budget -= n
    order = sorted(range(len(picks)), key=lambda i: (-picks[i].verdict.score, i))
    return [picks[i] for i in order[:count]]


# what SDXL must not lay out for a one-subject request (negatives of every picture stage)
SHEET_NEGATIVE = ("sprite sheet, item set, collection, grid, multiple objects, many items, "
                  "inventory, character sheet, reference sheet, turnaround, several views, "
                  "duplicates, pattern")

# per kind of picture: generation settings that keep it one subject (from the 2026-10-15 A/B)
SINGLE_OPTIONS: dict[str, dict] = {
    # 0.93: one subject every time in the A/B, a little freer than 0.9 (a forge, not a well)
    "icon": {"init_denoise": 0.93},
    "prop": {"init_denoise": 0.93},
    # painted 3D/portrait pictures (no pixel LoRA) kept the blob as a grey ghost at 0.9; noise
    # in the blob washed them out; 0.97 keeps the layout and paints it fully
    "model": {"init_denoise": 0.97},
    "model-object": {"init_denoise": 0.97},
}


def single_options(kind: str) -> dict:
    """Extra generate_images options for a single-subject picture of this kind
    (icon / prop / model / model-object)."""
    return dict(SINGLE_OPTIONS.get(kind, {}))


# one soft blob in the middle: an img2img start that says "ONE thing, here" (composition is
# set in the first steps; from an empty canvas SDXL often lays out a sheet)
INIT_SHAPES = {"object": (0.52, 0.62), "weapon": (0.16, 0.80), "person": (0.36, 0.86),
               "bottle": (0.34, 0.62), "tall": (0.36, 0.80), "flat": (0.66, 0.40),
               "animal": (0.70, 0.55), "building": (0.66, 0.62)}


def single_init(shape: str = "object", size: int = 1024, grey: int = 185, noise: float = 0.0,
                seed: int = 0, colour=None):
    """White square with one soft blob of the shape's (width, height) share, centred. noise > 0
    fills the blob with blurred colour noise of that strength (0..1) instead of flat grey, so a
    painted picture can't keep the blob as a flat grey silhouette."""
    from PIL import Image
    w, h = INIT_SHAPES.get(shape, INIT_SHAPES["object"])
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size - 0.5
    r = (xx / (w / 2)) ** 2 + (yy / (h / 2)) ** 2
    a = np.clip((1.15 - r) / 0.3, 0, 1)                    # soft edge
    a = cv2.GaussianBlur(a, (0, 0), size * 0.02)[..., None]
    fill = np.full((size, size, 3), float(grey), np.float32)
    if colour is not None:                 # the request's colour / material (init_colour)
        fill[:] = np.asarray(colour, np.float32)
    if noise > 0:
        rng = np.random.default_rng(seed)
        g = max(4, size // 32)                             # blotches ~3% of the picture
        n = rng.normal(0, 1, (g, g, 4)).astype(np.float32)
        n = cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
        n /= max(1e-6, float(n.std()))
        n = n[..., :1] + 0.35 * n[..., 1:]                 # mostly light/dark, a little colour
        fill += noise * 60 * n
    img = 255 * (1 - a) + fill * a
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


_SHAPE_WORDS = (   # first match wins (the words of the request)
    ("weapon", ("sword", "blade", "dagger", "spear", "staff", "wand", "axe", "mace", "hammer",
                "bow", "lance", "halberd", "scythe", "katana", "rapier", "club", "trident",
                "key", "arrow", "pickaxe", "shovel", "rod")),
    ("bottle", ("potion", "bottle", "flask", "vial", "elixir", "candle", "torch")),
    ("building", ("house", "cottage", "hut", "shop", "inn", "tavern", "barn", "church", "chapel",
                  "temple", "castle", "mill", "forge", "smithy", "building", "cabin", "manor",
                  "fort", "shed", "stable", "tent")),
    ("tall", ("tower", "tree", "oak", "pine", "willow", "birch", "palm", "lamp", "lantern",
              "statue", "totem", "pillar", "column", "obelisk", "signpost", "banner", "door")),
    ("flat", ("cart", "wagon", "bench", "table", "bed", "boat", "fence", "log", "sofa", "rug")),
)


def shape_for(text: str, kind: str = "object") -> str:
    """The start blob for a request: person / animal for creatures, else by the words."""
    if kind in ("person", "animal"):
        return kind
    t = (text or "").lower()
    for shape, words in _SHAPE_WORDS:
        if any(re.search(rf"\b{w}s?\b", t) for w in words):
            return shape
    return "object"


# a grey start blob greyed the pictures (2026-10-15: a red apple came out white-green, a wooden
# shield steel): the blob takes the request's colour / material instead. First match wins.
COLOUR_WORDS = (
    (("red", "crimson", "scarlet", "ruby", "apple", "blood", "fire"), (185, 55, 45)),
    (("orange", "pumpkin", "copper"), (190, 110, 50)),
    (("gold", "golden", "yellow", "brass", "honey", "banana"), (205, 165, 60)),
    (("green", "emerald", "leaf", "leafy", "moss", "mossy", "goblin", "slime", "oak", "tree",
      "pine", "bush"), (80, 140, 65)),
    (("blue", "sapphire", "ice", "frost", "water", "mana"), (70, 110, 185)),
    (("purple", "violet", "amethyst", "arcane"), (120, 75, 160)),
    (("pink", "rose"), (215, 120, 150)),
    (("white", "snow", "bone", "ivory", "marble"), (225, 222, 215)),
    (("black", "obsidian", "shadow", "dark"), (60, 55, 65)),
    (("brown", "wooden", "wood", "oak", "leather", "barrel", "crate", "cart", "bread", "bear",
      "thatched", "chest"), (140, 95, 55)),
    (("stone", "rock", "grey", "gray", "iron", "steel", "silver", "metal", "anvil", "wolf"),
     (150, 150, 155)),
)


def init_colour(words: str):
    """RGB for the start blob from the request's words, or None (neutral grey)."""
    t = (words or "").lower()
    for keys, rgb in COLOUR_WORDS:
        if any(re.search(rf"\b{k}s?\b", t) for k in keys):
            return rgb
    return None
