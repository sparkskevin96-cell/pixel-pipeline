"""Speed work keeps results identical: the fast paths against the plain versions they replaced,
worker processes against one process, cache hits against computing again (2026-10-14)."""
import math

import numpy as np
import pytest

from pixel_pipeline import cache, parallel
from pixel_pipeline import drawnrig as dr
from pixel_pipeline import finishing, pixelclean, pixelize, settings_for
from pixel_pipeline.anim3d import animated_turnaround, load_rig
from pixel_pipeline.finishing import _N4 as F_N4
from pixel_pipeline.finishing import _neighbors
from pixel_pipeline.pipeline import Sprite
from pixel_pipeline.turnaround import _splat, _splat_plain, pose_columns
from test_anim3d import write_robot


def random_sprite(rng, h=None, w=None, colours=None):
    h = h or int(rng.integers(8, 60))
    w = w or int(rng.integers(8, 60))
    pal = rng.integers(0, 256, (colours or int(rng.integers(3, 30)), 3)).astype(np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    inside = ((yy - h / 2) / (h * 0.45)) ** 2 + ((xx - w / 2) / (w * 0.4)) ** 2 < 1
    inside |= rng.random((h, w)) < 0.04
    idx = rng.integers(0, len(pal), (h, w))
    idx = np.where(rng.random((h, w)) < 0.6, np.roll(idx, 1, 0), idx)  # some patches
    idx = np.where(inside, idx, -1).astype(np.int32)
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[inside, :3] = pal[idx[inside]]
    rgba[inside, 3] = 255
    return idx, pal, rgba


# --------------------------------------------------------------------------- the plain versions

def flatten_spurs_plain(idx, body):
    h, w = idx.shape
    for y, x in zip(*np.nonzero(body)):
        nbs = [(y + dy, x + dx, dy, dx) for dy, dx in pixelclean._N4
               if 0 <= y + dy < h and 0 <= x + dx < w and body[y + dy, x + dx]]
        if len(nbs) != 1:
            continue
        by, bx, dy, dx = nbs[0]
        across = [(by + dx, bx + dy), (by - dx, bx - dy)]
        if not all(0 <= a < h and 0 <= b < w and body[a, b] for a, b in across):
            continue
        ring = [idx[y + ey, x + ex] for ey, ex in pixelclean._N4
                if 0 <= y + ey < h and 0 <= x + ex < w and idx[y + ey, x + ex] >= 0
                and not body[y + ey, x + ex]]
        if ring:
            idx[y, x] = ring[0]
            body[y, x] = False


def pixel_perfect_plain(idx, body):
    h, w = idx.shape
    mask = idx >= 0
    removed = np.zeros_like(mask)

    def at(y, x, grid):
        return 0 <= y < h and 0 <= x < w and grid[y, x]
    changed = True
    while changed:
        changed = False
        for y, x in zip(*np.nonzero(mask & ~body)):
            if any(at(y + dy, x + dx, body) for dy, dx in pixelclean._N4):
                continue
            horiz = [at(y, x + d, mask) for d in (-1, 1)]
            vert = [at(y + d, x, mask) for d in (-1, 1)]
            if not ((sum(horiz) == 1 and sum(vert) == 1) or sum(horiz) + sum(vert) == 1):
                continue
            mask[y, x] = False
            removed[y, x] = True
            changed = True
    idx[removed] = -1
    return removed


def selout_plain(idx, ring, palette, darken=0.55):
    lab = finishing.rgb_to_lab(palette)
    nb = _neighbors(idx, F_N4)
    out = np.full(idx.shape, -1, np.int16)
    darkest = int(np.argmin(lab[:, 0]))
    for y, x in zip(*np.nonzero(ring)):
        vals = nb[y, x][nb[y, x] >= 0]
        if not len(vals):
            out[y, x] = darkest
            continue
        src = lab[np.bincount(vals).argmax()]
        target = np.array([src[0] * darken, src[1] * 0.85, src[2] * 0.85])
        darker = np.nonzero(lab[:, 0] < src[0] - 4)[0]
        if not len(darker):
            out[y, x] = darkest
            continue
        out[y, x] = darker[((lab[darker] - target) ** 2).sum(1).argmin()]
    return out


def test_pixel_cleanup_fast_paths_match_the_plain_ones():
    rng = np.random.default_rng(3)
    for _ in range(150):
        idx, pal, _ = random_sprite(rng)
        lab = pixelclean.rgb_to_lab(pal.astype(np.float32))
        mask = idx >= 0
        body = mask & ~pixelclean.boundary(mask)
        for size in (1, 2):
            a, b = idx.copy(), idx.copy()
            pixelclean._merge_islands(a, body, lab, size, 30.0)
            pixelclean._merge_islands_labelled(b, body, lab, size, 30.0)
            assert np.array_equal(a, b)
        a, b = idx.copy(), idx.copy()
        ba, bb = body.copy(), body.copy()
        pixelclean._flatten_spurs(a, ba)
        flatten_spurs_plain(b, bb)
        assert np.array_equal(a, b) and np.array_equal(ba, bb)
        ra, rb = pixelclean._pixel_perfect(a, ba), pixel_perfect_plain(b, bb)
        assert np.array_equal(ra, rb) and np.array_equal(a, b)
        ring = finishing.outline_ring(mask)
        assert np.array_equal(finishing.selout_colors(idx, ring, pal), selout_plain(idx, ring, pal))


def test_downscale_median_matches_numpy_median():
    rng = np.random.default_rng(1)
    for _ in range(60):
        p = int(rng.integers(1, 9))
        h, w = (int(x) for x in rng.integers(1, 30, 2))
        img = rng.integers(0, 256, (h * p, w * p, 3)).astype(np.uint8)
        lab = pixelize.cv2.cvtColor(img, pixelize.cv2.COLOR_RGB2Lab).astype(np.float32)
        blocks = pixelize._blocks(lab, p)
        ref = np.median(blocks, axis=2)
        srt = np.sort(pixelize._blocks(pixelize.cv2.cvtColor(img, pixelize.cv2.COLOR_RGB2Lab), p),
                      axis=2)
        n = p * p
        mine = (srt[:, :, n // 2].astype(np.float32) if n % 2 else
                (srt[:, :, n // 2 - 1].astype(np.float32) + srt[:, :, n // 2]) / np.float32(2))
        assert np.array_equal(ref, mine)
        assert pixelize.downscale(img, p).shape == (h, w, 3)


def test_splat_matches_the_whole_canvas_version():
    rng = np.random.default_rng(0)
    pts = (rng.normal(size=(40_000, 3)) * [0.3, 0.5, 0.2] + [0, 0.5, 0]).astype(np.float32)
    nrm = rng.normal(size=pts.shape).astype(np.float32)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    base = rng.integers(0, 256, (len(pts), 3)).astype(np.float32)
    cols = pose_columns(pts)
    e = math.radians(30)
    ld = np.array([-0.45, 0.6, 0.66])
    ld /= np.linalg.norm(ld)
    for size, scale in ((96, 70.0), (64, 120.0)):           # 120: points off the canvas
        for k in range(8):
            t = math.radians(90 - k * 45)
            args = (math.cos(t), math.sin(t), math.cos(e), math.sin(e), scale, size / 2,
                    size * 0.9, size, 0.2, ld)
            for normals in (True, False):
                a = _splat_plain(pts, nrm, base, *args, normals)
                b = _splat(pts, nrm, base, *args, normals, cols)
                a, b = (a, b) if normals else ((a,), (b,))
                assert all(np.array_equal(x, y) for x, y in zip(a, b))


# --------------------------------------------------------------------------- workers + caches

def test_worker_count_follows_the_setting(monkeypatch):
    monkeypatch.setenv("PIXEL_WORKERS", "1")
    assert parallel.worker_count(16) == 1
    monkeypatch.setenv("PIXEL_WORKERS", "3")
    assert 1 <= parallel.worker_count(16, per_worker_mb=1) <= 3
    assert parallel.worker_count(1, per_worker_mb=1) == 1
    assert parallel.free_ram_mb() > 0


def _square(x):
    return x * x


def test_pool_runs_the_same_work_here_or_on_workers():
    assert parallel.run(_square, range(5), workers=1) == [0, 1, 4, 9, 16]
    assert parallel.run(_square, range(5), workers=2) == [0, 1, 4, 9, 16]


def test_animated_turnaround_workers_give_the_same_sprites(tmp_path):
    path = write_robot(tmp_path / "robot.glb")
    s = settings_for("person", width=24, height=32, colors=12)
    kw = dict(elevation=30.0, normals=True, clean=True, points=60_000, size=96)
    one, _ = animated_turnaround([path], s, 4, 3, workers=1, **kw)
    two, _ = animated_turnaround([path], s, 4, 3, workers=2, **kw)
    from pixel_pipeline import anim3d
    assert anim3d.STAGE_SECONDS["workers"] == 2                  # really on two processes
    framed, _ = animated_turnaround([path], s, 4, 3, workers=2, frame_with=["wave"], **kw)
    framed1, _ = animated_turnaround([path], s, 4, 3, workers=1, frame_with=["wave"], **kw)
    for a, b in ((one, two), (framed, framed1)):
        for x, y in zip(a.flat(), b.flat()):
            assert np.array_equal(x.rgba, y.rgba) and np.array_equal(x.normal, y.normal)
    again, _ = animated_turnaround([path], s, 4, 3, workers=1, **kw)   # converter from cache
    assert all(np.array_equal(x.rgba, y.rgba) and np.array_equal(x.normal, y.normal)
               for x, y in zip(one.flat(), again.flat()))
    assert any((cache.root() / "prep3d").rglob("*.npz"))


def test_bending_on_workers_and_from_the_cache_is_the_same(tmp_path):
    rig = load_rig([write_robot(tmp_path / "robot.glb")])
    body = dr.Body(rig, 40_000)
    rest = dr.rest_pose(body)
    poses = dr.clip_poses(body, rig.clips, 4)
    W, H = 48, 40
    cams = dr.framing(body, [p for row in poses for p in row] + [rest], ["E", "S", "W", "N"],
                      30.0, W, H)
    pal = np.array([[200, 30, 30], [30, 200, 30], [30, 30, 200], [230, 230, 60]], np.uint8)
    drawns, colours = [], []
    for cam in cams:
        view = dr.rest_view(body, cam, rest)
        ys, xs = np.mgrid[0:H, 0:W]
        key = np.where(view.mask, (xs // 3 + ys // 3) % 3, -1).astype(np.int32)
        d = dr.glue(body, view, cam, rest, key, pal)
        col = np.full(len(body.bind), 3, np.int32)
        dr.add_hidden(d, body, view, rest, col)
        drawns.append(d)
        colours.append(col)
    state = dr.BendState(dr.BodyArrays(body.weights, body.bind, body.bind_n), poses, rest, cams,
                         drawns, colours, pal, dr._torso_node(rig), 45.0, 0.55, True, True,
                         "selout")
    plain = [dr.bend_direction(state, k) for k in range(len(cams))]
    on_workers = dr.bend_directions(state, workers=2)       # also fills the cache
    cached = dr.bend_directions(state, workers=1)
    for res in (on_workers, cached):
        for (rows, fb, rf), (prow, pfb, prf) in zip(res, plain):
            assert fb == sum(pfb) and rf == sum(prf)
            for r, pr in zip(rows, prow):
                for x, y in zip(r, pr):
                    assert np.array_equal(x.rgba, y.rgba) and np.array_equal(x.indices, y.indices)
                    assert np.array_equal(x.normal, y.normal)


def test_cache_store_digest_and_trim(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXEL_CACHE", str(tmp_path / "c"))
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    k = cache.digest(a, {"b": 1, "a": [1.5, "x"]}, None)
    assert k == cache.digest(a.copy(), {"a": [1.5, "x"], "b": 1}, None)
    assert k != cache.digest(a.astype(np.float64), {"b": 1, "a": [1.5, "x"]}, None)
    assert cache.load("t", k) is None
    cache.save("t", k, x=a)
    assert np.array_equal(cache.load("t", k)["x"], a)
    for i in range(5):
        cache.save("t", cache.digest(i), x=np.random.default_rng(i).random(30_000))
    assert cache.trim(0.3) > 0 and cache.size_mb() <= 0.3
    monkeypatch.setenv("PIXEL_CACHE", "off")
    assert not cache.enabled() and cache.load("t", k) is None


def test_springs_rebake_reuses_clips(tmp_path):
    import hashlib

    from pixel_pipeline.autorig import auto_rig
    from pixel_pipeline.springs import add_springs, rebake
    from test_springs import caped
    rig = auto_rig(caped(tmp_path / "caped.glb"), tmp_path / "rig.glb",
                   actions=("idle", "walk", "death")).path
    add_springs(rig, tmp_path / "a.glb")
    rebake(tmp_path / "a.glb", tmp_path / "b.glb")
    n = len(list((cache.root() / "springs").rglob("*.npz")))
    rebake(tmp_path / "a.glb", tmp_path / "c.glb")                 # every clip from the cache
    assert len(list((cache.root() / "springs").rglob("*.npz"))) == n
    h = [hashlib.sha1((tmp_path / f).read_bytes()).hexdigest() for f in ("b.glb", "c.glb")]
    assert h[0] == h[1]


def test_rig_cache_gives_the_same_file(tmp_path):
    from pixel_pipeline.library import auto_rig_cached
    from test_springs import caped
    model = caped(tmp_path / "m.glb", cape=False)
    a = auto_rig_cached(model, tmp_path / "a.glb").read_bytes()
    assert any((cache.root() / "rig").rglob("*.npz"))
    assert auto_rig_cached(model, tmp_path / "b.glb").read_bytes() == a


def test_decimate_fast_and_its_fallback(monkeypatch):
    import sys

    from pixel_pipeline import gamemesh as gm
    from skimage.measure import marching_cubes
    z, y, x = np.mgrid[-1:1:40j, -1:1:40j, -1:1:40j]
    v, f, _, _ = marching_cubes(x ** 2 + y ** 2 / 0.6 + z ** 2 - 0.5, 0.0)
    m = gm.Mesh(v.astype(np.float64), f.astype(np.int64))
    if gm.fast_available():
        low = gm.decimate_fast(m, 500)
        assert 300 <= low.tris <= 500 and low.v.shape[1] == 3
    monkeypatch.setitem(sys.modules, "fast_simplification", None)    # not installed
    assert not gm.fast_available()
    low = gm.decimate_fast(m, 500, quiet=True)
    assert low.tris <= 500
