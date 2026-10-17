"""Pixel quality (2026-10-17): one pixel grid (assets.pixel_grid), the automatic per-direction
quality loop (dirquality + library.drawn_with_quality), the art-directed clean-up, normal maps
smoothed per material, character timing (metrics), the action map's walkability (diorama)."""
import json

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import dirquality as dq
from pixel_pipeline import drawnrig as dr
from pixel_pipeline.assets import Style, pixel_grid, style_for_type
from pixel_pipeline.pipeline import Sprite
from test_anim3d import write_robot
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

DIRS = dq.DIRS16


def _figure(noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A 40x24 outlined figure in big flat shades (+ salt noise inside when noise > 0)."""
    h, w = 40, 24
    a = np.zeros((h, w, 4), np.uint8)
    a[4:36, 5:19] = (30, 30, 40, 255)              # outline
    a[5:35, 6:18] = (150, 40, 40, 255)             # red cloth
    a[5:20, 6:18] = (120, 120, 140, 255)           # steel
    a[5:10, 9:15] = (220, 180, 150, 255)           # face
    if noise:
        rng = np.random.default_rng(seed)
        m = (rng.random((h, w)) < noise)
        m[:6] = m[34:] = False
        m[:, :7] = m[:, 17:] = False
        a[m, :3] = (90, 160, 60)                   # stray green specks
    return a


def write_sheet(path, frames: dict) -> None:
    """frames {name: rgba}: a one-row-per-frame sheet + json like write_animated's."""
    names = list(frames)
    h, w = frames[names[0]].shape[:2]
    sheet = np.zeros((h, w * len(names), 4), np.uint8)
    meta = {"frames": [], "meta": {}}
    for i, n in enumerate(names):
        sheet[:, i * w:(i + 1) * w] = frames[n]
        meta["frames"].append({"filename": n, "frame": {"x": i * w, "y": 0, "w": w, "h": h}, "duration": 100})
    Image.fromarray(sheet).save(path)
    path.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")


def test_dirquality_flags_the_speckled_direction_and_never_the_front(tmp_path):
    frames = {}
    for d in DIRS:
        f = _figure(0.18 if d in ("NE", "S") else 0.0, seed=hash(d) % 100)
        for i in range(2):
            frames[f"idle_{d} {i}"] = f
        frames[f"walk_{d} 0"] = f
    write_sheet(tmp_path / "sheet.png", frames)
    rep = dq.check_sheet(tmp_path / "sheet.png")
    assert rep["reference"] == "S"
    assert "NE" in rep["failed"] and "specks" in rep["directions"]["NE"]["why"]
    assert rep["passed"] >= 14
    assert dq.to_redraw(rep) == ["NE"]                     # the approved front is never redrawn
    assert dq.to_redraw(rep, skip=[("NE", tuple(rep["directions"]["NE"]["why"]))]) == []
    assert "directions passed" in dq.summary_text(dq.compact(rep))


def test_pixel_grid_one_grid_whole_scales():
    for t in ("hd2d", "topdown-rpg", "side-scroller", "isometric", "cards-ui", "custom"):
        for size in ("normal", "detailed"):
            st = style_for_type(t, size)
            g = pixel_grid(st)
            assert g["ppu"] == g["tile"] == st.tile           # sprite texels = tile texels
            assert isinstance(g["scale"], int) and g["scale"] >= 1
            assert g["base"][0] * g["scale"] == 1920 and g["base"][1] * g["scale"] == 1080
    hd = style_for_type("hd2d", "detailed")
    assert (hd.person_height, hd.tile) == (128, 64) and pixel_grid(hd)["scale"] == 2
    assert pixel_grid(style_for_type("hd2d"))["render"] == "hires"
    assert pixel_grid(style_for_type("topdown-rpg"))["render"] == "pixel"
    assert pixel_grid(Style(render="pixel", tile=32))["base"] == [480, 270]


def test_art_clean_merges_near_colours_drops_crumbs_keeps_details():
    from pixel_pipeline.pixelclean import art_clean
    pal = np.array([[150, 40, 40], [152, 42, 41], [30, 30, 40], [250, 250, 80], [120, 120, 140]], np.uint8)
    idx = np.full((30, 30), -1, np.int32)
    idx[4:26, 4:26] = 2                                     # outline ring
    idx[5:25, 5:25] = 0                                     # a red area...
    idx[5:25:3, 5:25:4] = 1                                 # ...sprinkled with a near-identical red
    idx[12, 12] = 3                                         # an eye: a far colour, kept
    idx[1, 28] = 4                                          # a crumb off the body
    rgba = np.zeros((30, 30, 4), np.uint8)
    rgba[idx >= 0, :3] = pal[idx[idx >= 0]]
    rgba[idx >= 0, 3] = 255
    out = art_clean([Sprite(rgba, idx, pal, None)], "selout")[0]
    assert not (out.indices == 1).any()                     # merged into one red
    assert out.indices[12, 12] == 3                         # the eye stays
    assert out.indices[1, 28] == -1                         # the crumb is gone


def test_normals_smoothed_within_a_material_not_across():
    from pixel_pipeline.normals import decode, encode, smooth_by_material
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[:, :10] = (200, 30, 30, 255)
    rgba[:, 10:] = (40, 60, 200, 255)
    rng = np.random.default_rng(1)
    n = np.zeros((20, 20, 3), np.float32)
    n[:, :10] = (-0.6, 0, 0.8)
    n[:, 10:] = (0.6, 0, 0.8)
    n += rng.normal(0, 0.25, n.shape)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    nm = np.concatenate([encode(n), np.full((20, 20, 1), 255, np.uint8)], -1)
    sm = decode(smooth_by_material(nm, rgba)[..., :3])
    before = decode(nm[..., :3])
    assert sm[:, 2:8].std(axis=(0, 1)).sum() < 0.6 * before[:, 2:8].std(axis=(0, 1)).sum()
    assert sm[:, 9, 0] .mean() < -0.3 and sm[:, 10, 0].mean() > 0.3   # the fold between them stays


def test_action_map_has_no_spot_the_feet_cannot_reach():
    from pixel_pipeline import diorama, playtest
    d = playtest.action_map()
    assert diorama.narrow_gaps(d) == []
    assert diorama.unreachable_cells(d, (7, 12)) == []
    # the old collision (a 0.24 circle, canopy-sized props) sealed spots off: the check sees it
    old = dict(diorama.BLOCKING_RADII)
    try:
        diorama.BLOCKING_RADII.update(bush=0.3, tree=0.3, rock=0.36, well=0.45, pot=0.22)
        assert diorama.unreachable_cells(d, (7, 12), feet=(0.24, 0.24))
    finally:
        diorama.BLOCKING_RADII.clear()
        diorama.BLOCKING_RADII.update(old)


def test_metrics_stages_and_aggregate(tmp_path):
    from pixel_pipeline import metrics
    f = tmp_path / "characters" / "bot"
    f.mkdir(parents=True)
    metrics.merge(f, {"model": 120.0, "rig": 10.0})
    with metrics.stage(f, "drawings"):
        pass
    metrics.add(f, "drawings", 300.0)
    s = metrics.summary(f)
    assert s["stages_min"]["model"] == 2.0 and s["gpu_min"] == 7.0
    assert s["wall_min"] == pytest.approx(7.2, abs=0.05)
    agg = metrics.aggregate(5, tmp_path)
    assert agg["count"] == 1 and agg["stages_min"]["drawings"]["median"] == 5.0
    assert "made in" in metrics.made_in(f)


def _robot_library(comfy, tmp_path, monkeypatch):  # noqa: F811
    from pixel_pipeline import autorig, generate, library
    fake, client = comfy
    monkeypatch.setattr(generate, "ComfyClient", lambda *a, **k: client)
    monkeypatch.setattr(dr, "_gpu_temperature", lambda: None)
    fake.history_delay = 0
    monkeypatch.setattr(library, "style", lambda: Style(directions=4))
    monkeypatch.setattr(library, "add_project_clips", lambda *a, **k: [])
    monkeypatch.setattr(autorig, "auto_rig", lambda model, out: write_robot(out))
    monkeypatch.setenv("PIXEL_DRAWN", "1")
    monkeypatch.setenv("PIXEL_QUALITY", "1")
    return fake, library, write_robot(tmp_path / "robot_model.glb")


def _bad_first(monkeypatch, d: str, times: int):
    """dirquality.measure with direction d made speckled for the first `times` calls."""
    real = dq.measure
    calls = []

    def fake(sheet_png, keys_report=None, kind="person"):
        rep = real(sheet_png, keys_report, kind)
        calls.append(1)
        if len(calls) <= times and d in rep["directions"]:
            rep["directions"][d]["specks"] = 99.0
        return rep
    monkeypatch.setattr(dq, "measure", fake)
    return calls


def test_a_new_character_is_checked_and_its_bad_direction_redrawn_automatically(comfy, tmp_path, monkeypatch):  # noqa: F811
    fake, library, model = _robot_library(comfy, tmp_path, monkeypatch)
    _bad_first(monkeypatch, "E", 1)                        # E looks bad once, fine when redrawn
    ch = library.add_character("Bot", model, prompt="a robot", timing={"model": 60.0})
    meta = json.loads((ch.folder / "keys" / "keys.json").read_text())
    q = meta["quality"]
    assert "E" in q["redrawn"] and "E" not in q["reverted"]
    assert q["total"] == 4 and "directions passed" in q["text"]
    assert int(meta["seeds"]["E"]) != int(meta["seeds"]["S"])          # E got a new seed
    extra = json.loads((ch.folder / "asset.json").read_text())["extra"]
    assert "E" in extra["quality"]["redrawn"]
    st = extra["timing"]["stages"]
    for k in ("model", "rig", "drawings", "quality_check", "quality_redraw", "bending"):
        assert k in st, k
    assert library.is_drawn(ch) and ch.sheet.exists()
    assert not (ch.folder / "keys" / "_check").exists()                 # the check sheet is gone


def test_a_redraw_that_comes_out_worse_is_put_back(comfy, tmp_path, monkeypatch):  # noqa: F811
    fake, library, model = _robot_library(comfy, tmp_path, monkeypatch)
    _bad_first(monkeypatch, "W", 2)                        # bad before AND after the redraw
    ch = library.add_character("Bot", model, prompt="a robot")
    keys = ch.folder / "keys"
    q = json.loads((keys / "keys.json").read_text())["quality"]
    assert "W" in q["reverted"] and "W" not in q["redrawn"]
    assert "specks" in q["tried"]["W"]
    assert list((keys / "rejected").glob("drawn_W_*.png"))             # the worse one set aside
    seed_w = json.loads((keys / "keys.json").read_text())["seeds"]["W"]
    rejected = len(list((keys / "rejected").glob("drawn_W_*.png")))
    # checking again doesn't redraw W for the same reason
    _bad_first(monkeypatch, "W", 1)
    library.check_and_fix(ch)
    assert json.loads((keys / "keys.json").read_text())["seeds"]["W"] == seed_w
    assert len(list((keys / "rejected").glob("drawn_W_*.png"))) == rejected


def test_check_and_fix_without_the_gpu_only_rebends_and_checks(comfy, tmp_path, monkeypatch):  # noqa: F811
    fake, library, model = _robot_library(comfy, tmp_path, monkeypatch)
    monkeypatch.setenv("PIXEL_QUALITY", "0")
    ch = library.add_character("Bot", model, prompt="a robot")
    assert library.quality_of(ch) == {}
    n = len(fake.prompts)
    monkeypatch.setenv("PIXEL_QUALITY", "1")
    q = library.check_and_fix(ch, redraw_bad=False)
    assert len(fake.prompts) == n and q["total"] == 4
    est = library.fix_estimate(ch)
    assert not est["new_camera"] and est["gpu_min"] > 0
