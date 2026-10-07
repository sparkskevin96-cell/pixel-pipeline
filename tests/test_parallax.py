"""Parallax layers. Fast: depth maps are synthetic, the depth model is never downloaded."""
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import depth as depth_mod
from pixel_pipeline.cutout import CutoutUnavailable
from pixel_pipeline.parallax import (
    assign_hidden,
    build_parallax,
    default_scroll_scales,
    depth_thresholds,
    layer_names,
    make_parallax,
    seamless_x,
    split_layers,
    write_godot_parallax,
)

FAR, MID, NEAR = 0.05, 0.4, 0.9
SKY, HILL, TREE = (90, 150, 230), (60, 140, 60), (120, 70, 30)


def scene(w=320, h=180):
    """Sky (far) over a band of hills (mid) with a tree (near) standing in front of both."""
    yy, xx = np.mgrid[0:h, 0:w]
    rgb = np.empty((h, w, 3), np.uint8)
    d = np.empty((h, w), np.float32)
    rgb[:], d[:] = SKY, FAR
    hills = yy > h * 0.45 + 12 * np.sin(xx / w * 2 * np.pi * 3)
    rgb[hills], d[hills] = HILL, MID
    tree = (abs(xx - w * 0.55) < w * 0.07) & (yy > h * 0.2)
    rgb[tree], d[tree] = TREE, NEAR
    noise = np.random.default_rng(0).integers(-6, 7, rgb.shape)
    return np.clip(rgb.astype(int) + noise, 0, 255).astype(np.uint8), d, tree, hills


def test_split_layers_far_opaque_near_cut_out():
    rgb, d, tree, hills = scene()
    far, mid, near = split_layers(rgb, d, 3, px=2)
    assert far.shape == mid.shape == near.shape == (180, 320, 4)
    assert (far[..., 3] == 255).all()
    near_a = near[..., 3] >= 128
    assert 0 < near_a.mean() < 0.5
    assert (near_a == tree).mean() > 0.97          # near layer = the tree
    assert (mid[..., 3][~hills & ~tree] < 128).mean() > 0.97   # sky belongs to the far layer


def test_hidden_parts_go_to_the_layer_that_continues_behind():
    rgb, d, tree, _ = scene()
    far, mid, _ = split_layers(rgb, d, 3, px=2)
    x = int(320 * 0.55)
    # behind the trunk at hill height the hills carry on (mid opaque, hill-coloured) ...
    y_hill = 170
    assert mid[y_hill, x, 3] == 255
    assert np.abs(mid[y_hill, x, :3].astype(int) - HILL).max() < 30
    # ... at sky height it's the far layer's job: mid stays transparent, far is sky-coloured
    y_sky = 45
    assert tree[y_sky, x]
    assert mid[y_sky, x, 3] == 0
    assert np.abs(far[y_sky, x, :3].astype(int) - SKY).max() < 30


def test_assign_hidden_continues_rows():
    labels = np.array([[0, 2, 2, 0],
                       [1, 2, 2, 1],
                       [2, 2, 2, 2]], np.uint8)
    out = assign_hidden(labels, labels == 2)
    assert out.tolist() == [[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]]


def test_depth_thresholds_fall_between_modes():
    rng = np.random.default_rng(1)
    d = np.concatenate([rng.normal(m, 0.02, 4000) for m in (0.1, 0.45, 0.85)]).clip(0, 1)
    cuts = depth_thresholds(d, 3)
    assert 0.18 < cuts[0] < 0.38 and 0.52 < cuts[1] < 0.78
    q = depth_thresholds(d, 3, "quantile")
    assert len(q) == 2 and q[0] < q[1]


def test_seamless_x_wraps():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 255, (20, 100, 3)).astype(np.uint8)
    out = seamless_x(img, 16)
    assert out.shape == (20, 84, 3)
    # right edge continues straight into the left edge: neighbours in the source
    assert np.abs(out[:, 0].astype(int) - img[:, 84].astype(int)).max() <= 16
    assert np.array_equal(out[:, -1], img[:, 83])
    assert np.array_equal(out[:, 16:], img[:, 16:84])


def _colors(rgba):
    px = rgba.reshape(-1, 4)
    return {tuple(c) for c in px[px[:, 3] == 255][:, :3]}


def test_make_parallax_writes_aligned_layers_with_one_palette(tmp_path, monkeypatch):
    rgb, d, *_ = scene(400, 240)
    monkeypatch.setattr(depth_mod, "estimate_depth",
                        lambda *a, **k: pytest.fail("model must not run when depth is given"))
    res = make_parallax(rgb, tmp_path, name="hills", width=64, height=36, colors=12, depth=d,
                        save_depth=True)
    paths = res["layers"]
    assert [p.name for p in paths] == ["hills_layer0_far.png", "hills_layer1_mid.png",
                                       "hills_layer2_near.png"]
    layers = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]
    assert all(lay.shape == (36, 64, 4) for lay in layers)
    assert (layers[0][..., 3] == 255).all()
    assert 0 < (layers[2][..., 3] == 255).mean() < 0.5
    assert set(np.unique(layers[2][..., 3])) <= {0, 255}
    union = set().union(*(_colors(lay) for lay in layers))
    assert len(union) <= 12                          # one shared palette across all layers
    prev = np.asarray(Image.open(res["preview"]).convert("RGBA"))
    assert prev.shape == (36, 64, 4) and (prev[..., 3] == 255).all()
    assert _colors(prev) <= union
    assert res["godot"].name == "hills_parallax.tscn" and res["godot"].exists()
    assert res["depth"].exists()
    assert res["scroll_scales"] == [0.2, 0.5, 0.85]


def test_depth_model_is_called_when_no_depth_given(monkeypatch):
    rgb, d, *_ = scene(400, 240)
    calls = []

    def fake(img, *a, **k):
        calls.append(img.shape)
        return np.asarray(Image.fromarray(d).resize((img.shape[1], img.shape[0])))

    monkeypatch.setattr(depth_mod, "estimate_depth", fake)
    res = build_parallax(rgb, 3, 48, 27, colors=8, dither="none")
    assert calls and len(res.sprites) == 3
    assert len(res.palette) <= 8


def _seam(rgba):
    return np.abs(rgba[:, 0, :3].astype(int) - rgba[:, -1, :3].astype(int)).mean()


def test_wrap_removes_the_horizontal_seam(gradient):
    d = np.zeros(gradient.shape[:2], np.float32)    # all far: the far layer is the photo
    kw = dict(layers=2, width=48, height=24, colors=16, dither="none", depth=d)
    plain = build_parallax(gradient, wrap=False, **kw).sprites[0].rgba
    tiled = build_parallax(gradient, wrap=True, **kw).sprites[0].rgba
    assert _seam(plain) > 100
    assert _seam(tiled) < _seam(plain) / 5
    inner = np.abs(np.diff(tiled[..., :3].astype(int), axis=1)).mean()
    assert _seam(tiled) < 4 * inner + 4     # the seam is no worse than any other column step


def test_layer_names_and_default_scales():
    assert layer_names(2) == ["far", "near"]
    assert layer_names(4) == ["far", "mid1", "mid2", "near"]
    s = default_scroll_scales(5)
    assert len(s) == 5 and s == sorted(s) and 0 < s[0] and s[-1] < 1


def _write_pngs(folder: Path, w=64, h=36, n=3):
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, role in enumerate(layer_names(n)):
        p = folder / f"t_layer{i}_{role}.png"
        Image.new("RGBA", (w, h), (i * 60, 100, 200, 255)).save(p)
        paths.append(p)
    return paths


def test_tscn_has_parallax2d_nodes_and_relative_paths(tmp_path):
    pngs = _write_pngs(tmp_path / "bg" / "layers")
    tscn = write_godot_parallax(pngs, tmp_path / "bg" / "t_parallax.tscn", [0.1, 0.4, 0.9])
    text = tscn.read_text()
    assert text.startswith("[gd_scene")
    assert text.count('type="Parallax2D"') == 3 and text.count('type="Sprite2D"') == 3
    order = re.findall(r'\[node name="(\w+)" type="Parallax2D"', text)
    assert order == ["Far", "Mid", "Near"]           # far first = drawn first (behind)
    for s in ("0.1", "0.4", "0.9"):
        assert f"scroll_scale = Vector2({s}, 1)" in text
    assert text.count("repeat_size = Vector2(64, 0)") == 3
    assert 'path="layers/t_layer0_far.png"' in text and "res://" not in text
    assert text.count("texture_filter = 1") == 3 and text.count("centered = false") == 3
    no_repeat = write_godot_parallax(pngs, tmp_path / "x.tscn", repeat=False,
                                     res_dir="res://art/bg/").read_text()
    assert "repeat_size" not in no_repeat
    assert 'path="res://art/bg/t_layer2_near.png"' in no_repeat
    with pytest.raises(ValueError):
        write_godot_parallax(pngs, tmp_path / "y.tscn", [0.5])


def test_missing_onnxruntime_raises_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(depth_mod, "model_file", lambda: pytest.fail("must not download"))
    with pytest.raises(CutoutUnavailable, match="onnxruntime"):
        depth_mod.estimate_depth(np.zeros((32, 32, 3), np.uint8))


def test_depth_pre_and_postprocessing():
    nh, nw = depth_mod.network_size(400, 600, [1, 3, "height", "width"])
    assert nh % 14 == 0 and nw % 14 == 0 and min(nh, nw) >= 518
    assert abs(nw / nh - 1.5) < 0.05
    assert depth_mod.network_size(400, 600, [1, 3, 518, 518]) == (518, 518)
    x = depth_mod.preprocess(np.full((40, 60, 3), 255, np.uint8), 28, 42)
    assert x.shape == (1, 3, 28, 42) and x.dtype == np.float32
    assert np.allclose(x[0, :, 0, 0], (1 - depth_mod.IMAGENET_MEAN) / depth_mod.IMAGENET_STD)
    raw = np.linspace(3, 9, 28 * 42, dtype=np.float32).reshape(1, 28, 42)
    out = depth_mod.postprocess(raw, 40, 60)
    assert out.shape == (40, 60) and out.min() == 0 and out.max() == 1


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot") or \
    "/tmp/claude-0/-home-user-pawgress/61ec288d-b22b-5b6f-bdb8-f7d32f2722bb/scratchpad/godot/godot"

CHECK_GD = """extends SceneTree
func _initialize() -> void:
\tvar packed: PackedScene = load("res://levels/bg/t_parallax.tscn")
\tvar scene := packed.instantiate()
\tfor layer in scene.get_children():
\t\tvar spr: Sprite2D = layer.get_child(0)
\t\tprint("LAYER ", layer.get_class(), " ", layer.scroll_scale, " ", layer.repeat_size, " ",
\t\t\tspr.texture.get_size(), " ", spr.texture.resource_path)
\tscene.free()
\tquit(0)
"""


@pytest.mark.skipif(not Path(GODOT).exists(), reason="Godot 4.3+ binary not found (set GODOT_BIN)")
def test_godot_loads_the_scene(tmp_path):
    proj = tmp_path / "proj"
    pngs = _write_pngs(proj / "levels" / "bg")
    write_godot_parallax(pngs, proj / "levels" / "bg" / "t_parallax.tscn")
    (proj / "project.godot").write_text('config_version=5\n\n[application]\n\nconfig/name="t"\n')
    (proj / "check.gd").write_text(CHECK_GD)
    run = lambda *args: subprocess.run([GODOT, "--headless", "--path", str(proj), *args],
                                       capture_output=True, text=True, timeout=180,
                                       check=False)
    run("--import")
    res = run("-s", "check.gd")
    out = res.stdout + res.stderr
    lines = [ln for ln in out.splitlines() if ln.startswith("LAYER ")]
    assert res.returncode == 0, out
    assert len(lines) == 3 and all("Parallax2D" in ln and "(64.0, 36.0)" in ln for ln in lines), out
    assert "(0.2, 1.0)" in lines[0] and "(0.85, 1.0)" in lines[2]
    assert "res://levels/bg/t_layer0_far.png" in lines[0]
    assert "ERROR" not in out and "WARNING" not in out, out
