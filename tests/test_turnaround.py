import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import settings_for
from pixel_pipeline.cli import main
from pixel_pipeline.turnaround import (direction_names, render_model, turnaround_sprites, union_crop,
                                       views_from_capture, write_turnaround)

trimesh = pytest.importorskip("trimesh")

YELLOW, BLUE = (240, 200, 60), (60, 90, 200)


@pytest.fixture
def toy(tmp_path):
    """A blue box body with a yellow head at its front (+z, the glTF front)."""
    def part(mesh, rgb):
        mesh.visual.vertex_colors = np.tile(np.array([*rgb, 255], np.uint8), (len(mesh.vertices), 1))
        return mesh
    body = part(trimesh.creation.box(extents=[0.6, 0.6, 1.4]), BLUE)
    body.apply_translation([0, 0.3, 0])
    head = part(trimesh.creation.icosphere(subdivisions=2, radius=0.35), YELLOW)
    head.apply_translation([0, 0.7, 0.8])
    path = tmp_path / "toy.glb"
    trimesh.Scene([body, head]).export(path)
    return path


def centroid(rgba, rgb):
    ys, xs = np.nonzero((np.abs(rgba[..., :3].astype(int) - rgb).sum(-1) < 90) & (rgba[..., 3] > 0))
    return xs.mean(), ys.mean()


def test_direction_names_run_clockwise_from_east():
    assert direction_names(4) == ["E", "S", "W", "N"]
    assert direction_names(8) == ["E", "SE", "S", "SW", "W", "NW", "N", "NE"]
    assert direction_names(16)[:5] == ["E", "ESE", "SE", "SSE", "S"]
    with pytest.raises(ValueError):
        direction_names(6)


def test_model_faces_each_direction(toy):
    views = render_model(toy, 4, elevation=30, size=96, points=150_000, light=0)
    assert len(views) == 4 and all(v.shape == (96, 96, 4) for v in views)
    e, s, w, n = views
    assert centroid(e, YELLOW)[0] > centroid(e, BLUE)[0] + 5    # E: head on the right
    assert centroid(w, YELLOW)[0] < centroid(w, BLUE)[0] - 5    # W: head on the left
    assert centroid(s, YELLOW)[1] > centroid(n, YELLOW)[1] + 5  # S: head nearer the viewer (lower)
    # one ground point and one scale: the two side views line up exactly
    def extent(v):
        ys, xs = np.nonzero(v[..., 3])
        return ys.min(), ys.max(), xs.max() - xs.min()
    assert np.allclose(extent(e), extent(w), atol=1)


def test_sprites_share_palette_and_size(toy):
    views = union_crop(render_model(toy, 8, size=96, points=150_000))
    sprites = turnaround_sprites(views, settings_for("object", width=32, height=32, colors=8))
    assert len(sprites) == 8
    assert {s.size for s in sprites} == {(32, 32)}
    assert all(np.array_equal(s.palette, sprites[0].palette) for s in sprites)


def capture(n, size=(30, 20)):
    """n cut-out 'photos', each a different grey level, with a dot marking its left side."""
    frames = []
    for k in range(n):
        im = np.zeros((size[0], size[1], 4), np.uint8)
        im[5:25, 4:16] = (10 * k, 10 * k, 10 * k, 255)
        im[10, 4] = (255, 0, 0, 255)
        frames.append(im)
    return frames


def grey(view):
    return int(np.median(view[view[..., 3] > 0][:, 1]))


def test_capture_order_reverse_and_mirror():
    views = views_from_capture(capture(8), 8)
    names = direction_names(8)
    # first photo faces the camera (S), walking right goes S -> SW -> W ...
    assert [grey(views[names.index(d)]) for d in ("S", "SW", "W", "N", "E")] == [0, 10, 20, 40, 60]
    rev = views_from_capture(capture(8), 8, reverse=True)
    assert grey(rev[names.index("SE")]) == 10
    half = views_from_capture(capture(5), 8, mirror=True)      # S, SW, W, NW, N
    assert grey(half[names.index("E")]) == grey(half[names.index("W")]) == 20
    assert np.array_equal(half[names.index("E")], half[names.index("W")][:, ::-1])
    with pytest.raises(ValueError, match="need 8 views"):
        views_from_capture(capture(7), 8)


def test_capture_scales_views_to_one_height():
    small = np.zeros((40, 20, 4), np.uint8)
    small[20:40, 5:15] = 255
    big = np.zeros((80, 40, 4), np.uint8)
    big[40:80, 10:30] = 255
    views = views_from_capture([small, big, small, big], 4)
    heights = {int((v[..., 3] > 127).any(1).sum()) for v in views}
    assert max(heights) - min(heights) <= 1


def test_write_turnaround_files(tmp_path, toy):
    sprites = turnaround_sprites(union_crop(render_model(toy, 16, size=64, points=80_000)),
                                 settings_for("object", width=24, height=24))
    written = write_turnaround(sprites, tmp_path / "toy.png", aseprite=True, godot=True)
    assert [p.suffix for p in written] == [".png", ".json", ".aseprite", ".tres", ".gd"]
    meta = json.loads((tmp_path / "toy.json").read_text())
    assert [t["name"] for t in meta["meta"]["frameTags"]] == direction_names(16)
    assert Image.open(tmp_path / "toy.png").size == (8 * 24, 2 * 24)
    tres = (tmp_path / "toy.tres").read_text()
    assert tres.count('"name": &"') == 16 and '&"SSW"' in tres
    assert "wrapi(roundi(direction.angle() / TAU * steps), 0, steps)" in (tmp_path / "toy_facing.gd").read_text()


def test_cli_model_and_photos(tmp_path, toy, capsys):
    out = tmp_path / "out" / "toy.png"
    assert main(["turnaround", str(toy), "-o", str(out), "--directions", "8", "-W", "24", "-H", "24",
                 "--godot", "--preview", "3"]) == 0
    assert "8 directions 24x24" in capsys.readouterr().out
    assert (tmp_path / "out" / "toy_spin.gif").exists() and (tmp_path / "out" / "toy.tres").exists()
    photos = tmp_path / "photos"
    photos.mkdir()
    for k, im in enumerate(capture(5)):   # names sort naturally: 2 before 10
        Image.fromarray(im).save(photos / f"view{[1, 2, 3, 10, 11][k]}.png")
    assert main(["turnaround", str(photos), "-o", str(tmp_path / "half.png"), "--directions", "8",
                 "--mirror", "-W", "16", "-H", "24"]) == 0
    assert "5 photos -> 8 directions" in capsys.readouterr().out


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to a Godot 4 binary to run")
def test_turnaround_loads_and_faces_in_godot(tmp_path, toy):
    sprites = turnaround_sprites(union_crop(render_model(toy, 16, size=64, points=80_000)),
                                 settings_for("object", width=24, height=24))
    write_turnaround(sprites, tmp_path / "toy.png", godot=True)
    (tmp_path / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (tmp_path / "check.gd").write_text(
        "extends SceneTree\nfunc _init():\n"
        "\tvar s = AnimatedSprite2D.new()\n"
        "\ts.set_script(load(\"res://toy_facing.gd\"))\n"
        "\ts.sprite_frames = load(\"res://toy.tres\")\n"
        "\troot.add_child(s)\n"
        "\tvar got = []\n"
        "\tfor v in [Vector2(1, 0), Vector2(0, 1), Vector2(-1, -1), Vector2(1, -0.42)]:\n"
        "\t\ts.face(v)\n\t\tgot.append(String(s.animation))\n"
        "\tprint(\"FACING \", \",\".join(got), \" \", s.sprite_frames.get_animation_names().size())\n"
        "\tquit()\n")
    subprocess.run([GODOT, "--headless", "--path", str(tmp_path), "--import"],
                   capture_output=True, timeout=120)
    out = subprocess.run([GODOT, "--headless", "--path", str(tmp_path), "-s", "check.gd"],
                         capture_output=True, text=True, timeout=120).stdout
    assert "FACING E,S,NW,ENE 16" in out


def test_uv_textured_model_keeps_texture_colours(tmp_path):
    # image-to-3D tools export UV-textured glTF: a plane facing +z, red left half, green right
    tex = np.zeros((32, 32, 3), np.uint8)
    tex[:, :16], tex[:, 16:] = (220, 30, 30), (30, 200, 60)
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=Image.fromarray(tex),
                                                   baseColorFactor=[255, 255, 255, 255])
    plane = trimesh.Trimesh([[-1, 0, 0], [1, 0, 0], [1, 2, 0], [-1, 2, 0]], [[0, 1, 2], [0, 2, 3]],
                            visual=trimesh.visual.TextureVisuals(uv=[[0, 0], [1, 0], [1, 1], [0, 1]],
                                                                 material=material), process=False)
    plane.export(tmp_path / "plane.glb")
    s = render_model(tmp_path / "plane.glb", 4, size=64, points=60_000, light=0)[1]  # S faces us
    xs = np.nonzero(s[..., 3])[1]
    mid = (xs.min() + xs.max()) // 2
    opaque = s[..., 3] > 0
    cols = np.arange(64)[None, :].repeat(64, 0)
    assert np.allclose(s[opaque & (cols < mid - 2)][:, :3].mean(0), (220, 30, 30), atol=6)
    assert np.allclose(s[opaque & (cols > mid + 2)][:, :3].mean(0), (30, 200, 60), atol=6)
