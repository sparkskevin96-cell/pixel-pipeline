import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import settings_for
from pixel_pipeline.animate import breathe, bob, idle_frames, sway
from pixel_pipeline.cli import main
from pixel_pipeline.export import load_sprite, pixel_scale
from pixel_pipeline.finishing import outline_ring, remove_orphans, selout_colors
from pixel_pipeline.godot import write_sprite_frames, write_tileset
from pixel_pipeline.video import align_frames, read_frames, video_to_sprites


def figure(h=20, w=12):
    """Opaque 'character' block: rows coloured by height so moves are traceable."""
    rgba = np.zeros((h, w, 4), np.uint8)
    for y in range(4, h):
        rgba[y, 3:9] = [y * 10, 50, 100, 255]
    return rgba


# ---- finishing -------------------------------------------------------------------------------

def test_remove_orphans_keeps_contrasting_highlight():
    pal = np.array([[100, 60, 40], [104, 62, 42], [255, 255, 255]], np.uint8)
    idx = np.zeros((7, 7), np.int16)
    idx[2, 2] = 1   # near-identical stray -> removed
    idx[4, 4] = 2   # bright glint -> kept
    out = remove_orphans(idx, pal)
    assert out[2, 2] == 0 and out[4, 4] == 2


def test_selout_picks_darker_shade_of_neighbour():
    pal = np.array([[20, 20, 20], [120, 30, 30], [240, 60, 60], [30, 30, 140], [90, 90, 255]], np.uint8)
    idx = np.full((5, 6), -1, np.int16)
    idx[1:4, 1:3] = 2   # light red
    idx[1:4, 3:5] = 4   # light blue
    ring = outline_ring(idx >= 0)
    colors = selout_colors(idx, ring, pal)
    assert colors[0, 1] == 1   # above red -> dark red
    assert colors[0, 4] == 3   # above blue -> dark blue


def test_selout_through_pipeline(blob):
    from pixel_pipeline import convert
    sprite = convert(blob, settings_for("object", outline_style="selout", palette="endesga-32"))
    assert sprite.rgba[..., 3].any()


# ---- idle animation --------------------------------------------------------------------------

def test_breathe_moves_torso_not_feet():
    rgba = figure()
    frames = breathe(rgba, frames=4, amount=1)
    assert len(frames) == 4
    assert np.array_equal(frames[0], rgba)
    squashed = frames[1]
    assert np.array_equal(squashed[-3:], rgba[-3:])          # feet planted
    assert not squashed[4].any() and squashed[5].any()        # top dropped by 1px


def test_bob_and_sway_keep_frame_size_and_pixels():
    rgba = figure()
    for frames in (bob(rgba, 4, 2), sway(rgba, 8, 1), idle_frames(rgba, "breathe")):
        assert len({f.shape for f in frames}) == 1
        assert all((f[..., 3] > 0).sum() <= (rgba[..., 3] > 0).sum() for f in frames)
    with pytest.raises(ValueError):
        idle_frames(rgba, "dance")


def test_pixel_scale_detects_upscaled_exports(tmp_path):
    rgba = figure()
    big = np.repeat(np.repeat(rgba, 8, 0), 8, 1)
    assert pixel_scale(big) == 8 and pixel_scale(rgba) == 1
    path = tmp_path / "big.png"
    Image.fromarray(big).save(path)
    assert load_sprite(path).shape == rgba.shape


# ---- video -----------------------------------------------------------------------------------

def make_video(path, n=24, fps=12):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 96))
    for i in range(n):
        f = np.full((96, 160, 3), (200, 180, 120), np.uint8)
        cv2.rectangle(f, (10 + i * 4, 40), (40 + i * 4, 80), (40, 60, 200), -1)
        vw.write(f)
    vw.release()


def test_read_frames_counts_and_timing(tmp_path):
    path = tmp_path / "clip.mp4"
    make_video(path)
    clip = read_frames(path, count=6)
    assert len(clip.frames) == 6 and clip.frames[0].shape == (96, 160, 4)
    assert clip.duration_ms == 333          # 2 s of video / 6 frames
    clip = read_frames(path, count=None, fps=6, start=0.5, end=1.5)
    assert len(clip.frames) == 6 and clip.duration_ms == 167


def test_align_track_vs_fixed():
    frames = []
    for i in range(3):
        f = np.zeros((50, 100, 4), np.uint8)
        f[20:40, 10 + 30 * i:30 + 30 * i] = [255, 0, 0, 255]
        frames.append(f)
    tracked = align_frames(frames, "track", margin=0)
    assert len({t.shape for t in tracked}) == 1
    assert all(np.array_equal(t, tracked[0]) for t in tracked)   # walks in place
    fixed = align_frames(frames, "fixed", margin=0)
    assert fixed[0].shape == (20, 80, 4)
    assert not np.array_equal(fixed[0], fixed[2])                # keeps the travel


def test_video_to_sprites_shares_palette(tmp_path):
    path = tmp_path / "clip.mp4"
    make_video(path)
    sprites, ms = video_to_sprites(path, settings_for("object", remove_background=False, width=24,
                                                      height=16), count=4)
    assert len(sprites) == 4 and ms == 500
    assert all(np.array_equal(s.palette, sprites[0].palette) for s in sprites)


# ---- Godot -----------------------------------------------------------------------------------

def test_sprite_frames_tres_text(tmp_path):
    tres = write_sprite_frames(tmp_path / "hero.tres", "hero.png", [(0, 0, 8, 8), (8, 0, 8, 8)],
                               name="walk", duration_ms=125)
    text = tres.read_text()
    assert '[gd_resource type="SpriteFrames"' in text and 'path="hero.png"' in text
    assert "region = Rect2(8, 0, 8, 8)" in text
    assert '"name": &"walk"' in text and '"speed": 8' in text


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to a Godot 4 binary to run")
def test_resources_load_in_godot(tmp_path):
    frames = [figure(), figure()]
    Image.fromarray(np.concatenate(frames, 1)).save(tmp_path / "hero.png")
    write_sprite_frames(tmp_path / "hero.tres", "hero.png", [(0, 0, 12, 20), (12, 0, 12, 20)],
                        name="idle")
    Image.fromarray(np.concatenate(frames, 1)).save(tmp_path / "tiles.png")
    write_tileset(tmp_path / "tiles.tres", "tiles.png", 4, 6, 30)
    (tmp_path / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (tmp_path / "check.gd").write_text(
        "extends SceneTree\nfunc _init():\n"
        "\tvar sf = load(\"res://hero.tres\")\n"
        "\tprint(\"FRAMES \", sf.get_frame_count(\"idle\"))\n"
        "\tvar ts = load(\"res://tiles.tres\")\n"
        "\tprint(\"TILES \", ts.get_source(0).get_tiles_count())\n\tquit()\n")
    subprocess.run([GODOT, "--headless", "--path", str(tmp_path), "--import"],
                   capture_output=True, timeout=120)
    out = subprocess.run([GODOT, "--headless", "--path", str(tmp_path), "-s", "check.gd"],
                         capture_output=True, text=True, timeout=120).stdout
    assert "FRAMES 2" in out and "TILES 30" in out


# ---- CLI -------------------------------------------------------------------------------------

def test_cli_idle_video_tileset_godot(tmp_path, blob_png):
    sprite = tmp_path / "s.png"
    assert main(["convert", str(blob_png), "-W", "24", "-H", "24", "--scale", "4",
                 "-o", str(sprite)]) == 0
    idle = tmp_path / "idle.png"
    assert main(["idle", str(sprite), "-o", str(idle), "--aseprite", "--godot"]) == 0
    assert Image.open(idle).size == (96, 24)                  # 4 frames at true 1x size
    assert idle.with_suffix(".tres").exists() and idle.with_suffix(".aseprite").exists()
    meta = json.loads(idle.with_suffix(".json").read_text())
    assert meta["meta"]["frameTags"][0]["name"] == "idle"

    clip = tmp_path / "clip.mp4"
    make_video(clip)
    walk = tmp_path / "walk.png"
    assert main(["video", str(clip), "-o", str(walk), "--frames", "4", "--no-bg-removal",
                 "-W", "20", "-H", "12", "--tag", "walk", "--godot"]) == 0
    assert Image.open(walk).size == (80, 12)
    assert '"name": &"walk"' in walk.with_suffix(".tres").read_text()

    tiles = tmp_path / "tiles.png"
    assert main(["tileset", str(idle), "--tile", "8", "-o", str(tiles), "--godot"]) == 0
    assert "TileSetAtlasSource" in tiles.with_suffix(".tres").read_text()
