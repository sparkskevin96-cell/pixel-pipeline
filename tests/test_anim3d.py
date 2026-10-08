"""Animated directions from a rigged glTF, built by hand here (no 3D package needed)."""
import io
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import settings_for
from pixel_pipeline.anim3d import (Channel, GlbWriter, Gltf, animated_turnaround, clip_name,
                                   clip_times, load_rig, select_clips, write_animated,
                                   write_animated_preview)
from pixel_pipeline.cli import main

RED, BLUE = (220, 40, 40), (40, 60, 220)


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
    f = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    return v, np.array(f)


def quat_z(deg):
    h = np.radians(deg) / 2
    return [0.0, 0.0, np.sin(h), np.cos(h)]


def write_robot(path, anims=True, mesh=True):
    """A body (textured: top half red, bottom half blue) on bone 'root', and an arm (COLOR_0
    green, linear 0.5) on bone 'arm' sticking out along +x. Animations: 'wave' raises the arm
    90 degrees and lowers it (a loop); 'slide' moves the root 2 units along +z (root motion);
    'fall' tips the arm down and stays (not a loop)."""
    g = GlbWriter()
    nodes = [{"name": "Armature", "children": [1]},
             {"name": "root", "translation": [0, 0, 0], "children": [2]},
             {"name": "arm", "translation": [0.3, 1.0, 0]}]
    if mesh:
        bv, bf = box((-0.3, 0, -0.2), (0.3, 1.2, 0.2))
        av, af = box((0.3, 0.9, -0.1), (1.0, 1.1, 0.1))
        pos = np.vstack([bv, av])
        faces = np.vstack([bf, af + len(bv)])
        joints = np.zeros((len(pos), 4), np.uint8)
        joints[len(bv):, 0] = 1
        weights = np.zeros((len(pos), 4), np.uint8)
        weights[:, 0] = 255
        uv = np.where(pos[:, 1:2] > 0.6, [[0.25, 0.25]], [[0.25, 0.75]])
        uv[len(bv):] = 0.25
        colour = np.ones((len(pos), 4), np.float32)
        colour[len(bv):] = [0.0, 0.5, 0.0, 1.0]
        tex = Image.new("RGB", (2, 2))
        tex.putdata([RED, RED, BLUE, BLUE])
        png = io.BytesIO()
        tex.save(png, "PNG")
        ibm = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        ibm[1, :3, 3] = [-0.3, -1.0, 0]                          # inverse of the arm's bind pose
        g.doc["images"] = [{"bufferView": g.view(png.getvalue()), "mimeType": "image/png"}]
        g.doc["textures"] = [{"source": 0}]
        g.doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
        g.doc["skins"] = [{"joints": [1, 2], "inverseBindMatrices": g.accessor(
            ibm.transpose(0, 2, 1).reshape(2, 16), "MAT4")}]
        g.doc["meshes"] = [{"primitives": [{"attributes": {
            "POSITION": g.accessor(pos, "VEC3"),
            "TEXCOORD_0": g.accessor(uv, "VEC2"),
            "COLOR_0": g.accessor(colour, "VEC4"),
            "JOINTS_0": g.accessor(joints, "VEC4", 5121),
            "WEIGHTS_0": g.accessor(weights, "VEC4", 5121, normalized=True)},
            "indices": g.accessor(faces.reshape(-1), "SCALAR", 5123), "material": 0}]}]
        nodes.append({"name": "robot", "mesh": 0, "skin": 0})
    g.doc["nodes"] = nodes
    g.doc["scenes"] = [{"nodes": [0] + ([3] if mesh else [])}]
    if anims:
        t = g.accessor(np.array([[0.0], [0.5], [1.0]]), "SCALAR")
        wave = g.accessor(np.array([quat_z(0), quat_z(90), quat_z(0)]), "VEC4")
        slide = g.accessor(np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2]]), "VEC3")
        fall = g.accessor(np.array([quat_z(0), quat_z(-45), quat_z(-90)]), "VEC4")
        g.doc["animations"] = [
            {"name": name, "samplers": [{"input": t, "output": out}],
             "channels": [{"sampler": 0, "target": {"node": node, "path": p}}]}
            for name, out, node, p in [("Wave", wave, 2, "rotation"), ("Slide", slide, 1, "translation"),
                                       ("Armature|Fall", fall, 2, "rotation")]]
    g.save(path)
    return path


@pytest.fixture
def robot(tmp_path):
    return write_robot(tmp_path / "robot.glb")


def test_reader_and_clips(robot):
    rig = load_rig([robot])
    assert [c.name for c in rig.clips] == ["wave", "slide", "fall"]
    assert [c.loop for c in rig.clips] == [True, True, False]   # slide loops once root motion is ignored
    wave = rig.clips[0]
    tip = rig.vertices(rig.worlds(wave, 0.5))[0][8:].max(0)
    rest = rig.vertices(rig.worlds())[0][8:].max(0)
    assert rest[1] < 1.2 and tip[1] > 1.6                       # arm raised at the half-way point
    assert np.allclose(rig.vertices(rig.worlds(wave, 0.0))[0], rig.vertices(rig.worlds())[0], atol=1e-6)
    assert clip_times(wave, 4) == [0, 0.25, 0.5, 0.75]          # loop: the end is the start again
    assert clip_times(rig.clips[2], 3) == [0, 0.5, 1.0]         # one-shot: ends on its last pose
    assert Gltf(robot).accessor(5).max() == 1.0                 # normalized ubyte weights


def test_clip_names():
    from pathlib import Path
    assert clip_name("mixamo.com", Path("Knight Walk.glb"), 1) == "knight_walk"
    assert clip_name("Armature|mixamo.com|Layer0", Path("run.glb"), 1) == "run"
    assert clip_name("Take 001", Path("idle.glb"), 1) == "idle"
    assert clip_name("Attack 2", Path("hero.glb"), 1) == "attack_2"
    assert clip_name("Armature|Jump", Path("hero.glb"), 3) == "jump"


def test_interpolation_modes():
    times = np.array([0.0, 1.0])
    step = Channel(0, "translation", times, np.array([[0.0, 0, 0], [2.0, 0, 0]]), "STEP")
    assert step.at(0.9)[0] == 0 and step.at(1.0)[0] == 2
    cubic = Channel(0, "translation", times,
                    np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [2, 0, 0], [0, 0, 0]]], float),
                    "CUBICSPLINE")
    assert cubic.at(0.5)[0] == pytest.approx(1.0) and cubic.at(0.25)[0] == pytest.approx(0.3125)


def test_animated_sprites_share_size_palette_and_ground(robot):
    s = settings_for("person", width=32, height=32)
    anim, what = animated_turnaround([robot], s, 4, 4, ["wave"], light=0, elevation=0)
    assert "wave (4 frames each)" in what
    assert len(anim.sprites) == 1 and len(anim.sprites[0]) == 4 and anim.frames == 4
    flat = anim.flat()
    assert len({sp.rgba.shape for sp in flat}) == 1
    assert all(np.array_equal(sp.palette, flat[0].palette) for sp in flat)
    east = anim.sprites[0][0]

    def top(sp):
        return np.nonzero(sp.rgba[..., 3].any(1))[0].min()

    def bottom(sp):
        return np.nonzero(sp.rgba[..., 3].any(1))[0].max()
    assert top(east[2]) < top(east[0]) - 4                      # arm up at t = 0.5
    assert len({bottom(sp) for sp in flat}) == 1                # feet stay on one ground line
    # texture: v = 0 is the top of the image, so the body's top is red and its bottom blue
    front = anim.sprites[0][1][0].rgba                          # S, rest pose

    def near(rgb):
        return (np.abs(front[..., :3].astype(int) - rgb).sum(-1) < 120) & (front[..., 3] > 0)
    ys_red, ys_blue = np.nonzero(near(RED))[0], np.nonzero(near(BLUE))[0]
    assert len(ys_red) and len(ys_blue) and ys_red.mean() < ys_blue.mean()


def test_in_place_removes_root_motion(robot):
    s = settings_for("person", width=32, height=32)
    anim, _ = animated_turnaround([robot], s, 4, 4, ["slide"], light=0, elevation=0)
    east = anim.sprites[0][0]
    xs = {int(np.nonzero(sp.rgba[..., 3].any(0))[0].min()) for sp in east}
    assert len(xs) == 1                                          # walks on the spot
    moving, _ = animated_turnaround([robot], s, 4, 4, ["slide"], light=0, elevation=0, in_place=False)
    xs = [int(np.nonzero(sp.rgba[..., 3].any(0))[0].min()) for sp in moving.sprites[0][0]]
    assert xs[3] > xs[0] + 4                                     # moves towards +z = right in E


def test_animations_from_another_file_and_renaming(tmp_path):
    model = write_robot(tmp_path / "hero.glb", anims=False)
    moves = write_robot(tmp_path / "moves.glb", mesh=False)
    rig = load_rig([model, moves])
    assert [c.name for c in rig.clips] == ["wave", "slide", "fall"]
    clips = select_clips(rig, ["hello=wave", "2"])
    assert [c.name for c in clips] == ["hello", "fall"]
    with pytest.raises(ValueError, match="have: wave, slide, fall"):
        select_clips(rig, ["dance"])
    with pytest.raises(ValueError, match="no animations"):
        select_clips(load_rig([model]))


def test_write_animated_files(robot, tmp_path):
    s = settings_for("person", width=24, height=24)
    anim, _ = animated_turnaround([robot], s, 4, 3, ["wave", "fall"], light=0)
    written = write_animated(anim, tmp_path / "robot.png", aseprite=True, godot=True)
    names = [p.name for p in written]
    assert names == ["robot.png", "robot.json", "robot.aseprite", "robot.tres", "robot_facing.gd"]
    sheet = Image.open(tmp_path / "robot.png")
    w, h = anim.flat()[0].size
    assert sheet.size == (3 * w, 2 * 4 * h)                      # rows: action x direction
    meta = json.loads((tmp_path / "robot.json").read_text())
    tags = {t["name"]: (t["from"], t["to"]) for t in meta["meta"]["frameTags"]}
    assert tags["wave_E"] == (0, 2) and tags["fall_N"] == (21, 23)
    assert meta["frames"][0]["duration"] == 333 and meta["frames"][12]["duration"] == 500
    tres = (tmp_path / "robot.tres").read_text()
    assert '&"wave_SE"' not in tres and '"name": &"fall_W"' in tres
    assert tres.count('"loop": true') == 4 and tres.count('"loop": false') == 4
    gd = (tmp_path / "robot_facing.gd").read_text()
    assert 'const ACTIONS: Array[StringName] = [&"wave", &"fall"]' in gd and "var facing := 1" in gd
    gif = write_animated_preview(anim, tmp_path / "robot.gif", scale=2)
    assert Image.open(gif).size == (4 * w * 2, h * 2)    # every direction side by side


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to check the files in Godot")
def test_godot_plays_action_and_facing(robot, tmp_path):
    s = settings_for("person", width=24, height=24)
    anim, _ = animated_turnaround([robot], s, 8, 3, ["wave", "fall"], light=0)
    write_animated(anim, tmp_path / "robot.png", godot=True)
    (tmp_path / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (tmp_path / "check.gd").write_text(
        "extends SceneTree\nfunc _init():\n"
        "\tvar s = AnimatedSprite2D.new()\n"
        "\ts.sprite_frames = load(\"res://robot.tres\")\n"
        "\ts.set_script(load(\"res://robot_facing.gd\"))\n"
        "\troot.add_child(s)\n\tawait process_frame\n"
        "\tprint(\"A \", s.animation, \" \", s.sprite_frames.get_animation_names().size(), \" \",\n"
        "\t\ts.sprite_frames.get_frame_count(&\"wave_NE\"), \" \",\n"
        "\t\ts.sprite_frames.get_animation_loop(&\"fall_E\"))\n"
        "\ts.face(Vector2(-1, -1))\n\tprint(\"B \", s.animation)\n"
        "\ts.act(&\"fall\")\n\tprint(\"C \", s.animation)\n"
        "\tquit()\n")
    run = [GODOT, "--headless", "--path", str(tmp_path)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=180)
    out = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=180).stdout
    lines = {x[0]: x[2:] for x in out.splitlines() if x[:2] in ("A ", "B ", "C ")}
    assert lines["A"] == "wave_S 16 3 false", out
    assert lines["B"] == "wave_NW" and lines["C"] == "fall_NW"


def test_cli_animate(robot, tmp_path, capsys):
    out = tmp_path / "o" / "robot.png"
    assert main(["turnaround", str(robot), "-o", str(out), "--animate", "--animations", "wave,fall",
                 "--frames", "3", "--directions", "4", "-W", "24", "-H", "24", "--godot",
                 "--preview", "2"]) == 0
    text = capsys.readouterr().out
    assert "2 animations x 4 directions x 3 frames" in text
    for name in ("robot.png", "robot.json", "robot.tres", "robot_facing.gd", "robot_anim.gif"):
        assert (out.parent / name).exists(), name


def test_mcp_animated_turnaround(robot, tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    result = asyncio.run(server.call_tool("turnaround", {
        "source": [str(robot)], "out_png": str(tmp_path / "r.png"), "directions": 4,
        "animate": True, "frames": 2, "animations": ["wave"], "width": 24, "height": 24}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    assert "wave (2 frames each)" in content[0].text and getattr(content[1], "type", "") == "image"
    assert (tmp_path / "r.tres").exists()


def test_colours_from_the_original_model(tmp_path):
    """Mixamo-style: the rigged copy lost its colours and is 100x bigger; the original has them."""
    trimesh = pytest.importorskip("trimesh")
    rigged = write_robot(tmp_path / "rigged.glb")
    bv, bf = box((-0.3, 0, -0.2), (0.3, 1.2, 0.2))
    av, af = box((0.3, 0.9, -0.1), (1.0, 1.1, 0.1))
    body, arm = trimesh.Trimesh(bv * 100, bf), trimesh.Trimesh(av * 100, af)
    body.visual.vertex_colors = np.tile([240, 200, 60, 255], (8, 1)).astype(np.uint8)
    arm.visual.vertex_colors = np.tile([200, 0, 200, 255], (8, 1)).astype(np.uint8)
    trimesh.util.concatenate([body, arm]).export(tmp_path / "original.glb")
    s = settings_for("person", width=32, height=32)
    anim, _ = animated_turnaround([rigged], s, 4, 1, ["wave"], light=0, elevation=0,
                                  colours_from=tmp_path / "original.glb")
    front = anim.sprites[0][1][0].rgba
    inside = front[..., 3] > 0
    cols = front[inside][:, :3].astype(int)
    yellow = (np.abs(cols - (240, 200, 60)).sum(1) < 60).mean()
    magenta = (np.abs(cols - (200, 0, 200)).sum(1) < 80).mean()
    assert yellow > 0.4 and magenta > 0.05
    assert not (np.abs(cols - RED).sum(1) < 60).any()             # the rigged copy's own colours are gone


def test_export_for_rigging(tmp_path):
    pytest.importorskip("trimesh")
    from pixel_pipeline.model3d import export_for_rigging
    obj = export_for_rigging(write_robot(tmp_path / "r.glb"), tmp_path / "r.obj")
    text = obj.read_text()
    assert text.count("\nv ") >= 16 and "\nf " in text
