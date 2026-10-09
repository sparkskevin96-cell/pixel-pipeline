"""Auto-rig on box humanoids built here (A-pose, T-pose, robe), plus the files it writes."""
import math
import os
import shutil
import subprocess

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from pixel_pipeline import settings_for  # noqa: E402
from pixel_pipeline.anim3d import animated_turnaround, load_rig  # noqa: E402
from pixel_pipeline.autorig import ACTIONS, auto_rig  # noqa: E402
from pixel_pipeline.cli import main  # noqa: E402

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")
SKIN, SHIRT, PANTS = (240, 190, 150), (40, 90, 220), (60, 60, 70)


def part(extents, centre, rgb, angle=0.0, pivot=None):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(centre)
    if angle:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(angle), [0, 0, 1], pivot))
    m.visual.vertex_colors = np.tile([*rgb, 255], (len(m.vertices), 1)).astype(np.uint8)
    return m


def humanoid(path, arms=40.0, robe=False):
    """1.8 tall. arms: degrees out from hanging straight down (90 = T-pose)."""
    parts = [part([0.42, 0.6, 0.26], [0, 1.15, 0], SHIRT),
             part([0.28, 0.3, 0.28], [0, 1.62, 0], SKIN)]
    if robe:
        parts.append(part([0.46, 0.85, 0.3], [0, 0.425, 0], PANTS))
    else:
        parts += [part([0.15, 0.85, 0.16], [s * 0.11, 0.425, 0], PANTS) for s in (1, -1)]
    for s in (1, -1):   # arm hangs from the shoulder (overlapping the torso), then swings out
        parts.append(part([0.12, 0.62, 0.1], [s * 0.25, 1.13, 0], SKIN, s * arms, [s * 0.22, 1.42, 0]))
    trimesh.util.concatenate(parts).export(path)
    return path


@pytest.fixture
def apose(tmp_path):
    return humanoid(tmp_path / "apose.glb")


def test_joints_of_an_a_pose(apose, tmp_path):
    res = auto_rig(apose, tmp_path / "a.rigged.glb")
    j, H = res.skeleton.joints, res.skeleton.height
    assert res.skeleton.notes == [] and H == pytest.approx(1.77, abs=0.03)
    assert j["hand_l"][0] > 0.45 and j["hand_r"][0] < -0.45            # out at the sides
    assert 0.8 < j["hand_l"][1] < 1.1
    assert abs(j["foot_l"][0] - 0.11) < 0.05 and j["foot_l"][1] < 0.12
    assert 0.8 < j["hips"][1] < 0.98 and j["head"][1] > 1.4
    assert j["upper_arm_l"][1] > 1.25 and j["shin_l"][1] == pytest.approx(0.45, abs=0.12)
    assert res.preview.shape[2] == 4 and res.path.exists()


def test_t_pose_and_robe(tmp_path):
    t = auto_rig(humanoid(tmp_path / "t.glb", arms=90), tmp_path / "t.rigged.glb")
    assert t.skeleton.notes == []
    assert t.skeleton.joints["hand_l"][0] > 0.6 and t.skeleton.tips["finger_l"][0] > 0.75
    assert abs(t.skeleton.joints["hand_l"][1] - 1.42) < 0.12
    robe = auto_rig(humanoid(tmp_path / "r.glb", robe=True), tmp_path / "r.rigged.glb")
    assert any("legs" in n for n in robe.skeleton.notes)
    assert len(load_rig([robe.path]).clips) == len(ACTIONS)


def test_animations_move_the_body_sensibly(apose, tmp_path):
    rig = load_rig([auto_rig(apose, tmp_path / "a.rigged.glb").path])
    clips = {c.name: c for c in rig.clips}
    assert [c.name for c in rig.clips] == list(ACTIONS)
    assert [clips[n].loop for n in ACTIONS] == [True, True, True, False, False, False]

    def verts(name, t):
        return np.concatenate(rig.vertices(rig.worlds(clips[name], t)))
    rest = np.concatenate(rig.vertices(rig.worlds()))
    H = np.ptp(rest[:, 1])
    for t in np.linspace(0, 1, 9):                                     # feet stay on the ground
        assert abs(verts("walk", t)[:, 1].min() - rest[:, 1].min()) < 0.03 * H
    stride = verts("walk", 0.25)
    feet = stride[:, 1] < 0.1 * H
    assert np.ptp(stride[feet, 2]) > 0.3                               # one foot ahead, one behind
    assert np.abs(verts("idle", 0.0) - verts("idle", clips["idle"].end)).max() < 1e-3
    arms_down = verts("idle", 0.0)
    assert arms_down[:, 0].max() < rest[:, 0].max() - 0.1              # arms brought down from the A
    windup = verts("attack", 0.35 * clips["attack"].end)
    assert windup[:, 1].max() > rest[:, 1].max()                       # weapon arm over the head
    dead = verts("death", clips["death"].end)
    assert dead[:, 1].max() < 0.45 * H and dead[:, 1].min() > -0.08 * H   # lying on the ground


def test_rigged_model_renders_and_keeps_colours(apose, tmp_path):
    res = auto_rig(apose, tmp_path / "a.rigged.glb", actions=["idle", "walk"])
    anim, _ = animated_turnaround([res.path], settings_for("person", width=32, height=32), 4, 2,
                                  light=0, elevation=0)
    front = anim.sprites[0][1][0].rgba
    cols = front[front[..., 3] > 0][:, :3].astype(int)
    assert (np.abs(cols - SHIRT).sum(1) < 60).mean() > 0.1
    assert (np.abs(cols - SKIN).sum(1) < 60).mean() > 0.05


def test_textured_model_keeps_its_texture(tmp_path):
    from PIL import Image
    mesh = humanoid(tmp_path / "plain.glb")
    m = trimesh.load(mesh, force="mesh")
    tex = Image.new("RGB", (4, 4), (200, 40, 160))
    uv = np.full((len(m.vertices), 2), 0.5)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=tex, baseColorFactor=[255, 255, 255, 255])
    m.visual = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    m.export(tmp_path / "tex.glb")
    res = auto_rig(tmp_path / "tex.glb", tmp_path / "tex.rigged.glb", actions=["idle"])
    anim, _ = animated_turnaround([res.path], settings_for("person", width=24, height=24), 4, 1, light=0)
    px = anim.sprites[0][1][0].rgba
    cols = px[px[..., 3] > 0][:, :3].astype(int)
    assert (np.abs(cols - (200, 40, 160)).sum(1) < 60).mean() > 0.5


def test_cli_rig_and_turnaround_auto_rig(apose, tmp_path, capsys):
    assert main(["rig", str(apose), "-o", str(tmp_path / "x.glb"), "--animations", "idle,walk",
                 "--preview", str(tmp_path / "x.png")]) == 0
    assert (tmp_path / "x.png").exists() and "skeleton + idle, walk" in capsys.readouterr().out
    out = tmp_path / "s" / "hero.png"
    assert main(["turnaround", str(apose), "--auto-rig", "--animations", "walk,attack", "-o", str(out),
                 "--directions", "4", "--frames", "2", "-W", "24", "-H", "24", "--godot"]) == 0
    assert "2 animations x 4 directions x 2 frames" in capsys.readouterr().out
    assert (out.parent / "hero.rigged.glb").exists() and (out.parent / "hero.tres").exists()


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to import the rigged model in Godot")
def test_godot_imports_the_rigged_model(apose, tmp_path):
    proj = tmp_path / "proj"
    proj.mkdir()
    auto_rig(apose, proj / "hero.glb")
    (proj / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (proj / "check.gd").write_text(
        "extends SceneTree\nfunc _init():\n"
        "\tvar n = load(\"res://hero.glb\").instantiate()\n"
        "\tvar ap = n.find_children(\"*\", \"AnimationPlayer\", true, false)[0]\n"
        "\tvar sk = n.find_children(\"*\", \"Skeleton3D\", true, false)[0]\n"
        "\tprint(\"R \", sk.get_bone_count(), \" \", \",\".join(ap.get_animation_list()))\n"
        "\tn.free()\n\tquit()\n")
    run = [GODOT, "--headless", "--path", str(proj)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    out = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=300).stdout
    line = next((x for x in out.splitlines() if x.startswith("R ")), out)
    assert line == "R 17 attack,death,hurt,idle,run,walk"


def test_a_floating_arm_is_still_found(tmp_path):
    parts = [part([0.42, 0.6, 0.26], [0, 1.15, 0], SHIRT), part([0.28, 0.3, 0.28], [0, 1.62, 0], SKIN)]
    parts += [part([0.15, 0.85, 0.16], [s * 0.11, 0.425, 0], PANTS) for s in (1, -1)]
    parts += [part([0.1, 0.62, 0.1], [s * 0.32, 1.13, 0], SKIN, s * 40, [s * 0.32, 1.42, 0]) for s in (1, -1)]
    trimesh.util.concatenate(parts).export(tmp_path / "f.glb")
    res = auto_rig(tmp_path / "f.glb", tmp_path / "f.rigged.glb", actions=["idle"])
    assert res.skeleton.notes == [] and res.skeleton.joints["hand_l"][0] > 0.4


def test_mcp_turnaround_auto_rig(apose, tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    result = asyncio.run(server.call_tool("turnaround", {
        "source": [str(apose)], "out_png": str(tmp_path / "h.png"), "directions": 4, "auto_rig": True,
        "animations": ["walk"], "frames": 2, "width": 24, "height": 24}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    assert "walk (2 frames each)" in content[0].text and (tmp_path / "h.rigged.glb").exists()
