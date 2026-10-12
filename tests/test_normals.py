"""Normal maps for rendered sprites: camera-space normals, aligned pixel for pixel."""
import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import settings_for
from pixel_pipeline.anim3d import animated_turnaround, write_animated
from pixel_pipeline.cli import main
from pixel_pipeline.normals import decode
from pixel_pipeline.turnaround import (render_model, turnaround_from, turnaround_sprites, union_crop,
                                       write_turnaround)

trimesh = pytest.importorskip("trimesh")


def _sphere(path, flip=False):
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
    if flip:
        mesh.invert()
    mesh.visual.vertex_colors = np.tile(np.array([200, 120, 60, 255], np.uint8),
                                        (len(mesh.vertices), 1))
    mesh.apply_translation([0, 0.5, 0])
    mesh.export(path)
    return path


@pytest.fixture
def sphere(tmp_path):
    return _sphere(tmp_path / "ball.glb")


def _sprites(path, s, directions=4):
    views, nviews = render_model(path, directions, elevation=0, size=96, points=120_000,
                                 normals=True)
    views, nviews = union_crop(views, s.margin, s.alpha_threshold, nviews)
    return turnaround_sprites(views, s, nviews)


def _check_sphere(n, mask):
    ys, xs = np.nonzero(mask)
    cy, cx = int(round(ys.mean())), int(round(xs.mean()))
    row = np.nonzero(mask[cy])[0]
    col = np.nonzero(mask[:, cx])[0]
    left, right = n[cy, row.min()], n[cy, row.max()]
    top, bottom = n[col.min(), cx], n[col.max(), cx]
    assert left[0] < 90 and right[0] > 166            # x: left edge points left
    assert top[1] > 166 and bottom[1] < 90             # y up (OpenGL convention)
    assert n[cy, cx, 2] > 240                          # centre faces the camera
    assert (n[mask, 2] >= 127).all()                   # never faces away


def test_raw_views_have_camera_space_normals(sphere):
    views, nviews = render_model(sphere, 4, elevation=0, size=96, points=120_000, normals=True)
    assert len(nviews) == 4
    for v, nv in zip(views, nviews):
        assert nv.shape == v.shape and (nv[..., 3] == v[..., 3]).all()
        _check_sphere(nv, nv[..., 3] > 0)
        vec = decode(nv[nv[..., 3] > 0][:, :3])
        assert np.allclose(np.linalg.norm(vec, axis=1), 1, atol=0.02)


def test_normal_sprites_match_colour_sprites(sphere):
    s = settings_for("object", width=24, height=24, colors=6)
    sprites = _sprites(sphere, s)
    for sp in sprites:
        assert sp.normal.shape == sp.rgba.shape
        assert ((sp.normal[..., 3] > 0) == (sp.rgba[..., 3] > 0)).all()   # same mask (outline too)
        _check_sphere(sp.normal, sp.rgba[..., 3] > 0)
        # averaged + renormalised, not snapped to a palette
        assert len(np.unique(sp.normal[sp.normal[..., 3] > 0][:, :3], axis=0)) > 6 + 1


def test_flipped_mesh_normals_face_the_camera(tmp_path):
    s = settings_for("object", width=24, height=24)
    sp = _sprites(_sphere(tmp_path / "inside_out.glb", flip=True), s, 4)[0]
    _check_sphere(sp.normal, sp.rgba[..., 3] > 0)


@pytest.mark.parametrize("downscale", ["area", "center", "median"])
def test_alignment_for_every_downscale(sphere, downscale):
    s = settings_for("object", width=20, height=28, downscale=downscale, outline=False)
    for sp in _sprites(sphere, s, 4):
        assert ((sp.normal[..., 3] > 0) == (sp.rgba[..., 3] > 0)).all()


def test_still_sheet_and_cli(sphere, tmp_path, capsys):
    s = settings_for("object", width=24, height=24)
    sprites, _ = turnaround_from([sphere], s, 4, elevation=30, normals=True)
    written = write_turnaround(sprites, tmp_path / "ball.png", columns=3, normals=True)
    assert [p.name for p in written] == ["ball.png", "ball.json", "ball_n.png"]
    sheet = np.asarray(Image.open(tmp_path / "ball.png").convert("RGBA"))
    nsheet = np.asarray(Image.open(tmp_path / "ball_n.png").convert("RGBA"))
    assert sheet.shape == nsheet.shape
    assert ((sheet[..., 3] > 0) == (nsheet[..., 3] > 0)).all()
    # default: no normal sheet, render_model still returns a plain list
    plain, _ = turnaround_from([sphere], s, 4)
    assert plain[0].normal is None
    assert write_turnaround(plain, tmp_path / "plain.png")[-1].name == "plain.json"
    assert not (tmp_path / "plain_n.png").exists()
    out = tmp_path / "c" / "ball.png"
    assert main(["turnaround", str(sphere), "-o", str(out), "--directions", "4", "-W", "16",
                 "-H", "16", "--normals"]) == 0
    assert Image.open(out.with_name("ball_n.png")).size == Image.open(out).size


def test_animated_sheet_writes_normals(tmp_path):
    from test_anim3d import write_robot
    robot = write_robot(tmp_path / "robot.glb")
    s = settings_for("person", width=24, height=24)
    anim, _ = animated_turnaround([robot], s, 4, 3, ["wave", "fall"], light=0, normals=True)
    assert all(sp.normal is not None for sp in anim.flat())
    written = write_animated(anim, tmp_path / "robot.png", normals=True)
    assert [p.name for p in written] == ["robot.png", "robot.json", "robot_n.png"]
    sheet = np.asarray(Image.open(tmp_path / "robot.png").convert("RGBA"))
    nsheet = np.asarray(Image.open(tmp_path / "robot_n.png").convert("RGBA"))
    assert sheet.shape == nsheet.shape
    assert ((sheet[..., 3] > 0) == (nsheet[..., 3] > 0)).all()
    assert (nsheet[nsheet[..., 3] > 0][:, 2] >= 127).all()
