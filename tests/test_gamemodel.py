"""Game-ready 3D models: remesh/decimate (gamemesh), unwrap/bake (uvbake), the model_game
folder with skin + clips carried over (gamemodel), engine exports, Godot import, Library UI."""
import json
import math
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

trimesh = pytest.importorskip("trimesh")
pytest.importorskip("skimage")

from pixel_pipeline import gamemesh as gm  # noqa: E402
from pixel_pipeline import gamemodel, uvbake  # noqa: E402
from pixel_pipeline.anim3d import load_rig  # noqa: E402
from pixel_pipeline.autorig import auto_rig  # noqa: E402

GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")
SKIN, SHIRT, PANTS = (240, 190, 150), (40, 90, 220), (60, 60, 70)


def part(extents, centre, rgb, angle=0.0, pivot=None):
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(centre)
    if angle:
        m.apply_transform(trimesh.transformations.rotation_matrix(math.radians(angle), [0, 0, 1], pivot))
    m.visual.vertex_colors = np.tile([*rgb, 255], (len(m.vertices), 1)).astype(np.uint8)
    return m


def humanoid(path):
    """Overlapping boxes (inner faces everywhere: what remeshing must clean) + a floater,
    densely subdivided like a scanned/generated mesh."""
    parts = [part([0.42, 0.6, 0.26], [0, 1.15, 0], SHIRT),
             part([0.28, 0.3, 0.28], [0, 1.62, 0], SKIN)]
    parts += [part([0.15, 0.85, 0.16], [s * 0.11, 0.425, 0], PANTS) for s in (1, -1)]
    for s in (1, -1):
        parts.append(part([0.12, 0.62, 0.1], [s * 0.25, 1.13, 0], SKIN, s * 40, [s * 0.22, 1.42, 0]))
    parts.append(part([0.02, 0.02, 0.02], [0.6, 1.9, 0.4], SKIN))          # a floater
    m = trimesh.util.concatenate(parts)
    v, f = np.asarray(m.vertices), np.asarray(m.faces)
    for _ in range(3):
        v, f = trimesh.remesh.subdivide(v, f)
    m = trimesh.Trimesh(v, f, process=False)
    centres = np.array([[0, 1.15, 0], [0, 1.62, 0], [0.11, 0.425, 0], [-0.11, 0.425, 0],
                        [0.4, 1.2, 0], [-0.4, 1.2, 0], [0.6, 1.9, 0.4]])
    rgb = np.array([SHIRT, SKIN, PANTS, PANTS, SKIN, SKIN, SKIN])
    near = np.argmin(((m.vertices[:, None] - centres[None]) ** 2).sum(2), 1)
    m.visual.vertex_colors = np.concatenate([rgb[near], np.full((len(near), 1), 255)], 1).astype(np.uint8)
    m.export(path)
    return path


@pytest.fixture(scope="module")
def rigged(tmp_path_factory):
    d = tmp_path_factory.mktemp("hero")
    model = humanoid(d / "model.glb")
    auto_rig(model, d / "rigged.glb")
    return d / "rigged.glb"


@pytest.fixture(scope="module")
def built(rigged):
    return gamemodel.make_game_model(rigged, kind="character", budget=1600, voxels=110,
                                     texture=256, name="Hero")


# --------------------------------------------------------------------------- gamemesh

def test_remesh_closes_boxes_drops_floaters_and_decimates(tmp_path):
    src = gamemodel.load_source(humanoid(tmp_path / "m.glb"))
    clean = gm.weld(src.v, src.f)
    e, cnt = gm.edges_of(clean.f)
    assert (cnt == 2).all()                           # each box is closed, but they overlap
    high, _ = gm.remesh(clean, voxels=90)
    k, _ = gm.components(high)
    e, cnt = gm.edges_of(high.f)
    assert k == 1 and (cnt == 2).all()                # one closed manifold surface, no floater
    # inner faces are gone: nothing sits deep inside the torso box
    inside = (np.abs(high.v[:, 0]) < 0.12) & (np.abs(high.v[:, 1] - 1.15) < 0.2) & \
             (np.abs(high.v[:, 2]) < 0.06)
    assert not inside.any()
    low = gm.decimate(high, 800)
    assert 760 <= low.tris <= 820
    e, cnt = gm.edges_of(low.f)
    assert (cnt >= 2).all()                           # still closed
    lo, hi = low.v.min(0), low.v.max(0)
    assert np.allclose(hi[1] - lo[1], 1.77, atol=0.05)   # keeps its height


def test_taubin_keeps_size_and_symmetry_mirrors():
    sphere = trimesh.creation.icosphere(4)
    rng = np.random.default_rng(0)
    v = sphere.vertices * (1 + 0.04 * rng.standard_normal((len(sphere.vertices), 1)))
    m = gm.Mesh(v, np.asarray(sphere.faces))
    sm = gm.taubin(m, 10)
    r0 = np.linalg.norm(m.v, axis=1)
    r1 = np.linalg.norm(sm.v, axis=1)
    assert r1.std() < 0.5 * r0.std() and abs(r1.mean() - r0.mean()) < 0.02
    sym, share = gm.symmetrize(sm, 0, 0.0)
    assert share > 0.6
    d, _ = __import__("scipy.spatial", fromlist=["cKDTree"]).cKDTree(sym.v).query(
        sym.v * np.array([-1, 1, 1]))
    assert np.median(d) < 0.01


# --------------------------------------------------------------------------- uvbake

def test_unwrap_has_no_flips_or_overlaps_and_fills_the_atlas():
    m = gm.Mesh(*[np.asarray(x) for x in (trimesh.creation.icosphere(3).vertices,
                                          trimesh.creation.icosphere(3).faces)])
    at = uvbake.unwrap(m, 512)
    assert len(at.f) == m.tris and at.uv.min() >= 0 and at.uv.max() <= 1
    assert not uvbake._flipped(at.uv, at.f).any()
    px, fi, _ = uvbake.rasterize(at.uv[at.f] * 512, 512, 512)
    cnt = np.bincount(px, minlength=512 * 512)
    assert (cnt > 1).sum() < 0.02 * (cnt > 0).sum()          # only shared edges double up
    assert (cnt > 0).mean() > 0.4                             # the atlas is used
    assert np.allclose(at.v, m.v[at.source])


def test_bake_puts_colours_where_they_belong():
    box = trimesh.creation.box(extents=[1, 1, 1])
    v, f = trimesh.remesh.subdivide_to_size(box.vertices, box.faces, 0.05)
    col = np.where(v[:, 1:2] > 0, [[220, 30, 30]], [[30, 30, 220]]).astype(float)
    surf = gm.Surface(v, f)
    low = gm.Mesh(np.asarray(box.vertices, float), np.asarray(box.faces))
    at = uvbake.unwrap(low, 128)
    bk = uvbake.bake(at, gm.vertex_normals(low.v, low.f), (surf, col), gm.Mesh(v, f))
    top = np.nonzero(at.v[:, 1] > 0.49)[0]
    uv = at.uv[at.f[np.all(np.isin(at.f, top), 1)]].mean(1)      # top face centres
    px = bk.colour[(uv[:, 1] * 128).astype(int), (uv[:, 0] * 128).astype(int)]
    assert (px[:, 0] > 150).all() and (px[:, 2] < 80).all()
    # the normal map turns the smooth-shaded box's bent normals back to the flat faces'
    fi = np.nonzero(np.all(np.isin(at.f, top), 1))[0]
    b = np.full((len(fi), 3), 1 / 3)
    n = (bk.vertex_normals[at.f[fi]] * b[..., None]).sum(1)
    t = (bk.tangents[at.f[fi], :3] * b[..., None]).sum(1)
    w = np.sign(bk.tangents[at.f[fi], 3].sum(1))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t -= n * (t * n).sum(1, keepdims=True)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    bt = np.cross(n, t) * w[:, None]
    tex = bk.normal[(uv[:, 1] * 128).astype(int), (uv[:, 0] * 128).astype(int)] / 127.5 - 1
    world = t * tex[:, :1] + bt * tex[:, 1:2] + n * tex[:, 2:3]
    world /= np.linalg.norm(world, axis=1, keepdims=True)
    assert n[:, 1].min() < 0.95 and (world[:, 1] > 0.97).all()
    pix = uvbake.quantize_texture(bk.colour, bk.mask, 4)
    assert len(np.unique(pix[bk.mask][:, :3], axis=0)) <= 4


# --------------------------------------------------------------------------- gamemodel

def test_game_model_folder(built, rigged):
    mg = built.folder
    for name in ("lod0.glb", "lod1.glb", "lod2.glb", "lod0_albedo.png", "lod0_normal.png",
                 "preview.png", "preview.gif", "wire.png", "info.json", "Hero.ppmodel"):
        assert (mg / name).exists(), name
    info = json.loads((mg / "info.json").read_text())
    tris = [lod["tris"] for lod in info["lods"]]
    assert tris[0] <= 1700 and tris[0] > tris[1] > tris[2]
    assert info["skinned"] and info["bones"] == 17 and "walk" in info["clips"]
    # same skeleton and every clip as the heavy rigged model; the clips bend the new mesh
    heavy, game = load_rig([rigged]), load_rig([mg / "lod0.glb"])
    assert sorted(c.name for c in game.clips) == sorted(c.name for c in heavy.clips)
    walk = next(c for c in game.clips if c.name == "walk")
    rest = game.vertices(game.worlds())[0]
    mid = game.vertices(game.worlds(walk, walk.duration / 2))[0]
    assert np.abs(mid - rest).max() > 0.05


def test_weights_follow_the_body(built):
    lod = built.lods[0]
    from pixel_pipeline.anim3d import Gltf
    doc = Gltf(built.folder / "lod0.glb").doc
    joint_names = [doc["nodes"][j]["name"] for j in doc["skins"][0]["joints"]]
    v, top = lod.atlas.v, lod.joints[:, 0]
    head = v[:, 1] > 1.6
    assert np.mean([joint_names[j] in ("head", "neck") for j in top[head]]) > 0.8
    foot = (v[:, 1] < 0.15) & (v[:, 0] > 0.05)
    assert np.mean([joint_names[j].endswith("_l") for j in top[foot]]) > 0.8
    assert np.allclose(lod.weights.sum(1), 1, atol=1e-5)


def test_textures_carry_the_painting(built):
    lod = built.lods[0]
    at, img = lod.atlas, lod.bake.colour
    face_c = at.v[at.f].mean(1)
    uv = at.uv[at.f].mean(1)
    px = img[np.clip((uv[:, 1] * img.shape[0]).astype(int), 0, img.shape[0] - 1),
             np.clip((uv[:, 0] * img.shape[1]).astype(int), 0, img.shape[1] - 1), :3].astype(int)
    head = face_c[:, 1] > 1.6
    torso = (np.abs(face_c[:, 0]) < 0.12) & (np.abs(face_c[:, 1] - 1.15) < 0.15)
    torso &= face_c[:, 2] > np.percentile(face_c[torso, 2], 70)          # its front
    assert np.abs(np.median(px[head], 0) - SKIN).max() < 40
    assert np.abs(np.median(px[torso], 0) - SHIRT).max() < 40


def test_pixel_look_and_static_prop(tmp_path):
    m = humanoid(tmp_path / "model.glb")
    g = gamemodel.make_game_model(m, kind="prop", look="pixel", budget=600, voxels=90,
                                  texture=64, colours=8, lods=2)
    mg = g.folder
    assert (mg / "lod0.obj").exists() and (mg / "lod0.mtl").exists() and not (mg / "lod2.glb").exists()
    alb = np.asarray(Image.open(mg / "lod0_albedo.png"))
    assert alb.shape[:2] == (64, 64) and len(np.unique(alb.reshape(-1, 3), axis=0)) <= 9
    from pixel_pipeline.anim3d import Gltf
    doc = Gltf(mg / "lod0.glb").doc
    assert doc["samplers"][0]["magFilter"] == gamemodel.NEAREST and "skins" not in doc
    assert not g.info["skinned"]


def test_engine_exports(built, tmp_path):
    from pixel_pipeline import exports
    asset = built.folder.parent                    # the folder holding rigged.glb + model_game
    gfiles = exports.export_asset(asset, "godot-3d-model", tmp_path / "godot")
    names = {p.name for p in gfiles}
    assert {"Hero.glb", "Hero_lod1.glb", "Hero_lod2.glb", "Hero.glb.import"} <= names
    imp = (tmp_path / "godot" / "Hero.glb.import").read_text()
    assert '"walk": {\n"settings/loop_mode": 1' in imp and '"death": {\n"settings/loop_mode": 0' in imp
    ufiles = exports.export_asset(asset, "unity-3d-model", tmp_path / "unity")
    pp = json.loads((tmp_path / "unity" / "Hero.ppmodel").read_text())
    assert len(pp["lods"]) == 3 and len(pp["skinBones"]) == 17 and pp["skinned"]
    walk = next(c for c in pp["clips"] if c["name"] == "walk")
    assert walk["loop"] and walk["curves"] and (tmp_path / "unity" / pp["lods"][0]["albedo"]).exists()
    assert (tmp_path / "unity" / "Editor" / "PixelPipelineModelImporter.cs").exists()
    assert any(p.suffix == ".ppmodel" for p in ufiles)


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to check the files in Godot")
def test_godot_imports_and_plays_the_game_model(built, tmp_path):
    from pixel_pipeline import exports
    proj = tmp_path / "proj"
    exports.export_asset(built.folder.parent, "godot-3d-model", proj)
    (proj / "project.godot").write_text('config_version=5\n[application]\nconfig/name="t"\n')
    (proj / "check.gd").write_text(
        "extends SceneTree\nfunc _initialize():\n\t_run.call_deferred()\n"
        "func _run():\n"
        "\tvar n = load(\"res://Hero.glb\").instantiate()\n\troot.add_child(n)\n"
        "\tvar ap = n.find_children(\"*\", \"AnimationPlayer\", true, false)[0]\n"
        "\tvar sk = n.find_children(\"*\", \"Skeleton3D\", true, false)[0]\n"
        "\tvar mi = n.find_children(\"*\", \"MeshInstance3D\", true, false)[0]\n"
        "\tvar mat = mi.mesh.surface_get_material(0)\n"
        "\tvar b = sk.find_bone(\"foot_l\")\n\tap.play(\"walk\")\n\tawait process_frame\n"
        "\tvar p0 = sk.get_bone_global_pose(b).origin\n\tap.seek(0.5, true)\n"
        "\tawait process_frame\n\tvar p1 = sk.get_bone_global_pose(b).origin\n"
        "\tprint(\"R \", sk.get_bone_count(), \" \", ap.get_animation(\"walk\").loop_mode, \" \", "
        "mat.albedo_texture != null, \" \", mat.normal_enabled, \" \", p0.distance_to(p1) > 0.02)\n"
        "\tquit()\n")
    run = [GODOT, "--headless", "--path", str(proj)]
    subprocess.run(run + ["--import"], capture_output=True, timeout=300)
    out = subprocess.run(run + ["-s", "check.gd"], capture_output=True, text=True, timeout=300).stdout
    line = next((x for x in out.splitlines() if x.startswith("R ")), out)
    assert line == "R 17 1 true true true"


# --------------------------------------------------------------------------- Library page

def test_library_page_exports_a_3d_model(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    from pixel_pipeline import assets as A
    from pixel_pipeline.ui.assets_tab import AssetsTab
    from pixel_pipeline.ui.game_model_dialog import GameModelDialog, can_make
    qt_app()
    barrel = A.new_asset("prop", "Box Man")
    humanoid(barrel.folder / "model.glb")
    Image.fromarray(np.full((16, 16, 4), 200, np.uint8), "RGBA").save(barrel.folder / "thumb.png")
    tab = AssetsTab()
    tab.sync = True
    tab.select("Box Man")
    assert tab.model_btn.isEnabled() and can_make(tab.selected)
    files = tab.export_model_selected(("godot",), tmp_path / "out", look="smooth", budget=500,
                                      voxels=80, texture=128, lods=2)
    assert (tmp_path / "out" / "Box_Man.glb").exists() and (tmp_path / "out" / "Box_Man_lod1.glb").exists()
    assert any(p.name == "preview.png" for p in files)
    dlg = GameModelDialog(tab.selected)
    assert len(dlg.frames) == 8 and "Built" in dlg.info.text()
    opts, engines, folder, rebuild = dlg.values()
    assert opts["look"] in ("smooth", "pixel") and engines == ("godot", "unity")
    dlg.setParent(None)
