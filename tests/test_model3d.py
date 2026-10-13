"""Prompt/picture -> 3D model, against the fake ComfyUI from test_generate (no GPU)."""
import numpy as np
import pytest
from PIL import Image

from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

trimesh = pytest.importorskip("trimesh")

from pixel_pipeline.model3d import (ShapeOptions, framed_on_white, hunyuan3d_workflow,  # noqa: E402
                                    orient_to_picture, paint_from_picture, prompt_to_model)
from pixel_pipeline.turnaround import render_model  # noqa: E402
from pixel_pipeline.generate import ComfyUIModelMissing  # noqa: E402

HUNYUAN = "hunyuan_3d_v2.1.safetensors"
R = trimesh.transformations.rotation_matrix


def figure(path, colours=True):
    """Red body, yellow head, and a sword sticking out on the figure's right (asymmetric)."""
    def part(m, rgb):
        if colours:
            m.visual.vertex_colors = np.tile(np.array([*rgb, 255], np.uint8), (len(m.vertices), 1))
        return m
    body = part(trimesh.creation.box(extents=[0.6, 1.0, 0.4]), (200, 40, 40))
    body.apply_translation([0, 0.5, 0])
    head = part(trimesh.creation.icosphere(subdivisions=2, radius=0.25), (240, 200, 60))
    head.apply_translation([0, 1.25, 0])
    sword = part(trimesh.creation.box(extents=[0.5, 0.08, 0.08]), (40, 40, 220))
    sword.apply_translation([-0.55, 0.7, 0])
    mesh = trimesh.util.concatenate([body, head, sword])
    mesh.export(path)
    return mesh


def front_picture(tmp_path):
    figure(tmp_path / "coloured.glb")
    return render_model(tmp_path / "coloured.glb", 4, elevation=0, size=256, light=0,
                        points=300_000)[1]


def test_workflow_matches_comfyui_template():
    wf = hunyuan3d_workflow("pic.png", seed=7, checkpoint=HUNYUAN)
    by_class = {n["class_type"]: n["inputs"] for n in wf.values()}
    assert by_class["EmptyLatentHunyuan3Dv2"]["resolution"] == 4096          # 2.1 template value
    assert by_class["KSampler"]["steps"] == 30 and by_class["KSampler"]["cfg"] == 5.0
    assert by_class["VoxelToMesh"] == {"voxel": ["8", 0], "algorithm": "surface net", "threshold": 0.6}
    mv = hunyuan3d_workflow("pic.png", seed=7, checkpoint="hunyuan3d-dit-v2-mv.safetensors")
    assert any(n["class_type"] == "Hunyuan3Dv2ConditioningMultiView" for n in mv.values())
    assert next(n for n in mv.values() if n["class_type"] == "EmptyLatentHunyuan3Dv2")["inputs"]["resolution"] == 3072


def test_orientation_is_found_for_a_tipped_over_model(tmp_path):
    pic = front_picture(tmp_path)
    mesh = figure(tmp_path / "shape.glb", colours=False)
    mesh.apply_transform(R(np.radians(90), [1, 0, 0]) @ R(np.radians(90), [0, 1, 0]))
    pts = mesh.sample(100_000)
    rot, match = orient_to_picture(pts, pic[..., 3], upright=False)   # any file: 24 axis turns
    assert match > 0.9
    turned = (np.asarray(mesh.vertices) @ rot.T)
    # back to glTF: standing along y, the sword on the picture's left (-x)
    assert np.ptp(turned[:, 1]) > 1.4
    assert turned[:, 0].min() < -0.7 and turned[:, 0].max() < 0.45


def test_paint_puts_picture_colours_on_the_front(tmp_path):
    pic = front_picture(tmp_path)
    mesh = figure(tmp_path / "shape.glb", colours=False)
    mesh.apply_transform(R(np.radians(-90), [0, 1, 0]))          # front now faces +x
    painted, match = paint_from_picture(tmp_path / "shape.glb", pic)
    assert match > 0.9
    painted.export(tmp_path / "painted.glb")
    view = render_model(tmp_path / "painted.glb", 4, elevation=0, size=128, light=0, points=200_000)[1]
    ys, xs = np.nonzero(view[..., 3])
    body = view[int(np.percentile(ys, 75)), int(np.median(xs)), :3].astype(int)
    head = view[int(np.percentile(ys, 8)), int(np.median(xs)), :3].astype(int)
    assert np.abs(body - (200, 40, 40)).max() < 40
    assert np.abs(head - (240, 200, 60)).max() < 40
    back = render_model(tmp_path / "painted.glb", 4, elevation=0, size=128, light=0, points=200_000)[3]
    by = np.nonzero(back[..., 3])[0]
    assert back[int(np.percentile(by, 75)), back.shape[1] // 2, 0] > 120   # red guessed behind too


def test_prompt_to_model_end_to_end(comfy, tmp_path):
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    steps = []
    res = prompt_to_model("a knight", tmp_path / "knight.glb", client=client, seed=5,
                          progress=lambda step, text: steps.append(step))
    assert steps == ["picture", "cutout", "shape", "colour"]
    assert res.model.exists() and res.picture.exists() and res.raw.exists() and res.seed == 5
    picture_wf, shape_wf = fake.prompts
    assert not any(n["class_type"] == "LoraLoader" for n in picture_wf.values())   # no pixel LoRA
    assert "facing the viewer" in next(n for n in picture_wf.values()
                                       if n["_meta"]["title"] == "Prompt")["inputs"]["text"]
    assert next(n for n in shape_wf.values() if n["class_type"] == "ImageOnlyCheckpointLoader")[
        "inputs"]["ckpt_name"] == HUNYUAN
    mesh = trimesh.load(res.model, force="mesh")
    cols = np.asarray(mesh.visual.vertex_colors)[:, :3]
    assert np.ptp(cols, 0).max() > 0 or cols.mean() < 240                      # painted, not blank
    views = render_model(res.model, 8, size=64, points=50_000)
    assert len(views) == 8 and all(v[..., 3].any() for v in views)


def test_missing_hunyuan_checkpoint_says_what_to_download(comfy, tmp_path):
    fake, client = comfy
    with pytest.raises(ComfyUIModelMissing, match="hunyuan_3d_v2.1.safetensors"):
        prompt_to_model("a knight", tmp_path / "k.glb", client=client, seed=1)


def test_picture_and_ready_shape_skip_the_gpu(tmp_path):
    pic = front_picture(tmp_path)
    Image.fromarray(pic).save(tmp_path / "front.png")
    figure(tmp_path / "shape.glb", colours=False)
    res = prompt_to_model(None, tmp_path / "out.glb", picture=tmp_path / "front.png",
                          shape=tmp_path / "shape.glb")
    assert res.raw is None and res.model.exists()


def test_framed_on_white_centres_the_subject():
    rgba = np.zeros((100, 300, 4), np.uint8)
    rgba[10:90, 200:240] = (200, 0, 0, 255)
    img = np.asarray(framed_on_white(rgba, size=200))
    ys, xs = np.nonzero((img < 250).any(-1))
    assert abs((xs.min() + xs.max()) / 2 - 100) <= 1 and ys.max() - ys.min() + 1 == 170


def test_shape_options_template_needs_image(comfy, tmp_path):
    import json
    fake, client = comfy
    bad = tmp_path / "wf.json"
    bad.write_text(json.dumps({"1": {"class_type": "SaveGLB", "inputs": {"mesh": ["2", 0],
                                                                        "filename_prefix": "x"}}}))
    from pixel_pipeline.model3d import shape_from_picture
    with pytest.raises(ValueError, match="image"):
        shape_from_picture(Image.new("RGB", (64, 64), "white"), client=client,
                           options=ShapeOptions(template=str(bad)))


def test_cli_model_with_direction_sprites(comfy, tmp_path, monkeypatch, capsys):
    from pixel_pipeline.cli import main
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    out = tmp_path / "k.glb"
    assert main(["model", "a knight", "-o", str(out), "--seed", "3", "--comfyui", client.base_url,
                 "--directions", "4", "-W", "24", "-H", "24", "--godot"]) == 0
    text = capsys.readouterr().out
    assert "Building the 3D shape" in text and "(seed 3)" in text
    for name in ("k.glb", "k.front.png", "k.shape.glb", "k_dirs.png", "k_dirs.tres", "k_dirs_facing.gd"):
        assert (tmp_path / name).exists(), name


def test_mcp_prompt_to_model(comfy, tmp_path, monkeypatch):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    monkeypatch.setenv("COMFYUI_URL", client.base_url)
    result = asyncio.run(server.call_tool("prompt_to_model", {
        "out_glb": str(tmp_path / "hero.glb"), "prompt": "a hero", "seed": 9, "directions": 8,
        "width": 24, "height": 24}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    kinds = [getattr(c, "type", "") for c in content]
    assert kinds == ["text", "image", "image"]
    assert "(seed 9)" in content[0].text and (tmp_path / "hero_dirs.tres").exists()


def test_plain_material_is_not_colours():
    """ComfyUI's SaveGLB writes the shape with a grey PBR material (no image, no uvs): that
    must still be painted from the picture."""
    from pixel_pipeline.model3d import has_colours
    box = trimesh.creation.box()
    box.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=[56, 56, 56, 255]))
    assert not has_colours(box.export(file_type="glb"))
    painted = trimesh.creation.box()
    painted.visual.vertex_colors = np.tile([[255, 0, 0, 255], [0, 0, 255, 255]], (4, 1))
    assert has_colours(painted.export(file_type="glb"))
    textured = trimesh.creation.box()
    uv = np.random.default_rng(0).random((len(textured.vertices), 2))
    tex = Image.fromarray(np.random.default_rng(1).integers(0, 255, (8, 8, 3), np.uint8))
    textured.visual = trimesh.visual.TextureVisuals(uv=uv, image=tex)
    assert has_colours(textured.export(file_type="glb"))


def test_parts_outside_the_picture_take_the_nearest_colour_not_black():
    """A shape that doesn't match its picture (e.g. a turned head) must not come out black
    where it reaches past the picture's outline."""
    from pixel_pipeline.model3d import paint_from_picture
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    pic = np.zeros((256, 256, 4), np.uint8)
    pic[40:216, 126:130] = (200, 40, 40, 255)         # outline much narrower than the shape
    mesh, _ = paint_from_picture(cube, pic, samples=40_000)
    cols = np.asarray(mesh.visual.vertex_colors)[:, :3].astype(int)
    assert (cols.max(1) < 40).mean() == 0


def test_only_these_colours():
    from pixel_pipeline.model3d import parse_colours, picture_to_colours
    assert parse_colours("black, White").tolist() == [[28, 28, 32], [238, 238, 236]]
    assert parse_colours("#ff0000, 00ff00").tolist() == [[255, 0, 0], [0, 255, 0]]
    with pytest.raises(ValueError, match="Unknown colour"):
        parse_colours("blak")
    # a black dog with a shaded (grey-blue) white chest and a thin streak of sun glare
    pic = np.zeros((200, 200, 4), np.uint8)
    pic[20:180, 20:180] = (30, 30, 34, 255)
    pic[20:180, 120:170] = (120, 128, 150, 255)      # white fur in shade
    pic[60:62, 30:110] = (200, 200, 205, 255)        # glare, 2 px thin
    out = picture_to_colours(pic, "black, white")
    assert (out[100, 140, :3] == [238, 238, 236]).all()          # shade still reads white
    assert (out[61, 70, :3] == [28, 28, 32]).all()               # glare is gone
    assert (out[0, 0] == pic[0, 0]).all()                        # background untouched


def test_mirror_back_and_colours_paint_only_those_colours(tmp_path):
    from pixel_pipeline.model3d import prompt_to_model
    box = trimesh.creation.box(extents=(2.0, 1.0, 0.8))
    shape = tmp_path / "shape.glb"
    box.export(shape)
    pic = np.zeros((100, 200, 4), np.uint8)
    pic[...] = (40, 40, 40, 255)
    pic[:, :80] = (150, 150, 160, 255)
    res = prompt_to_model(None, tmp_path / "m.glb", picture=pic, shape=shape, mirror_back=True,
                          colours="black, white")
    m = trimesh.load(res.model, force="mesh")
    cols = {tuple(c) for c in np.asarray(m.visual.vertex_colors)[:, :3].tolist()}
    assert cols <= {(28, 28, 32), (238, 238, 236)} and len(cols) == 2


def test_upright_model_is_turned_not_tipped(tmp_path):
    """An image-to-3D model is upright; a three-quarter picture must be matched by turning it
    about y at the real angle (not the nearest 90 degrees, and never tipped over)."""
    from pixel_pipeline.model3d import _ry, orient_to_picture_view
    pic = front_picture(tmp_path)
    mesh = figure(tmp_path / "shape.glb", colours=False)
    mesh.apply_transform(np.vstack([np.hstack([_ry(35), np.zeros((3, 1))]), [0, 0, 0, 1]]))
    pts = mesh.sample(100_000)
    rot, tilt, match = orient_to_picture_view(pts, pic[..., 3])
    assert match > 0.9
    assert np.allclose(rot[1], [0, 1, 0])                     # y stays up
    turned = pts @ rot.T
    assert np.ptp(turned[:, 0]) > np.ptp(turned[:, 2])        # wide side faces the camera again
