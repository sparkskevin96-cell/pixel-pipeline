"""Items and props (objects.py + ui/objects_tab.py) against the fake ComfyUI and procedural
models: no GPU."""
import json

import numpy as np
import pytest
from PIL import Image

from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

trimesh = pytest.importorskip("trimesh")

from pixel_pipeline import assets as A  # noqa: E402
from pixel_pipeline import objects as O  # noqa: E402
from pixel_pipeline.palettes import load_palette  # noqa: E402
from test_model3d import HUNYUAN, figure  # noqa: E402


def project(name="Game", **style):
    return A.create_project(name, A.Style(**style))


def barrel(path, height=1.0, radius=0.35):
    """An upright brown cylinder standing on y = 0 with a dark band (glTF: y up)."""
    m = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
    m.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0]))
    m.apply_translation([0, height / 2, 0])
    cols = np.tile(np.array([150, 95, 45, 255], np.uint8), (len(m.vertices), 1))
    m.visual.vertex_colors = cols
    m.export(path)
    return path


def colours(rgba):
    return {tuple(c) for c in rgba[rgba[..., 3] > 0][:, :3]}


def opaque_height(rgba):
    ys = np.nonzero(rgba[..., 3])[0]
    return ys.max() - ys.min() + 1


def ring_is_darkest(rgba):
    """Every opaque pixel touching transparency (or the edge) is the darkest colour."""
    a = np.pad(rgba[..., 3] > 0, 1)
    inner = a[1:-1, 1:-1]
    edge = inner & ~(a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:])
    lum = rgba[..., :3].astype(int) @ [299, 587, 114]
    return edge.any() and (lum[edge] == lum[inner].min()).all()


# --------------------------------------------------------------------------- helpers

def test_guess_prop_and_prompts():
    assert O.guess_prop("a small stone cottage") == (3.0, 4)       # building before stone
    assert O.guess_prop("Oak trees") == (2.5, 1)
    assert O.guess_prop("a wooden barrel") == (0.6, 1)
    assert O.guess_prop("a mysterious thing") == (1.0, 1)
    pos, neg = O.icon_prompt("pixel art of an iron sword", "weapon", 16)
    assert pos.startswith("pixel art game item icon of an iron sword") and "upright" in pos
    assert "chunky pixels" in pos and "multiple objects" in neg
    with pytest.raises(ValueError):
        O.icon_prompt("  ")


# --------------------------------------------------------------------------- items

def test_item_from_words_icons_follow_the_style(comfy, monkeypatch):
    """A 3D world sprite (style "3d": weapons default to flat, see the flat tests)."""
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    proj = project(icon=24, outline="solid", palette="pico-8", directions=8)
    steps = []
    item = O.make_item("Iron Sword", "an iron sword", category="weapon", project=proj,
                       client=client, seed=11, progress=steps.append, style="3d")
    f = item.folder
    assert f.parent == proj.kind_folder("item")
    # 3 icon pictures (pixel LoRA, icon prompt), then the 3D shape from the chosen one
    icon_wfs, (shape_wf,) = fake.prompts[:3], fake.prompts[3:]
    assert all(any(n["class_type"] == "LoraLoader" for n in wf.values()) for wf in icon_wfs)
    assert "game item icon of an iron sword" in next(
        n for n in icon_wfs[0].values() if n["_meta"]["title"] == "Prompt")["inputs"]["text"]
    assert any(n["class_type"] == "SaveGLB" for n in shape_wf.values())
    assert item.extra["model_from"] == 1

    icons = [np.asarray(Image.open(f / f"icon_v{i}.png")) for i in (1, 2, 3)]
    pico = {tuple(c) for c in load_palette("pico-8")}
    for icon in icons:
        assert icon.shape == (24, 24, 4)
        assert set(np.unique(icon[..., 3])) <= {0, 255}
        assert icon[0, 0, 3] == 0 and icon[-1, -1, 3] == 0            # transparent corners
        assert colours(icon) <= pico
        assert ring_is_darkest(icon)                                 # 1px solid outline
    assert not np.array_equal(icons[0], icons[1])                    # different seeds
    assert np.array_equal(np.asarray(Image.open(f / "icon.png")), icons[0])
    assert Image.open(f / "icon_2x.png").size == (48, 48)
    assert Image.open(f / "icon_4x.png").size == (96, 96)
    assert (f / "thumb.png").exists()

    meta = json.loads((f / "asset.json").read_text())
    assert meta["kind"] == "item"
    ex = meta["extra"]
    assert ex["seeds"] == [11, 12, 13] and ex["category"] == "weapon" and ex["size"] == "medium"
    assert ex["prompt"] == "an iron sword" and ex["icon"] == 24 and ex["icon_choice"] == 1
    assert ex["world"] and ex["world_from"] == "model" and ex["directions"] == 8

    for name in ("model.glb", "model.front.png", "model.shape.glb", "sheet.png", "sheet.json",
                 "sheet_n.png", "sheet.aseprite", "sheet.tres", "sheet_facing.gd"):
        assert (f / name).exists(), name
    frames = O.load_world(item)
    assert len(frames) == 8
    assert max(np.ptp(np.nonzero(frames[0][..., 3])[k]) + 1 for k in (0, 1)) <= 24 + 3
    assert any(isinstance(s, str) and "3D" in s for s in steps)

    # pick another variation: icon.png follows
    O.choose_icon(item, 2, proj)
    assert np.array_equal(np.asarray(Image.open(f / "icon.png")), icons[1])
    assert O.open_asset(f).extra["icon_choice"] == 2
    # a new style: icons redone from the saved pictures, no GPU
    sent = len(fake.prompts)
    proj.save_style(A.Style(icon=16, outline="solid", palette="pico-8", directions=8))
    O.redo_icons(item, proj)
    assert np.asarray(Image.open(f / "icon.png")).shape == (16, 16, 4)
    assert len(fake.prompts) == sent


def test_item_from_picture_flat_world_sprite(blob_png, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = project(icon=32, outline="selout", colours=8)
    item = O.make_item("Red Potion", picture=blob_png, category="potion", directions=1,
                       project=proj)
    f = item.folder
    assert [p.name for p in O.icon_paths(item)] == ["icon_v1.png"]
    icon = np.asarray(Image.open(f / "icon.png"))
    assert icon.shape == (32, 32, 4) and len(colours(icon)) <= 8 + 8   # palette + selout shades
    assert not (f / "model.glb").exists()                             # flat: no 3D, no GPU
    frames = O.load_world(item)
    assert len(frames) == 1 and frames[0].shape[0] == A.Style().scale(24) + 2
    normal = np.asarray(Image.open(f / "sheet_n.png"))
    assert (normal[frames[0][..., 3] > 0][:, :3] == (128, 128, 255)).all()
    assert json.loads((f / "sheet.json").read_text())["frames"][0]["filename"] == "default"
    assert 'name": &"default"' in (f / "sheet.tres").read_text()
    with pytest.raises(ValueError):                                   # no model: no 8 directions
        O.render_item(item, directions=8, project=proj)


def test_item_from_model_without_gpu(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = project(directions=4)
    figure(tmp_path / "fig.glb")
    item = O.make_item("Toy", model=tmp_path / "fig.glb", size="large", project=proj)
    assert np.asarray(Image.open(item.folder / "icon.png")).shape == (32, 32, 4)
    frames = O.load_world(item)
    assert len(frames) == 4
    longest = max(opaque_height(fr) for fr in frames)
    assert abs(longest - (A.Style().scale(36) + 2)) <= 3
    O.render_item(item, size="small", project=proj)                   # smaller, no GPU
    assert max(opaque_height(fr) for fr in O.load_world(item)) <= A.Style().scale(16) + 3
    assert item.extra["size"] == "small"
    with pytest.raises(ValueError):
        O.make_item("Nothing", project=proj)
    assert [a.name for a in A.list_assets("item", project=proj)] == ["Toy"]


def test_failed_item_leaves_no_folder(comfy, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    fake, client = comfy
    proj = project()
    fake.models["checkpoints"] = []                                   # no SDXL at all
    with pytest.raises(Exception):
        O.make_item("Broken", "a thing", project=proj, client=client)
    assert A.list_assets("item", project=proj) == []


# --------------------------------------------------------------------------- props

def test_prop_scale_follows_person_height(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    barrel(tmp_path / "barrel.glb")
    proj = project(elevation=0)                       # side view: on-screen height = height
    b = O.make_prop("Barrel", model=tmp_path / "barrel.glb", project=proj)
    assert b.extra["height_people"] == 0.6 and b.extra["directions"] == 1     # guessed
    frames = O.load_world(b)
    assert len(frames) == 1
    assert abs(opaque_height(frames[0]) - (round(0.6 * 64) + 2)) <= 1          # + outline
    for name in ("model.glb", "sheet.png", "sheet.json", "sheet_n.png", "sheet.aseprite",
                 "sheet.tres", "thumb.png"):
        assert (b.folder / name).exists(), name

    # redo at another height and with 4 directions: no GPU, files replaced
    O.render_prop(b, height_people=2.5, directions=4, project=proj)
    frames = O.load_world(b)
    assert len(frames) == 4 and (b.folder / "sheet_facing.gd").exists()
    assert abs(opaque_height(frames[0]) - (160 + 2)) <= 2
    saved = O.open_asset(b.folder).extra
    assert saved["height_people"] == 2.5 and saved["directions"] == 4
    O.rerender(b, project=proj, directions=1)
    assert len(O.load_world(b)) == 1 and not (b.folder / "sheet_facing.gd").exists()

    # a smaller project scale: half the pixels
    small = project("Tiny", elevation=0, person_height=32)
    s = O.make_prop("Barrel", model=tmp_path / "barrel.glb", height_people=0.6, project=small)
    assert abs(opaque_height(O.load_world(s)[0]) - (round(0.6 * 32) + 2)) <= 1


def test_prop_directions_and_tilt(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    figure(tmp_path / "statue.glb")
    proj = project(elevation=30, outline="none")
    st = O.make_prop("Statue", model=tmp_path / "statue.glb", project=proj, height_people=1.0)
    assert st.extra["directions"] == 4 and st.extra["elevation"] == 30       # statues: 4
    frames = O.load_world(st)
    assert len(frames) == 4 and len({f.shape for f in frames}) == 1
    assert not np.array_equal(frames[0], frames[2])                           # E vs W differ
    meta = json.loads((st.folder / "sheet.json").read_text())
    assert [t["name"] for t in meta["meta"]["frameTags"]] == ["E", "S", "W", "N"]
    with pytest.raises(ValueError):
        O.render_prop(st, directions=3, project=proj)


def test_prop_from_words(comfy, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    proj = project()
    house = O.make_prop("Cottage", "a small stone cottage", project=proj, client=client, seed=4,
                        look="3d")
    assert house.extra["height_people"] == 3.0 and house.extra["directions"] == 4
    assert house.extra["seed"] == 4 and house.extra["source"] == "words"
    pic_wf = fake.prompts[0]                      # then the shape (and maybe a back picture)
    assert any(n["class_type"] == "SaveGLB" for wf in fake.prompts for n in wf.values())
    text = next(n for n in pic_wf.values() if n["_meta"]["title"] == "Prompt")["inputs"]["text"]
    assert text.startswith("a single small stone cottage standing on its own, one object only")
    assert "product sheet" in next(n for n in pic_wf.values()
                                   if n["_meta"]["title"] == "Negative prompt")["inputs"]["text"]
    for name in ("model.glb", "model.front.png", "model.shape.glb", "sheet.png", "thumb.png"):
        assert (house.folder / name).exists(), name
    assert [a.name for a in A.list_assets("prop", project=proj)] == ["Cottage"]


# --------------------------------------------------------------------------- UI

@pytest.fixture
def app():
    from conftest import qt_app
    return qt_app()


def test_items_tab_make_pick_and_open(app, comfy, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    from pixel_pipeline.ui.objects_tab import ObjectsTab
    fake, client = comfy
    A.set_current_project("Default")
    A.current_project().save_style(A.Style(icon=16, directions=4))
    said = []
    tab = ObjectsTab("item", status=said.append)
    tab.sync, tab.client = True, client
    fired = []
    tab.changed.connect(lambda: fired.append(1))
    tab.prompt.setText("a red health potion")
    tab.name.setText("Health Potion")
    tab.category.setCurrentText("potion")
    tab.directions.setCurrentIndex(tab.directions.findData(1))        # flat: no 3D needed
    tab.make()
    assert tab.library.count() == 1 and fired
    asset = tab.asset
    assert asset.name == "Health Potion" and asset.extra["category"] == "potion"
    assert len(tab.var_buttons) == 3 and tab.var_buttons[0].isChecked()
    assert len(tab.shown["world"]) == 1 and tab.icon_view.rgba.shape == (16, 16, 4)
    tab.pick_icon(3)
    assert tab.var_buttons[2].isChecked() and not tab.var_buttons[0].isChecked()
    assert np.array_equal(np.asarray(Image.open(asset.folder / "icon.png")),
                          np.asarray(Image.open(asset.folder / "icon_v3.png")))
    assert O.open_asset(asset.folder).extra["icon_choice"] == 3
    tab._tick()
    # open it again from the list
    tab.new_object()
    assert tab.asset is None and not tab.var_buttons
    tab._item_clicked(tab.library.item(0))
    assert tab.asset.name == "Health Potion" and len(tab.var_buttons) == 3
    assert tab.var_buttons[2].isChecked()
    # redo with a bigger world size (no GPU)
    sent = len(fake.prompts)
    tab.size_.setCurrentText("large")
    tab.redo()
    assert len(fake.prompts) == sent and tab.asset.extra["size"] == "large"
    tab.deleteLater()


def test_props_tab_from_model_and_redo(app, tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    from pixel_pipeline.ui.objects_tab import ObjectsTab
    A.set_current_project("Default")
    A.current_project().save_style(A.Style(elevation=0))
    barrel(tmp_path / "barrel.glb")
    tab = ObjectsTab("prop")
    tab.sync = True
    tab.name.setText("Barrel")
    tab.from_model(tmp_path / "barrel.glb")
    assert tab.library.count() == 1 and tab.asset.extra["height_people"] == 0.6
    assert tab.height_.value() == pytest.approx(0.6) and len(tab.shown["spin"]) == 16
    assert len(tab.shown["world"]) == 1
    # "like a house": 3 people, 4 directions
    tab.like.setCurrentIndex(next(i for i in range(tab.like.count())
                                  if tab.like.itemText(i).startswith("like a house")))
    assert tab.height_.value() == 3.0 and tab.directions.currentData() == 4
    tab.redo()
    assert len(tab.shown["world"]) == 4
    assert abs(opaque_height(tab.shown["world"][0]) - (192 + 2)) <= 2
    tab.height_.setValue(1.2)
    assert tab.like.currentData() == "own"
    yaw = tab.asset.extra["yaw"]
    tab.redo(turn=90)
    assert tab.asset.extra["yaw"] == pytest.approx((yaw + 90) % 360)
    with pytest.raises(ValueError):
        ObjectsTab("map")
    tab.deleteLater()


def test_long_items_lie_diagonally_head_top_right():
    """A sword drawn hilt-up (as SDXL draws them): the icon puts the blade top right."""
    rgba = np.zeros((200, 60, 4), np.uint8)
    rgba[40:195, 24:36] = (200, 200, 210, 255)              # blade, long and wide
    rgba[30:40, 5:55] = (120, 120, 120, 255)                # crossguard: the widest part
    rgba[5:30, 27:33] = (110, 60, 30, 255)                  # grip: narrow
    out = O.diagonal(rgba)
    ratio, angle = O.elongation(out[..., 3])
    assert abs(angle - 45) < 8
    ys, xs = np.nonzero(out[..., 3] > 128)
    blade = (out[..., :3] == (200, 200, 210)).all(-1) & (out[..., 3] > 128)
    grip = (np.abs(out[..., :3].astype(int) - (110, 60, 30)).max(-1) < 30) & (out[..., 3] > 128)
    by, bx = np.nonzero(blade)
    gy, gx = np.nonzero(grip)
    assert bx.mean() > gx.mean() and by.mean() < gy.mean()   # blade up-right of the grip
    round_thing = np.zeros((60, 60, 4), np.uint8)
    round_thing[10:50, 10:50] = 255
    assert O.diagonal(round_thing) is round_thing             # not long: left alone


def test_lay_flat_puts_long_models_on_the_ground():
    pts = np.random.default_rng(0).random((1000, 3)) * [0.1, 1.0, 0.3]
    flat, _ = O.lay_flat(pts, pts.copy())
    ext = np.ptp(flat, 0)
    assert ext[0] == pytest.approx(1.0, abs=0.01) and ext[1] < 0.11 and flat[:, 1].min() == 0
    box = np.random.default_rng(0).random((1000, 3))
    assert O.lay_flat(box, box)[0] is box


def test_main_subject_drops_the_extra_things_beside_it():
    rgba = np.zeros((100, 200, 4), np.uint8)
    rgba[10:90, 10:70] = (150, 90, 40, 255)                 # the barrel
    rgba[20:30, 30:40, 3] = 0                               # (a hole: still one piece)
    rgba[40:60, 120:150] = (150, 90, 40, 255)               # another barrel beside it
    rgba[50:52, 69:72] = (150, 90, 40, 255)                 # touching the first: kept
    out = O.main_subject(rgba)
    assert out[50, 15, 3] == 255 and out[50, 130, 3] == 0 and out[51, 71, 3] == 255
    single = rgba.copy()
    single[:, 100:] = 0
    assert O.main_subject(single) is single


def test_repaint_puts_the_pictures_sharp_detail_on_the_front(tmp_path, monkeypatch):
    """A striped barrel picture painted onto a plain barrel shape keeps its stripes (the
    front must take the picture's own colours, not only the blurred guess)."""
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    from pixel_pipeline.turnaround import render_model
    proj = project(elevation=0, light=0.0)
    striped = barrel(tmp_path / "striped.glb")
    m = trimesh.load(striped, force="mesh")
    m = trimesh.Trimesh(m.vertices, m.faces)
    m = trimesh.Trimesh(*trimesh.remesh.subdivide_to_size(m.vertices, m.faces, 0.01))
    band = (np.asarray(m.vertices)[:, 1] * 25).astype(int) % 2 == 0
    cols = np.where(band[:, None], [40, 30, 20, 255], [220, 160, 90, 255]).astype(np.uint8)
    m.visual.vertex_colors = cols
    m.export(tmp_path / "striped.glb")
    pic = render_model(tmp_path / "striped.glb", 4, 0, size=256, light=0, points=400_000)[1]
    b = O.make_prop("Barrel", model=tmp_path / "striped.glb", height_people=1.0, project=proj)
    trimesh.Trimesh(m.vertices, m.faces).export(b.folder / "model.shape.glb")   # plain shape
    Image.fromarray(pic).save(b.folder / "model.front.png")
    O.repaint(b, project=proj)
    view = render_model(b.folder / "model.glb", 4, 0, size=256, light=0, points=400_000)[1]
    ys, xs = np.nonzero(view[..., 3])
    column = view[ys.min() + 5:ys.max() - 5, int(np.median(xs)), :3].astype(int).sum(1)
    assert column.max() - column.min() > 250                      # dark and light bands both there


def test_buildings_are_squared_to_the_grid(tmp_path, monkeypatch):
    """A house model turned 30 degrees renders with its walls along the grid."""
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    house = trimesh.creation.box(extents=[2.0, 1.0, 1.0])
    house.apply_translation([0, 0.5, 0])
    house.apply_transform(trimesh.transformations.rotation_matrix(np.radians(30), [0, 1, 0]))
    house.visual.vertex_colors = np.tile(np.array([150, 150, 160, 255], np.uint8),
                                         (len(house.vertices), 1))
    house.export(tmp_path / "house.glb")
    pts = np.asarray(house.sample(20000))
    assert abs(abs(O.square_turn(pts)) - 30) <= 1.5
    proj = project(elevation=0, outline="none")
    h = O.make_prop("House", model=tmp_path / "house.glb", height_people=1.0, project=proj)
    assert h.extra["directions"] == 4 and abs(abs(h.extra["yaw"]) - 30) <= 1.5
    widths = sorted(np.ptp(np.nonzero(f[..., 3])[1]) + 1 for f in O.load_world(h))
    assert widths[-1] > 1.8 * widths[0]               # long wall vs gable end, not corners
    O.render_prop(h, yaw=h.extra["yaw"] + 90, project=proj)
    assert O.open_asset(h.folder).extra["yaw"] == pytest.approx((h.extra["yaw"]) % 360)


# --------------------------------------------------------------------------- thin items, hand-pixelled props

def sword_picture(path=None):
    """A sword drawn upright on a transparent background, the way icon pictures come: a thin
    shaded blade (light left edge, dark right edge), a crossguard and a grip."""
    rgba = np.zeros((420, 120, 4), np.uint8)
    rgba[110:410, 54:61] = (215, 215, 225, 255)             # blade: 7 px of 300
    rgba[110:410, 61:66] = (120, 120, 135, 255)             # its shadowed edge
    rgba[90:110, 20:100] = (140, 100, 40, 255)              # crossguard
    rgba[20:90, 57:63] = (110, 60, 30, 255)                 # grip: narrower than the blade
    rgba[10:20, 55:65] = (140, 100, 40, 255)                # pommel
    if path is not None:
        Image.fromarray(rgba, "RGBA").save(path)
    return rgba


def test_thicken_keeps_thick_parts_bigger():
    pose = O.lying_pose(sword_picture())
    assert pose.shape[1] > 3 * pose.shape[0]                    # lies left to right
    solid = pose[..., 3] >= 128
    cols = solid.sum(0)
    xs = np.nonzero(cols)[0]
    assert np.argmax(cols) < pose.shape[1] / 2                  # the long blade goes right
    blade = O.thinnest(pose[..., 3])
    assert 8 <= blade <= 16
    out = O.thicken(pose, 40)
    thick = O.thinnest(out[..., 3])
    assert thick >= 36                                           # thin parts reach ~40 rows
    guard = (out[..., 3] >= 128).sum(0).max()
    assert guard >= cols.max() + 25                              # the crossguard grew too
    assert out.shape[1] == pose.shape[1]                         # no longer
    assert O.thicken(pose, 2) is pose                            # thick enough already


def test_weapon_world_sprite_drawn_flat_from_the_icon(tmp_path, monkeypatch):
    """Weapons default to "flat (from icon)": no 3D model or GPU, every direction from the
    icon picture, the blade thick enough to have a body inside its outline."""
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = project(icon=32, outline="selout", directions=8, palette="pico-8")
    sword_picture(tmp_path / "sword.png")
    item = O.make_item("Sword", picture=tmp_path / "sword.png", category="weapon", project=proj)
    f = item.folder
    assert O.world_style(item) == "flat" and item.extra["world_from"] == "icon"
    assert not (f / "model.glb").exists()
    frames = O.load_world(item)
    assert len(frames) == 8 and len({fr.shape for fr in frames}) == 1
    pico = {tuple(c) for c in load_palette("pico-8")}
    for fr in frames:
        assert colours(fr) <= pico
    east = frames[0]                                             # lying left to right
    solid = east[..., 3] > 0
    cols = solid.sum(0)[solid.any(0)]
    assert np.median(cols) >= 4                                  # 2 px of blade + outline
    assert max(np.ptp(np.nonzero(east[..., 3])[1]) + 1, 1) >= A.Style().scale(24) - 2
    # the light edge of the blade faces the light (up) lying east and lying west alike
    for fr in (frames[0], frames[4]):
        a = fr[..., 3] > 0
        lum = fr[..., :3].astype(int) @ [299, 587, 114]
        mid = np.nonzero(a)[0].mean()
        top = a & (np.arange(a.shape[0])[:, None] < mid)
        assert lum[top].mean() > lum[a & ~top].mean()
    normal = np.asarray(Image.open(f / "sheet_n.png"))
    assert len({tuple(c) for c in normal[normal[..., 3] > 0][:, :3]}) <= 2   # flat on the ground
    # another size and 1 direction: still no model needed; style can be switched
    O.render_item(item, size="large", directions=1, project=proj)
    one = O.load_world(item)
    assert len(one) == 1 and item.extra["world_from"] == "icon"
    with pytest.raises(ValueError):
        O.render_item(item, directions=8, style="3d", project=proj)   # 3D needs a model
    with pytest.raises(ValueError):
        O.render_item(item, style="wobbly", project=proj)


def test_world_style_auto_rules(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = project()
    sword_picture(tmp_path / "s.png")
    for cat, want in (("weapon", "flat"), ("key", "flat"), ("potion", "3d"), ("food", "3d"),
                      ("misc", "flat")):
        it = O.make_item(cat, picture=tmp_path / "s.png", category=cat, world=False,
                         project=proj)
        assert O.world_style(it) == want, cat
        assert O.world_style(it, "3d") == "3d"
    figure(tmp_path / "fig.glb")
    toy = O.make_item("Toy", model=tmp_path / "fig.glb", category="weapon", project=proj,
                      directions=4)
    assert O.world_style(toy) == "3d" and toy.extra["world_from"] == "model"   # your model


def test_equalise_faces_lifts_the_side_the_picture_saw_badly():
    rng = np.random.default_rng(0)
    n = 4000
    side = rng.integers(0, 4, n)                                 # four walls
    ang = side * np.pi / 2
    nrm = np.stack([np.sin(ang), np.zeros(n), np.cos(ang)], 1)
    pts = nrm * 1.0 + np.stack([np.zeros(n), rng.random(n), np.zeros(n)], 1)
    cols = np.tile(np.array([150, 150, 160], np.uint8), (n, 1))
    cols[side == 2] = (60, 60, 66)                               # the back: dark and greyer
    out = O.equalise_faces(pts, cols, nrm, 1.0)
    lum = out.astype(int).sum(1)
    assert abs(lum[side == 2].mean() - lum[side == 0].mean()) < 40
    assert O.equalise_faces(pts, cols, nrm, 0.0) is cols


def test_props_are_hand_pixelled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    barrel(tmp_path / "barrel.glb")
    proj = project(elevation=30, colours=24)
    b = O.make_prop("Barrel", model=tmp_path / "barrel.glb", project=proj)
    assert b.extra["hand_pixel"] is True
    hand = O.load_world(b)[0]
    O.render_prop(b, hand=False, project=proj)
    plain = O.load_world(b)[0]
    assert O.open_asset(b.folder).extra["hand_pixel"] is False
    assert hand.shape == plain.shape and not np.array_equal(hand, plain)
    assert len(colours(hand)) <= len(colours(plain))
    normal = np.asarray(Image.open(b.folder / "sheet_n.png"))
    assert normal.shape[:2] == plain.shape[:2]
    O.rerender(b, project=proj, hand=True)
    assert np.array_equal(O.load_world(b)[0], hand)                # same again: deterministic


def test_prop_drawn_from_words(comfy, monkeypatch):
    """From words a prop is drawn as pixel art (not a posterised 3D render): 3 versions to
    pick from, exactly as tall as asked in people, the project's camera in the prompt."""
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    proj = project(elevation=30)
    hut = O.make_prop("Cottage", "a small stone cottage", project=proj, client=client, seed=4)
    # a building in a three-quarter project: drawn again from the corner over its 3D shape
    assert (hut.folder / "drawn_corner.png").exists() and (hut.folder / "model.glb").exists()
    assert hut.extra["look"] == "drawn" and hut.extra["directions"] == 1
    assert hut.extra["height_people"] == 3.0
    for i in (1, 2, 3):
        assert (hut.folder / f"drawn_v{i}.png").exists() and (hut.folder / f"sprite_v{i}.png").exists()
    text = next(n for n in fake.prompts[0].values() if n["_meta"]["title"] == "Prompt")["inputs"]["text"]
    assert "pixel art" in text and "isometric" in text and "plain white background" in text
    assert any(n["class_type"] == "LoraLoader" for n in fake.prompts[0].values())   # pixel LoRA
    frames = O.load_world(hut)
    alpha = frames[0][..., 3] > 0
    rows = np.nonzero(alpha.any(1))[0]
    tall = rows.max() - rows.min() + 1
    assert abs(tall - 3 * proj.style.person_height) <= 4, tall               # outline room
    assert frames[0].shape[0] >= tall
    O.choose_drawn(hut, 2, project=proj)
    assert hut.extra["chosen"] == 2


def test_prop_drawn_four_sides_over_its_model(comfy, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    proj = project(elevation=30)
    tower = O.make_prop("Tower", "a stone watch tower", project=proj, client=client, seed=2,
                        directions=4, height_people=1.0)
    assert tower.extra["directions"] == 4 and (tower.folder / "model.glb").exists()
    for d in ("E", "S", "W", "N"):
        assert (tower.folder / f"drawn_{d}.png").exists()
    assert len(O.load_world(tower)) == 4
    sides = [wf for wf in fake.prompts if any(n["class_type"] == "VAEEncode" for n in wf.values())]
    assert len(sides) >= 4                                   # each side drawn over the model
