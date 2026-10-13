"""House style + project types (assets.py): style words reach every generator's prompt (the fake
ComfyUI captures what it was sent), palette lock, presets, 2 directions, the new-project
wizard. No GPU."""
import json

import numpy as np
import pytest
from PIL import Image

from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)

trimesh = pytest.importorskip("trimesh")

from pixel_pipeline import assets as A  # noqa: E402
from pixel_pipeline import library, maplib, model3d, objects, portraits, teams  # noqa: E402
from pixel_pipeline.generate import generate_sprite  # noqa: E402
from pixel_pipeline.palettes import load_palette  # noqa: E402
from pixel_pipeline.settings import settings_for  # noqa: E402
from pixel_pipeline.turnaround import direction_names  # noqa: E402
from test_model3d import HUNYUAN  # noqa: E402
from test_objects import barrel, colours  # noqa: E402

HOUSE = A.Style(style_prompt="16-bit SNES JRPG, bold outlines, warm palette, ZQXHOUSE",
                style_negative="ZQXNEVER", shading="cel", proportions="chibi", detail="chunky",
                outline="solid")


def texts(wf):
    """(positive, negative) prompt texts of one ComfyUI workflow."""
    pos = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "Prompt")
    neg = next(n for n in wf.values() if n.get("_meta", {}).get("title") == "Negative prompt")
    return pos["inputs"]["text"], neg["inputs"]["text"]


def image_prompts(fake):
    return [texts(wf) for wf in fake.prompts if any(n["class_type"] == "CLIPTextEncode" for n in wf.values())]


@pytest.fixture
def house(monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = A.create_project("House", HOUSE)
    A.set_current_project("House")
    return proj


# --------------------------------------------------------------------------- words

def test_style_words_per_kind():
    assert A.style_words(A.Style(), "sprite") == ("", "")            # old projects: unchanged
    for kind in A.STYLE_KINDS:
        assert A.style_words(None, kind) == ("", "")
    pos, neg = A.style_words(HOUSE, "character")
    assert pos.startswith("16-bit SNES JRPG, bold outlines, warm palette, ZQXHOUSE")
    assert "chibi" in pos and "cel shading" in pos and "chunky pixels" in pos
    assert "bold dark outline" in pos and neg.startswith("ZQXNEVER")
    item, _ = A.style_words(HOUSE, "icon")
    assert "chibi" not in item and "ZQXHOUSE" in item                # proportions: characters only
    model, mneg = A.style_words(HOUSE, "model")
    assert model.startswith("JRPG, bold outlines")                   # pixel-art words dropped
    assert "16-bit" not in model and "SNES" not in model and "pixel" not in model.lower()
    assert "chibi" in model and "bold dark outline" not in model and "ZQXNEVER" in mneg
    pos, neg = A.with_style("a sword, bold dark outline", "blurry", HOUSE, "icon")
    assert pos.count("bold dark outline") == 1 and neg.startswith("blurry, ZQXNEVER")
    with pytest.raises(ValueError):
        A.style_words(HOUSE, "nonsense")


def test_style_json_backward_compatible_and_round_trip(monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    old = A.projects_folder() / "Old"
    old.mkdir(parents=True)
    (old / "style.json").write_text(json.dumps({"palette": "pico-8", "person_height": 32,
                                                "directions": 8, "gone": 1}))
    st = A.Project("Old", old).style
    assert st.person_height == 32 and st.style_prompt == "" and st.shading == "soft"
    assert st.animations == [] and st.project_type == "custom"
    proj = A.create_project("Round", A.style_for_type("side-scroller", proportions="tall"))
    back = proj.style
    assert back.proportions == "tall" and back.animations[-1] == "crouch"
    assert A.set_shading(back, "flat").light == 0.0 and back.ramp_steps() == 2
    assert A.Style(proportions="chibi").heads_tall() == 2.5


# --------------------------------------------------------------------------- project types

def test_project_types_set_the_style(monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    want = {"hd2d": (30, 16, 64, "godot-hd2d", "hd2d"),
            "topdown-rpg": (55, 4, 32, "godot-2d", "topdown"),
            "side-scroller": (0, 2, 48, "godot-2d", "side"),
            "isometric": (30, 8, 48, "godot-2d", "iso"),
            "cards-ui": (30, 4, 64, "godot-2d", "none")}
    assert set(A.PROJECT_TYPES) == {*want, "custom"}
    for t, (elev, dirs, height, export, tiles) in want.items():
        st = A.create_project(f"P {t}", type=t).style
        assert (st.elevation, st.directions, st.person_height, st.export, st.tile_style) == \
            (elev, dirs, height, export, tiles), t
        assert st.project_type == t and st.style_prompt
    assert A.style_for_type("isometric").camera == "ortho"
    side = A.style_for_type("side-scroller")
    assert {"jump", "fall", "land", "climb", "crouch"} <= set(side.animations)
    assert A.missing_clips(side) == ["fall", "land", "climb", "crouch"]   # no motions yet
    assert A.extra_clips(side, have=A.PEOPLE_CLIPS) == ["jump"]
    assert set(A.style_for_type("topdown-rpg").animations) == set(A.PEOPLE_CLIPS)
    assert A.style_for_type("hd2d", "small").person_height == 48
    assert A.style_for_type("hd2d", engine="unity").export == "unity"
    assert A.create_project("Plain").style == A.Style()
    assert maplib.defaults_for(A.style_for_type("isometric")) == ("castle", "godot-iso")
    assert maplib.defaults_for(A.style_for_type("side-scroller", engine="unity")) == ("flat", "tiled")
    assert maplib.defaults_for(A.style_for_type("hd2d")) == ("harbour", "godot-3d")
    with pytest.raises(ValueError):
        A.create_project("Bad", type="racing")


# --------------------------------------------------------------------------- palette lock

def test_lock_palette_writes_palette_hex_and_is_used(blob_png, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = A.create_project("Locked", A.Style(icon=24, colours=16))
    rng = np.random.default_rng(3)
    for i in range(3):
        a = A.new_asset("item", f"Thing {i}", project=proj)
        img = np.zeros((16, 16, 4), np.uint8)
        img[..., :3] = rng.integers(0, 256, 3)
        img[4:12, 4:12, :3] = rng.integers(0, 256, 3)
        img[2:14, 2:14, 3] = 255
        Image.fromarray(img).save(a.folder / "icon.png")
    A.set_tags(A.list_assets(project=proj)[0], ["approved"])
    assert len(A.approved_assets(proj)) == 1
    path = A.lock_palette(proj, colours=8, assets=A.list_assets(project=proj))
    assert path == proj.folder / "palette.hex" and path.exists()
    pal = load_palette(str(path))
    assert 2 <= len(pal) <= 6                           # 6 distinct colours were given
    st = proj.style
    assert st.palette == str(path)                      # full path when read...
    assert json.loads((proj.folder / "style.json").read_text())["palette"] == "palette.hex"
    assert objects.style_settings(st, 16, 16).palette == str(path)
    item = objects.make_item("Blob", picture=blob_png, project=proj, world=False)
    icon = np.asarray(Image.open(item.folder / "icon.png").convert("RGBA"))
    assert colours(icon) <= {tuple(c) for c in pal}     # ...and the converter snaps to it
    with pytest.raises(ValueError):
        A.lock_palette(A.create_project("Empty"))


# --------------------------------------------------------------------------- every generator

def test_style_words_flow_into_every_generator(comfy, house, tmp_path, monkeypatch):
    fake, client = comfy
    fake.models["checkpoints"].append(HUNYUAN)
    monkeypatch.setattr(teams, "_face_model_path", None)

    # Sprite tab / generate_sprite (current project's style by default)
    generate_sprite("a knight", settings_for("person", width=24, height=32), client=client,
                    subject="person", seed=1)
    pos, neg = image_prompts(fake)[-1]
    assert "ZQXHOUSE" in pos and "chibi" in pos and "ZQXNEVER" in neg
    n = len(fake.prompts)
    generate_sprite("a knight", settings_for("person", width=24, height=32), client=client,
                    subject="person", seed=1, style=False)
    assert "ZQXHOUSE" not in image_prompts(fake)[-1][0] and len(fake.prompts) == n + 1

    # items: icons
    start = len(fake.prompts)
    objects.make_item("Sword", "an iron sword", category="weapon", project=house, client=client,
                      seed=3, world=False, variations=1)
    pos, neg = texts(fake.prompts[start])
    assert pos.startswith("pixel art game item icon of an iron sword") and "ZQXHOUSE" in pos
    assert "chibi" not in pos and "ZQXNEVER" in neg

    # props drawn as pixel art
    start = len(fake.prompts)
    objects.make_prop("Crate", "a wooden crate", project=house, client=client, seed=4)
    pos, neg = texts(fake.prompts[start])
    assert pos.startswith("pixel art") and "ZQXHOUSE" in pos and "ZQXNEVER" in neg

    # props from words as 3D: the clean picture (no pixel-art words) + its back
    start = len(fake.prompts)
    objects.make_prop("Barrel", "an oak barrel", project=house, client=client, seed=5, look="3d")
    pics = [texts(wf) for wf in fake.prompts[start:]
            if any(n["class_type"] == "CLIPTextEncode" for n in wf.values())]
    assert pics and all("JRPG" in p and "16-bit" not in p and "ZQXHOUSE" in p for p, _ in pics)
    assert pics[0][0].startswith("a single oak barrel standing on its own")

    # characters: Prompt to Model (front picture + back view)
    start = len(fake.prompts)
    model3d.prompt_to_model("a knight", tmp_path / "knight.glb", client=client, seed=6)
    pics = [texts(wf) for wf in fake.prompts[start:]
            if any(n["class_type"] == "CLIPTextEncode" for n in wf.values())]
    assert len(pics) >= 2 and all("chibi" in p and "ZQXHOUSE" in p and "SNES" not in p
                                  for p, _ in pics)
    assert all("ZQXNEVER" in n for _, n in pics)

    # portraits: the painted bust
    ch = library.Character("Hero", "person", "normal", "", 1.0, library.root() / "hero")
    ch.save()
    img = np.full((320, 160, 3), 255, np.uint8)
    img[20:90, 50:110] = (235, 190, 150)
    img[90:320, 20:140] = (60, 70, 160)
    Image.fromarray(img).save(ch.picture)
    start = len(fake.prompts)
    portraits.make_portraits(ch, ("happy",), client=client, project=house, painted_size=64, size=32)
    pos, neg = texts(fake.prompts[start])
    assert "head and shoulders portrait" in pos and "ZQXHOUSE" in pos and "JRPG" in pos
    assert "16-bit" not in pos and "ZQXNEVER" in neg


# --------------------------------------------------------------------------- 2 directions

def test_two_directions_for_side_scrollers(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    assert direction_names(2) == ["E", "W"]
    with pytest.raises(ValueError):
        direction_names(3)
    assert 2 in objects.DIRECTION_CHOICES
    proj = A.create_project("Side", type="side-scroller")
    prop = A.new_asset("prop", "Barrel", project=proj, height_people=0.6)
    barrel(prop.folder / "model.glb")
    objects.render_prop(prop, directions=2, project=proj)
    frames = objects.load_world(prop)
    assert len(frames) == 2 and (prop.folder / "thumb.png").exists()
    assert '&"E", &"W"' in (prop.folder / "sheet_facing.gd").read_text()
    assert not np.array_equal(frames[0], frames[1])          # facing right vs left


def test_library_renders_two_directions_with_the_projects_clips(tmp_path, monkeypatch):
    from test_autorig import humanoid
    from pixel_pipeline.autorig import auto_rig
    st = A.style_for_type("side-scroller", "small")
    st.animations = ["idle", "walk", "jump", "fall"]
    monkeypatch.setattr(library, "style", lambda: st)
    ch = library.Character("Runner", "person", "normal", "", 1.0, library.root() / "runner")
    ch.folder.mkdir(parents=True)
    auto_rig(humanoid(tmp_path / "h.glb"), ch.rigged, actions=("idle", "walk"))
    said = []
    assert library.add_project_clips(ch.rigged, "person", progress=said.append) == ["jump"]
    assert any("fall" in s and "skipped" in s for s in said)
    library.render_sheets(ch.rigged, ch.sheet, "person", "normal", frames=2)
    meta = json.loads(ch.sheet.with_suffix(".json").read_text())
    names = {f["filename"].rsplit(" ", 1)[0] for f in meta["frames"]}
    assert names == {f"{a}_{d}" for a in ("idle", "walk", "jump") for d in ("E", "W")}
    facing = ch.folder / "sheet_facing.gd"
    assert facing.exists() and '&"E", &"W"' in facing.read_text()


# --------------------------------------------------------------------------- wizard + restyle

@pytest.fixture
def app():
    from conftest import qt_app
    return qt_app()


def test_new_project_wizard_creates_a_project(app, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    from pixel_pipeline.ui.assets_tab import AssetsTab, NewProjectWizard, StyleDialog
    tab = AssetsTab()
    tab.sync = True
    dlg = NewProjectWizard(None, lock_from=tab.project.name)
    assert dlg.pages.currentIndex() == 0 and dlg.project_type() == "hd2d"
    dlg.set_type("side-scroller")
    assert "side" in dlg.about.text().lower()
    assert dlg.style_prompt.text() == A.style_for_type("side-scroller").style_prompt
    assert "48 px" in dlg.size.currentText()
    dlg.next_page()
    assert dlg.pages.currentIndex() == 0 and dlg.error.text()           # needs a name
    dlg.name.setText("Platformer")
    dlg.engine.setCurrentIndex(dlg.engine.findData("unity"))
    dlg.next_page()
    assert dlg.pages.currentIndex() == 1 and "left and right" in dlg.summary.text()
    dlg.style_prompt.setText("chunky 8-bit platformer, ZQXHOUSE")
    dlg.proportions.setCurrentIndex(dlg.proportions.findData("chibi"))
    dlg.shading.setCurrentIndex(dlg.shading.findData("flat"))
    dlg.size.setCurrentIndex(dlg.size.findData("large"))
    dlg.palette.setCurrentIndex(dlg.palette.findData("pico-8"))
    name, ptype, style, lock = dlg.values()
    dlg.setParent(None)
    assert (name, ptype, lock) == ("Platformer", "side-scroller", False)
    tab.new_project(name, style, ptype)
    st = A.current_project().style
    assert tab.project.name == "Platformer" and st.project_type == "side-scroller"
    assert st.directions == 2 and st.elevation == 0 and st.person_height == 64
    assert st.style_prompt.endswith("ZQXHOUSE") and st.proportions == "chibi"
    assert st.shading == "flat" and st.light == 0.0 and st.palette == "pico-8"
    assert st.export == "unity" and "jump" in st.animations
    assert tab.type_label.text() == "Side-scroller / platformer"

    # the look dialog keeps the house style and type fields, and lock-from-assets works
    look = StyleDialog(None, st, "Platformer", "Project look", lock_label="Lock")
    back = look.style()
    look.setParent(None)
    assert back.style_prompt == st.style_prompt and back.project_type == "side-scroller"
    assert back.directions == 2 and back.animations == st.animations
    a = A.new_asset("item", "Gem", project=tab.project)
    gem = np.zeros((8, 8, 4), np.uint8)
    gem[2:6, 2:6] = (40, 200, 90, 255)
    Image.fromarray(gem).save(a.folder / "icon.png")
    plan = tab.save_look(back, lock=True)
    assert tab.project.style.palette.endswith("palette.hex") and plan
    tab.new_project("Copy", None, "topdown-rpg", lock_from="Platformer")
    assert A.current_project().style.palette.endswith("palette.hex")
    assert A.current_project().style.directions == 4


def test_restyle_plan_and_rerender(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXEL_LIBRARY", raising=False)
    proj = A.create_project("Re", A.Style(directions=4, person_height=32))
    prop = A.new_asset("prop", "Barrel", project=proj, height_people=0.6, directions=1)
    barrel(prop.folder / "model.glb")
    objects.render_prop(prop, project=proj)
    before = Image.open(prop.folder / "sheet.png").size
    new = A.Style(directions=4, person_height=64, style_prompt="ZQXHOUSE")
    plan = A.restyle_plan(proj, proj.style, new)
    assert any("person_height 32 -> 64" in line for line in plan)
    assert any(line.startswith("prop Barrel: re-render") for line in plan)
    assert any("Style words changed" in line for line in plan)
    proj.save_style(new)
    done = A.restyle(proj)
    assert done == ["Barrel: restyled"]
    after = Image.open(prop.folder / "sheet.png").size
    assert after[1] > before[1]                           # twice the person height now
