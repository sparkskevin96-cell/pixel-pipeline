"""UI kit: font, pieces per theme and house style, 9-slice stretching, the mock screen, ui
assets, Godot / Unity exports, GPU icons against the fake ComfyUI and the UI kit page."""
import json
import os
import re
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import assets, uikit as U
from pixel_pipeline.assets import Style
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)


def colours(rgba):
    return {tuple(c) for c in rgba[rgba[..., 3] > 0][:, :3]}


def test_kind_and_folder():
    assert "ui" in assets.KINDS and assets.FOLDERS["ui"] == "ui"
    assert assets.current_project().kind_folder("ui").name == "ui"


def test_font_covers_ascii(tmp_path):
    for c in range(32, 127):
        assert chr(c) in U.KIT_GLYPHS, chr(c)
    fnt, png = U.write_font(tmp_path, 2)
    text = fnt.read_text()
    assert "chars count=95" in text and "lineHeight=20" in text and f'file="{png.name}"' in text
    ids = {int(m) for m in re.findall(r"char id=(\d+)", text)}
    assert ids == set(range(32, 127))
    atlas = np.asarray(Image.open(png))
    w, h = map(int, re.search(r"scaleW=(\d+) scaleH=(\d+)", text).groups())
    assert atlas.shape[:2] == (h, w)
    x, y, gw, gh = map(int, re.search(r"char id=65 x=(\d+) y=(\d+) width=(\d+) height=(\d+)", text).groups())
    assert atlas[y:y + gh, x:x + gw, 3].any()                     # "A" is drawn where it says


@pytest.mark.parametrize("theme", U.THEME_NAMES)
def test_every_theme_has_every_piece(theme):
    kit = U.make_kit(theme, Style(), 1)
    need = {"panel", "window", "dialogue", "nameplate", "inset", "button_normal", "button_hover",
            "button_pressed", "button_disabled", "tab_active", "tab_inactive", "checkbox_on",
            "checkbox_off", "slider_track", "slider_fill", "slider_knob", "scroll_track",
            "scroll_thumb", "bar_frame", "bar_health", "bar_mana", "bar_xp", "bar_seg_frame",
            "bar_seg_xp", "slot", "slot_selected", "cursor", "cursor_hand", "inventory",
            *[f"icon_{n}" for n in U.ICONS]}
    assert need <= set(kit.pieces)
    for name, p in kit.pieces.items():
        assert p.rgba.dtype == np.uint8 and p.rgba[..., 3].any(), name
        assert set(np.unique(p.rgba[..., 3])) <= {0, 255}, name          # binary alpha
        if p.margins:
            h, w = p.rgba.shape[:2]
            m = p.margins
            assert m["left"] + m["right"] < w and m["top"] + m["bottom"] < h, name
    for n in U.ICONS:
        assert kit.pieces[f"icon_{n}"].rgba.shape == (U.ICON, U.ICON, 4)
    # states differ; disabled is grey
    btn = {s: kit.pieces[f"button_{s}"].rgba for s in ("normal", "hover", "pressed", "disabled")}
    assert len({b.tobytes() for b in btn.values()}) == 4
    d = btn["disabled"][btn["disabled"][..., 3] > 0][:, :3].astype(int)
    assert np.abs(d[:, 0] - d[:, 1]).mean() < 12


def test_house_style_shapes_the_kit():
    soft = U.make_kit("wood", Style(shading="soft"), 1)
    flat = U.make_kit("wood", Style(shading="flat"), 1)
    assert len(colours(flat.pieces["button_normal"].rgba)) < len(colours(soft.pieces["button_normal"].rgba))
    solid = U.make_kit("wood", Style(outline="solid"), 1)
    assert solid.pieces["panel"].rgba[8, 0, :3].tolist() == list(U._rgb(U.OUTLINE_SOLID))
    none = U.make_kit("wood", Style(outline="none"), 1)
    assert tuple(U._rgb(U.OUTLINE_SOLID)) not in colours(none.pieces["icon_sword"].rgba)
    chunky = U.make_kit("crystal", Style(detail="chunky"), 1)
    normal = U.make_kit("crystal", Style(), 1)
    assert not np.array_equal(chunky.pieces["panel"].rgba, normal.pieces["panel"].rgba)  # no gems
    locked = U.make_kit("sci-fi", Style(palette="pico-8"), 1)
    from pixel_pipeline.palettes import load_palette
    pal = {tuple(c) for c in load_palette("pico-8")[:, :3].tolist()}
    for p in locked.pieces.values():
        assert colours(p.rgba) <= pal
    assert all(tuple(U._rgb(v)) in pal for v in locked.colours.values() if v)


def test_scale_and_nine():
    k1 = U.make_kit("stone", Style(), 1)
    k3 = U.make_kit("stone", Style(), 3)
    assert k3.image("panel").shape[0] == 3 * k1.pieces["panel"].rgba.shape[0]
    assert k3.margins("panel")["left"] == 3 * k1.pieces["panel"].margins["left"]
    assert U.default_scale(Style(tile=16)) == 1 and U.default_scale(Style(tile=32)) == 1
    assert U.default_scale(Style(tile=64)) == 2
    p = k1.pieces["window"]
    big = U.nine(p.rgba, p.margins, 90, 60)
    m = p.margins
    assert big.shape == (60, 90, 4)
    assert np.array_equal(big[:m["top"], :m["left"]], p.rgba[:m["top"], :m["left"]])          # corner kept
    assert np.array_equal(big[-m["bottom"]:, -m["right"]:], p.rgba[-m["bottom"]:, -m["right"]:])
    small = U.nine(p.rgba, p.margins, 8, 8)
    assert small.shape == (8, 8, 4)
    screen = U.mock_screen(k3)
    assert screen.shape == (U.SCREEN[1] * 3, U.SCREEN[0] * 3, 4)
    with pytest.raises(ValueError):
        U.make_kit("lava", Style())
    with pytest.raises(ValueError):
        U.make_kit("wood", Style(), 5)


def test_ui_asset(tmp_path):
    a = U.new_ui("Castle UI", "stone", scale=2)
    assert a.folder.parent.name == "ui" and a.kind == "ui"
    for rel in ("asset.json", "kit.json", "sheet.png", "preview.png", "thumb.png",
                "pieces/panel.png", "pieces/icon_heart.png", "fonts/pixelkit_2x.fnt",
                "fonts/pixelkit_4x.fnt"):
        assert (a.folder / rel).exists(), rel
    data = json.loads((a.folder / "kit.json").read_text())
    assert data["theme"] == "stone" and data["scale"] == 2
    panel = next(p for p in data["pieces"] if p["name"] == "panel")
    img = np.asarray(Image.open(a.folder / panel["file"]))
    assert img.shape[:2] == (panel["h"], panel["w"]) and panel["nine"]
    sheet = np.asarray(Image.open(a.folder / "sheet.png"))
    assert np.array_equal(sheet[panel["y"]:panel["y"] + panel["h"], panel["x"]:panel["x"] + panel["w"]], img)
    assert any(d["kind"] == "button" for d in data["demo"])
    assert [x.name for x in U.list_uis()] == ["Castle UI"]
    U.update_ui(a, theme="parchment", scale=1)
    data = json.loads((a.folder / "kit.json").read_text())
    assert data["theme"] == "parchment" and data["scale"] == 1
    assert U.open_ui(a.folder).extra["theme"] == "parchment"
    assert "ui" in {x.kind for x in assets.list_assets()}


def _ext_paths(text):
    return re.findall(r'\[ext_resource type="\w+" path="([^"]+)"', text)


def test_export_godot(tmp_path):
    kit = U.make_kit("metal", Style(), 2)
    out = tmp_path / "ui"
    U.export_kit(kit, out, "godot")
    theme = (out / "theme.tres").read_text()
    steps = int(theme.split("load_steps=")[1].split()[0])
    assert steps == theme.count("[ext_resource") + theme.count("[sub_resource") + 1
    for path in _ext_paths(theme):
        assert (out / path).exists(), path
    assert 'Button/styles/normal = SubResource("sb_button_normal")' in theme
    assert 'WindowPanel/base_type = &"Panel"' in theme and "default_font = ExtResource" in theme
    m = kit.margins("button_normal")
    block = theme.split('id="sb_button_normal"]')[1].split("[sub_resource")[0]
    assert f"texture_margin_left = {float(m['left'])}" in block
    assert "content_margin_top" in theme.split('id="sb_button_pressed"]')[1].split("[sub_resource")[0]
    demo = (out / "UIKitDemo.tscn").read_text()
    steps = int(demo.split("load_steps=")[1].split()[0])
    assert steps == demo.count("[ext_resource") + 1
    for path in _ext_paths(demo):
        assert (out / path).exists(), path
    for node in ("Options", "Inventory", "Dialogue", "Health", "Mana", "Xp"):
        assert f'[node name="{node}"' in demo
    assert (out / "uikit_demo.gd").exists() and (out / "README.md").exists()
    assert not (out / "preview.png").exists()
    with pytest.raises(ValueError):
        U.export_kit(kit, out, "gamemaker")


def test_export_unity(tmp_path):
    kit = U.make_kit("wood", Style(), 2)
    out = tmp_path / "UIKit"
    U.export_kit(kit, out, "unity")
    cs = (out / "Editor" / "PixelPipelineUIKit.cs").read_text()
    assert "public static Font BuildFont" in cs and "public static GameObject BuildDemo" in cs
    meta = (out / "pieces" / "button_normal.png.meta").read_text()
    m = kit.margins("button_normal")
    assert f"spriteBorder: {{x: {m['left']}, y: {m['bottom']}, z: {m['right']}, w: {m['top']}}}" in meta
    assert "filterMode: 0" in meta
    assert "spriteBorder: {x: 0, y: 0, z: 0, w: 0}" in (out / "pieces" / "icon_coin.png.meta").read_text()
    assert (out / "fonts" / "pixelkit_2x.png.meta").exists() and (out / "fonts" / "pixelkit_2x.fnt").exists()
    data = json.loads((out / "kit.json").read_text())
    assert isinstance(data["pieces"], list) and data["font"]["file"] == "fonts/pixelkit_2x.fnt"


def test_gpu_icons(comfy):
    fake, client = comfy
    a = U.new_ui("Hero UI", "crystal")
    before = np.asarray(Image.open(a.folder / "pieces" / "icon_coin.png")).copy()
    U.make_gpu_icons(a, client=client, seed=5)
    assert len(fake.prompts) == len(U.GPU_ICONS) and a.extra["icons"] == "gpu"
    assert sorted(p.stem for p in (a.folder / "icons_gpu").glob("*.png")) == sorted(U.GPU_ICONS)
    texts = " ".join(json.dumps(wf) for wf in fake.prompts)
    assert "gold coin" in texts and "icon" in texts
    after = np.asarray(Image.open(a.folder / "pieces" / "icon_coin.png"))
    assert after.shape == before.shape and not np.array_equal(after, before)
    U.update_ui(a, drawn_icons=True)
    assert a.extra["icons"] == "drawn" and not (a.folder / "icons_gpu").exists()
    assert np.array_equal(np.asarray(Image.open(a.folder / "pieces" / "icon_coin.png")), before)


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the theme and demo in Godot")
def test_theme_and_demo_in_godot(tmp_path):
    proj = tmp_path / "game"
    proj.mkdir()
    (proj / "project.godot").write_text('config_version=5\n[application]\nconfig/name="ui"\n')
    U.export_kit(U.make_kit("sci-fi", Style(), 2), proj / "ui" / "kit", "godot")
    (proj / "check.gd").write_text('''extends SceneTree
func _init():
	var th = load("res://ui/kit/theme.tres")
	var sb = th.get_stylebox("normal", "Button")
	var ok = th is Theme and sb is StyleBoxTexture and sb.texture != null and sb.texture_margin_left == 8.0
	ok = ok and th.default_font is FontFile and th.default_font.has_char(126)
	ok = ok and th.get_stylebox("fill", "ManaBar") is StyleBoxTexture and th.get_icon("checked", "CheckBox") != null
	var demo = load("res://ui/kit/UIKitDemo.tscn").instantiate()
	root.add_child(demo)
	await process_frame
	ok = ok and demo.get_node("Inventory").get_tab_count() == 3
	ok = ok and demo.get_node("Options/List/Resume").size.y >= 32
	ok = ok and demo.get_node("Inventory/Items/Grid/Slot2").get_theme_stylebox("panel").texture.resource_path.ends_with("slot_selected.png")
	ok = ok and demo.get_node("Mana").get_theme_stylebox("fill").texture.resource_path.ends_with("bar_mana.png")
	print("UI_OK" if ok else "UI_BAD")
	quit()
''')
    subprocess.run([GODOT, "--headless", "--path", str(proj), "--import"], capture_output=True,
                   timeout=300)
    r = subprocess.run([GODOT, "--headless", "--path", str(proj), "-s", "res://check.gd"],
                       capture_output=True, text=True, timeout=300)
    assert "UI_OK" in r.stdout, r.stdout + r.stderr


def test_uikit_tab(comfy, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.uikit_tab import UIKitTab
    fake, client = comfy
    tab = UIKitTab()
    tab.sync = True
    tab.client = client
    got = []
    tab.changed.connect(lambda: got.append(1))
    first = tab.view.rgba.copy()
    assert first.shape[:2] == (U.SCREEN[1], U.SCREEN[0])
    tab.theme.setCurrentText("parchment")
    assert not np.array_equal(first, tab.view.rgba)
    tab.scale.setCurrentIndex(tab.scale.findData(2))
    assert tab.view.rgba.shape[:2] == (U.SCREEN[1] * 2, U.SCREEN[0] * 2)
    tab.show_.setCurrentIndex(1)
    assert tab.view.rgba.shape != (U.SCREEN[1] * 2, U.SCREEN[0] * 2, 4)
    tab.name.setText("Scroll UI")
    asset = tab.save()
    assert got and tab.saved.count() == 1 and asset.extra["theme"] == "parchment"
    tab.make_icons()
    assert len(fake.prompts) == len(U.GPU_ICONS) and "GPU" in tab.icons_note.text()
    out = tab.export("godot", str(tmp_path))
    assert (out / "theme.tres").exists()
    out = tab.export("unity", str(tmp_path))
    assert (out / "Editor" / "PixelPipelineUIKit.cs").exists()
    tab.new_kit()
    assert tab.asset is None
    tab.open_kit(U.list_uis()[0])
    assert tab.theme.currentText() == "parchment" and tab.scale.currentData() == 2
