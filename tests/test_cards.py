"""Cards: pixel font, frames, layout, composed cards, art against the fake ComfyUI, engine
exports and the Cards tab (no GPU)."""
import json
import os
import shutil
import subprocess

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import assets, cards, library
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)


def classes(wf):
    return [n["class_type"] for n in wf.values()]


def ksampler(wf):
    return next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]


def hero():
    """A library character with a picture on white (head, body, a sword)."""
    ch = library.Character("Hero Knight", "person", "normal", "swordsman", 1.0,
                           library.root() / "hero-knight", "a knight in blue armour")
    ch.save()
    img = np.full((320, 160, 3), 255, np.uint8)
    img[20:90, 50:110] = (235, 190, 150)
    img[90:320, 20:140] = (60, 70, 160)
    img[60:300, 140:146] = (200, 200, 210)
    Image.fromarray(img).save(ch.picture)
    return ch


def potion():
    a = assets.new_asset("item", "Red Potion", prompt="a red potion", icon_choice=1)
    img = np.full((256, 256, 3), 255, np.uint8)
    img[60:220, 70:190] = (200, 30, 40)
    img[30:60, 110:150] = (120, 80, 40)
    Image.fromarray(img).save(a.folder / "picture_v1.png")
    return a


# --------------------------------------------------------------------------- basics

def test_card_kind_registered():
    assert "card" in assets.KINDS and assets.FOLDERS["card"] == "cards"
    assert assets.current_project().kind_folder("card").name == "cards"


def test_font_glyphs_and_wrap():
    for ch, g in cards.GLYPHS.items():
        assert g.shape[0] == cards.FONT_ROWS, ch
        rows = cards._GLYPHS_SRC.get(ch, "").split()
        assert len({len(r) for r in rows}) <= 1, f"ragged glyph {ch!r}"
    for c in "AZaz09.,!?":
        assert c in cards.GLYPHS
    f = cards.FONT
    assert f.width("I") == 3 and f.width("II") == 7 and f.width("II", 2) == 14
    assert f.width("’") == f.width("'")                 # curly quote -> ascii
    assert (f.mask("~") == f.mask("?")).all()                # unknown -> ?
    assert f.mask("Hi", 3).shape == (cards.FONT_ROWS * 3, f.width("Hi", 3))
    lines = f.wrap("Deal 6 damage to a unit and 2 to its neighbours.", 60)
    assert len(lines) > 1 and all(f.width(line) <= 60 for line in lines)
    assert f.wrap("a\n\nb", 100) == ["a", "", "b"]
    assert all(f.width(x) <= 20 for x in f.wrap("Supercalifragilistic", 20))   # long word split


def test_bmfont(tmp_path):
    fnt, png = cards.write_bmfont(tmp_path, "f", 2, outline=True)
    text = fnt.read_text()
    assert "lineHeight=20" in text and f'file="{png.name}"' in text
    assert f"char id={ord('A')} " in text
    atlas = np.asarray(Image.open(png))
    assert (atlas[..., 3] > 0).any() and atlas.shape[1] > 0


@pytest.mark.parametrize("size", list(cards.SIZES))
def test_layout_slots_inside(size):
    for stats in ("unit", "power", "none"):
        lay = cards.card_layout(size, stats)
        W, H = cards.SIZES[size]
        assert (lay["width"], lay["height"]) == (W, H)
        for name, s in lay["slots"].items():
            if s is None:
                continue
            assert s["x"] >= 0 and s["y"] >= 0 and s["x"] + s["w"] <= W and s["y"] + s["h"] <= H, name
        a = lay["slots"]["art"]
        assert abs(a["w"] / a["h"] - 1.5) < 0.05
        assert (lay["slots"]["attack"] is not None) == (stats == "unit")
        assert (lay["slots"]["power"] is not None) == (stats == "power")
    assert abs(W / H - 63 / 88) < 0.01


def test_frame_window_and_palette():
    img, lay = cards.render_frame("legendary", "fire", "medium", "unit")
    a = lay["slots"]["art"]
    hole = img[a["y"] + a["h"] // 2:a["y"] + a["h"] - 2, a["x"] + a["w"] // 2:a["x"] + a["w"] - 2]
    assert (hole[..., 3] == 0).all()                         # art goes underneath
    c = lay["slots"]["cost"]
    assert img[c["y"] + c["h"] // 2, c["x"] + c["w"] // 2, 3] == 255
    assert (img[..., 3] > 0).mean() > 0.4
    alpha = set(np.unique(img[..., 3]).tolist())
    assert alpha <= {0, 255}                                  # crisp
    pal = np.array([[0, 0, 0], [255, 255, 255], [200, 40, 40], [40, 40, 200], [220, 180, 60]],
                   np.uint8)
    snapped, _ = cards.render_frame("rare", "water", "small", "none", pal)
    on = snapped[..., 3] > 0
    used = {tuple(c) for c in snapped[on][:, :3]}
    assert used <= {tuple(c) for c in pal}
    # every theme renders, and they differ
    a1, _ = cards.render_frame("common", "neutral")
    a2, _ = cards.render_frame("epic", "shadow")
    assert not np.array_equal(a1, a2)
    with pytest.raises(ValueError):
        cards.render_frame("mythic")
    with pytest.raises(ValueError):
        cards.render_frame(faction="space")


def test_compose_card_text_and_numbers():
    info = cards.CardInfo("Swordswoman", 3, "Unit - Warrior", "Charge.", 4, 5)
    frame, lay = cards.render_frame("common", "nature", "small", info.stats)
    blank, _ = cards.compose_card(cards.CardInfo("", None, "", "", None, None, None, "common",
                                                 "nature"), None, *cards.render_frame(
        "common", "nature", "small", "unit"))
    card, used = cards.compose_card(info, None, frame, lay)
    assert card.shape == (176, 126, 4)
    for slot in ("name", "cost", "attack", "health", "type"):
        s = lay["slots"][slot]
        region = (slice(s["y"], s["y"] + s["h"]), slice(s["x"], s["x"] + s["w"]))
        assert not np.array_equal(card[region], blank[region]), slot
    assert used["text"]["lines"] == ["Charge."] and not used["text"]["overflow"]
    long = cards.CardInfo("X", 1, "Spell", "word " * 80)
    _, used = cards.compose_card(long)
    assert used["text"]["overflow"] and used["text"]["lines"][-1].endswith("..")
    name = cards.CardInfo("An Extremely Long Card Name That Never Ends")
    _, used = cards.compose_card(name)
    assert used["name"]["text"].endswith("..")


def test_nine_slices_and_write_frame(tmp_path):
    pieces = cards.nine_slices("rare", "holy", "small")
    assert set(pieces) == {"border", "panel", "dark", "banner"}
    img, m = pieces["border"]
    n = img.shape[0]
    assert img.shape[1] == n and n == 3 * m["left"]
    assert img[n // 2, n // 2, 3] == 0 and pieces["panel"][0][n // 2, n // 2, 3] == 255
    written = cards.write_frame(tmp_path, "rare", "holy", "small", "unit", palette=None)
    names = {p.relative_to(tmp_path).as_posix() for p in written}
    assert {"frame.png", "layout.json", "ui/border.png", "ui/border.tres", "ui/panel.json"} <= names
    lay = json.loads((tmp_path / "layout.json").read_text())
    assert lay["rarity"] == "rare" and lay["colours"]["ink"]
    tres = (tmp_path / "ui" / "banner.tres").read_text()
    assert "StyleBoxTexture" in tres and "texture_margin_left" in tres


def test_to_pixel_art_and_fit():
    img = np.zeros((400, 600, 4), np.uint8)
    img[..., 0] = np.linspace(0, 255, 600)[None]
    img[..., 3] = 255
    sp = cards.to_pixel_art(img, 108, 72, assets.Style(colours=8))
    assert sp.rgba.shape == (72, 108, 4) and (sp.rgba[..., 3] == 255).all()
    assert len(np.unique(sp.rgba[..., :3].reshape(-1, 3), axis=0)) <= 8
    small = np.zeros((36, 54, 4), np.uint8)
    small[::2, ::2] = 255
    big = cards.fit_art(small, 108, 72)
    assert big.shape == (72, 108, 4) and set(np.unique(big[..., 0]).tolist()) <= {0, 255}  # crisp


# --------------------------------------------------------------------------- art + assets

def test_character_card_painted(comfy):
    fake, client = comfy
    ch = hero()
    src = cards.source_from("character", ch)
    assert src.kind == "character" and src.prompt == "a knight in blue armour"
    info = cards.default_info(src)
    assert info.type == "Unit" and info.attack and info.health      # from the swordsman class
    a = cards.new_card(info, src)
    assert a.kind == "card" and (a.folder / "card.png").exists()
    assert not (a.folder / "art.png").exists()                       # placeholder until made
    said = []
    cards.make_art(a, client=client, progress=said.append)
    assert len(fake.prompts) == 6                        # 3 x (scene inpaint + blend pass)
    scene, blend = fake.prompts[0], fake.prompts[1]
    assert "SetLatentNoiseMask" in classes(scene) and ksampler(scene)["denoise"] == 1.0
    assert "SetLatentNoiseMask" not in classes(blend)
    assert ksampler(blend)["denoise"] == cards.DENOISE[("character", "painted")]
    assert not any("LoraLoader" in classes(wf) for wf in fake.prompts)     # painted: no LoRA
    text = next(n["inputs"]["text"] for n in scene.values() if n["class_type"] == "CLIPTextEncode")
    assert "a knight in blue armour" in text and "battlefield" in text
    assert [ksampler(fake.prompts[i])["seed"] for i in (0, 2, 4)] == \
        [a.extra["seeds"][0] + k for k in range(3)]
    w, h = cards._art_window(info)
    assert [p.name for p in cards.art_variations(a)] == ["art_v1.png", "art_v2.png", "art_v3.png"]
    art = np.asarray(Image.open(a.folder / "art.png"))
    assert art.shape[:2] == (h, w)
    assert Image.open(a.folder / "art_hd.png").size == (w * cards.HD, h * cards.HD)
    data = json.loads((a.folder / "card.json").read_text())
    assert data["name"] == "Hero Knight" and data["attack"] == info.attack
    assert data["art"] == "art.png" and data["layout"]["slots"]["art"]["w"] == w
    assert "power" not in data                                       # not on this card
    before = (a.folder / "art.png").read_bytes()
    cards.choose_art(a, 2)
    assert a.extra["art_choice"] == 2 and (a.folder / "art.png").read_bytes() != before
    with pytest.raises(ValueError):
        cards.choose_art(a, 9)
    assert [c.name for c in cards.list_cards()] == ["Hero Knight"]
    again = cards.open_card(a.folder)
    assert again.extra["art_choice"] == 2


def test_item_card_pixel_and_size_change(comfy):
    fake, client = comfy
    item = potion()
    src = cards.source_from("item", item)
    a = cards.new_card(cards.default_info(src), src, art_style="pixel")
    cards.make_art(a, client=client, count=2)
    assert len(fake.prompts) == 4
    scene, pixel = fake.prompts[0], fake.prompts[1]
    assert "LoraLoader" not in classes(scene) and "LoraLoader" in classes(pixel)
    neg = [n["inputs"]["text"] for n in pixel.values() if n["class_type"] == "CLIPTextEncode"]
    assert any("several bottles" in t for t in neg)
    art = np.asarray(Image.open(a.folder / "art.png"))
    assert len(np.unique(art[..., :3].reshape(-1, 3), axis=0)) <= assets.Style().colours
    cards.update_card(a, size="large", text="Restore 5 health.")
    w, h = cards._art_window(cards._info_of(a))
    assert Image.open(a.folder / "art.png").size == (w, h)
    assert Image.open(a.folder / "card.png").size == cards.SIZES["large"]
    assert json.loads((a.folder / "card.json").read_text())["text"] == "Restore 5 health."
    with pytest.raises(TypeError):
        cards.update_card(a, colour="red")
    with pytest.raises(ValueError):
        cards.update_card(a, rarity="mythic")


def test_words_cards(comfy):
    fake, client = comfy
    src = cards.source_from("words", words="a fireball")
    info = cards.CardInfo("Fireball", 4, "Spell", "Deal 6 damage.", power=6, faction="fire")
    a = cards.new_card(info, src)
    cards.make_art(a, client=client)
    assert len(fake.prompts) == 3 and all(ksampler(wf)["denoise"] == 1.0 for wf in fake.prompts)
    assert all("EmptyLatentImage" in classes(wf) for wf in fake.prompts)
    lat = next(n for n in fake.prompts[0].values() if n["class_type"] == "EmptyLatentImage")
    assert lat["inputs"]["width"] > lat["inputs"]["height"]          # landscape like the window
    fake.prompts.clear()
    b = cards.new_card(info, src, art_style="pixel")
    cards.make_art(b, client=client, count=1)
    assert len(fake.prompts) == 2                     # clean scene, then the pixel LoRA over it
    assert "LoraLoader" not in classes(fake.prompts[0]) and "LoraLoader" in classes(fake.prompts[1])
    assert ksampler(fake.prompts[1])["denoise"] == cards.PIXEL_OVER_DENOISE
    with pytest.raises(ValueError):
        cards.source_from("words", words="  ")


def test_set_art_and_sources(tmp_path):
    a = cards.new_card(cards.CardInfo("Own Art"), cards.Source("words", "", "x", "x"))
    pic = tmp_path / "p.png"
    Image.fromarray(np.full((90, 120, 3), 80, np.uint8)).save(pic)
    cards.set_art(a, pic)
    w, h = cards._art_window(cards._info_of(a))
    assert Image.open(a.folder / "art.png").size == (w, h)
    with pytest.raises(ValueError):
        cards.source_from("item", tmp_path)
    with pytest.raises(ValueError):
        cards.source_from("monster", "x")
    prop = assets.new_asset("prop", "Barrel", prompt="a barrel")
    Image.fromarray(np.full((64, 64, 3), 255, np.uint8)).save(prop.folder / "thumb.png")
    src = cards.source_from("prop", prop)
    assert cards.default_info(src).type == "Relic" and src.prompt == "a barrel"


# --------------------------------------------------------------------------- exports

def _two_cards():
    a = cards.new_card(cards.CardInfo("Knight", 3, "Unit", "Guard.", 2, 4, rarity="rare"),
                       cards.Source("words", "", "knight", "a knight"))
    b = cards.new_card(cards.CardInfo("Fireball", 4, "Spell", "Deal 6.", power=6, faction="fire",
                                      size="large"),
                       cards.Source("words", "", "fireball", "a fireball"))
    return a, b


def test_export_godot(tmp_path):
    a, b = _two_cards()
    out = tmp_path / "godot"
    cards.export_cards([a, b.folder], out, "godot")
    for rel in ("Card.tscn", "Card.gd", "README.md", "cards.json", f"cards/{a.folder.name}/card.json",
                f"cards/{b.folder.name}/frame.png", f"cards/{a.folder.name}/ui/panel.tres",
                "fonts/font_regular_1x.fnt", "fonts/font_number_2x.fnt", "fonts/font_number_4x.fnt"):
        assert (out / rel).exists(), rel
    tscn = (out / "Card.tscn").read_text()
    assert 'path="res://cards/Card.gd"' in tscn
    for node in ("Art", "Frame", "Name", "Type", "Text", "Cost", "Attack", "Health"):
        assert f'[node name="{node}"' in tscn
    assert tscn.count("[ext_resource") == int(tscn.split("load_steps=")[1].split()[0]) - 1
    gd = (out / "Card.gd").read_text()
    assert "func set_card" in gd and "res://cards/fonts" in gd
    index = json.loads((out / "cards.json").read_text())
    assert len(index["cards"]) == 2
    with pytest.raises(ValueError):
        cards.export_cards([a], out, "gamemaker")


def test_export_unity(tmp_path):
    a, _ = _two_cards()
    out = tmp_path / "unity"
    cards.export_cards([a], out, "unity")
    cs = (out / "CardView.cs").read_text()
    assert "class CardView : MonoBehaviour" in cs and "JsonUtility.FromJson<CardData>" in cs
    meta = (out / "cards" / a.folder.name / "card.png.meta").read_text()
    assert "filterMode: 0" in meta and "guid:" in meta
    ui = (out / "cards" / a.folder.name / "ui" / "border.png.meta").read_text()
    m = json.loads((out / "cards" / a.folder.name / "ui" / "border.json").read_text())["patch_margin"]
    assert f"spriteBorder: {{x: {m['left']}, y: {m['bottom']}" in ui
    data = json.loads((out / "cards" / a.folder.name / "card.json").read_text())
    assert data["layout"]["slots"]["text"]["w"] > 0


GODOT = os.environ.get("GODOT_BIN") or shutil.which("godot")


@pytest.mark.skipif(not GODOT, reason="set GODOT_BIN to load the card scene in Godot")
def test_card_scene_in_godot(tmp_path):
    proj = tmp_path / "game"
    proj.mkdir()
    (proj / "project.godot").write_text('config_version=5\n[application]\nconfig/name="cards"\n')
    a, b = _two_cards()
    cards.export_cards([a, b], proj / "cards", "godot")
    (proj / "check.gd").write_text(f'''extends SceneTree
func _init():
	var card = load("res://cards/Card.tscn").instantiate()
	root.add_child(card)
	card.load_card("res://cards/cards/{b.folder.name}/card.json")
	await process_frame
	var ok = card.get_node("Name").text == "Fireball" and card.get_node("Power").visible \\
		and not card.get_node("Attack").visible and card.get_node("Art").texture != null \\
		and card.size == Vector2(252, 352) and card.get_node("Name").get_theme_font("font") is FontFile
	print("CARD_OK" if ok else "CARD_BAD")
	quit()
''')
    subprocess.run([GODOT, "--headless", "--path", str(proj), "--import"], capture_output=True,
                   timeout=300)
    r = subprocess.run([GODOT, "--headless", "--path", str(proj), "-s", "res://check.gd"],
                       capture_output=True, text=True, timeout=300)
    assert "CARD_OK" in r.stdout, r.stdout + r.stderr


# --------------------------------------------------------------------------- UI

def test_cards_tab(comfy, tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.cards_tab import CardsTab
    fake, client = comfy
    hero()
    tab = CardsTab()
    tab.sync = True
    tab.client = client
    got = []
    tab.changed.connect(lambda: got.append(1))
    assert tab.view.rgba is not None and tab.view.rgba.shape[:2] == (176, 126)
    tab.select_source("character", "Hero Knight")
    assert tab.name.text() == "Hero Knight" and tab.attack.value() > 0
    tab.text.setPlainText("Guard.")
    tab.rarity.setCurrentText("epic")
    before = tab.view.rgba.copy()
    tab.size_.setCurrentIndex(1)
    assert tab.view.rgba.shape[:2] == (264, 189) and not np.array_equal(before.shape, tab.view.rgba.shape)
    tab.make_art()
    assert len(fake.prompts) == 6 and tab.asset is not None and got
    assert tab.variations.count() == 3 and tab.saved.count() == 1
    tab.pick_art(3)
    assert tab.asset.extra["art_choice"] == 3
    tab.health.setValue(-1)
    tab.attack.setValue(-1)
    tab.power.setValue(5)
    tab.save()
    data = json.loads((tab.asset.folder / "card.json").read_text())
    assert data["power"] == 5 and "attack" not in data and data["rarity"] == "epic"
    tab.new_card()
    assert tab.asset is None
    tab.select_source("words", words="a fireball")
    assert tab.type_.text() == "Spell"
    tab.save()
    assert tab.saved.count() == 2
    tab.open_card(cards.list_cards()[-1])
    assert tab.name.text() == "Hero Knight" and tab.variations.count() == 3
    out = tab.export("godot", str(tmp_path))
    assert (out / "Card.tscn").exists()
    out = tab.export("unity", str(tmp_path))
    assert (out / "CardView.cs").exists()
