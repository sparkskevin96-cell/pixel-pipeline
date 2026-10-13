"""Portraits from a library character, against the fake ComfyUI (no GPU)."""
import json

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import library, portraits, teams
from pixel_pipeline.assets import list_assets
from test_generate import comfy, white_key  # noqa: F401  (pytest fixtures)


@pytest.fixture(autouse=True)
def no_face_model(monkeypatch):
    monkeypatch.setattr(teams, "_face_model_path", None)


def hero(tmp_path, picture=True):
    """A library character with a front picture on white (head, body) and no sheet."""
    ch = library.Character("Hero Knight", "person", "normal", "", 1.0,
                           library.root() / "hero-knight")
    ch.save()
    if picture:
        img = np.full((320, 160, 3), 255, np.uint8)
        img[20:90, 50:110] = (235, 190, 150)      # head
        img[90:320, 20:140] = (60, 70, 160)       # body
        Image.fromarray(img).save(ch.picture)
    return ch


def test_make_portraits_painted(comfy, tmp_path):
    fake, client = comfy
    ch = hero(tmp_path)
    said = []
    asset = portraits.make_portraits(ch, ("happy", "angry"), client=client, progress=said.append,
                                     painted_size=128, size=48)
    assert asset.kind == "portrait"
    assert asset.extra["expressions"] == ["neutral", "happy", "angry"]
    assert asset.extra["character"] == "Hero Knight"
    assert asset.extra["character_folder"] == str(ch.folder)
    # neutral from the picture, expressions from the neutral, all one seed, no LoRA
    ks = [next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"] for wf in fake.prompts]
    from pixel_pipeline.portraits import EXPRESSION_DENOISE, NEUTRAL_DENOISE
    assert [k["denoise"] for k in ks] == [NEUTRAL_DENOISE, EXPRESSION_DENOISE, EXPRESSION_DENOISE]
    assert len({k["seed"] for k in ks}) == 1
    assert not any(n["class_type"] == "LoraLoader" for wf in fake.prompts for n in wf.values())
    texts = [n["inputs"]["text"] for wf in fake.prompts for n in wf.values()
             if n["class_type"] == "CLIPTextEncode"]
    assert any("Hero Knight, head and shoulders portrait" in t and "angry" in t for t in texts)
    for e in ("neutral", "happy", "angry"):
        p = np.asarray(Image.open(asset.folder / f"portrait_{e}.png"))
        assert p.shape == (128, 128, 4) and (p[..., 3] == 0).any()      # background cut out
        px = np.asarray(Image.open(asset.folder / f"portrait_{e}_pixel.png"))
        assert px.shape == (48, 48, 4)
    meta = json.loads((asset.folder / "sheet.json").read_text())
    assert [f["filename"] for f in meta["frames"]] == ["neutral", "happy", "angry"]
    assert meta["meta"]["style"] == "painted"
    assert Image.open(asset.folder / "sheet_pixel.png").size == (144, 48)
    for f in ("thumb.png", "frame.png", "frame.tres", "frame.json", "preview.png", "raw_neutral.png"):
        assert (asset.folder / f).exists(), f
    # the pixel versions share one palette
    cols = [set(map(tuple, np.asarray(Image.open(asset.folder / f"portrait_{e}_pixel.png"))
                    .reshape(-1, 4)[:, :3].tolist())) for e in ("neutral", "happy")]
    assert len(cols[0] | cols[1]) <= 24 + 2
    assert [a.name for a in list_assets("portrait")] == ["Hero Knight portraits"]
    assert portraits.load_portraits(asset).keys() == {"neutral", "happy", "angry"}


def test_pixel_style_by_name_and_sprite_fallback(comfy, tmp_path):
    fake, client = comfy
    ch = hero(tmp_path, picture=False)
    sheet = np.zeros((64, 96, 4), np.uint8)
    sheet[4:20, 40:56] = (230, 190, 150, 255)
    sheet[20:62, 34:62] = (150, 40, 40, 255)
    Image.fromarray(sheet).save(ch.sheet)
    ch.sheet.with_suffix(".json").write_text(json.dumps({"frames": [
        {"filename": "idle_S 0", "frame": {"x": 0, "y": 0, "w": 96, "h": 64}}]}))
    asset = portraits.make_portraits("Hero Knight", ["neutral"], style="pixel", client=client,
                                     painted_size=64, size=32)
    ks = next(n for n in fake.prompts[0].values() if n["class_type"] == "KSampler")["inputs"]
    assert ks["denoise"] == portraits.SPRITE_DENOISE and asset.extra["source"] == "sprite"
    assert Image.open(asset.folder / "sheet.png").size == (32, 32)
    assert asset.extra["style"] == "pixel"


def test_errors(tmp_path):
    with pytest.raises(ValueError):
        portraits.make_portraits("nobody")
    ch = hero(tmp_path, picture=False)
    with pytest.raises(ValueError, match="no picture"):
        portraits.make_portraits(ch, client=object())
    with pytest.raises(ValueError):
        portraits.make_portraits(ch, style="oil")


def test_portrait_frame(tmp_path):
    from pixel_pipeline.assets import Style
    files = portraits.portrait_frame(tmp_path, 8, style=Style(palette="pico-8"))
    img = np.asarray(Image.open(files[0]))
    assert img.shape == (24, 24, 4)
    assert img[12, 12, 3] == 0 and img[0, 12, 3] == 255            # open middle, solid edge
    from pixel_pipeline.palettes import BUILTIN, hex_to_rgb
    pico = {tuple(c) for c in hex_to_rgb(BUILTIN["pico-8"]).tolist()}
    assert {tuple(c) for c in img[img[..., 3] > 0][:, :3].tolist()} <= pico
    tres = files[2].read_text()
    assert "StyleBoxTexture" in tres and "texture_margin_left = 8.0" in tres
    big = portraits.nine_slice(img, 50, 30, 8)
    assert big.shape == (30, 50, 4) and (big[:8, :8] == img[:8, :8]).all()
    assert (big[-8:, -8:] == img[-8:, -8:]).all()


def test_portraits_tab(comfy, tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.portraits_tab import PortraitsTab
    fake, client = comfy
    hero(tmp_path)
    tab = PortraitsTab()
    tab.sync = True
    tab.client = client
    assert tab.chars.count() == 1 and tab.current_character().name == "Hero Knight"
    for e, c in tab.checks.items():
        c.setChecked(e in ("neutral", "sad"))
    tab.size_.setValue(32)
    got = []
    tab.changed.connect(lambda: got.append(1))
    tab.make()
    assert got == [1] and tab.saved.count() == 1
    assert tab.asset.extra["expressions"] == ["neutral", "sad"]
    assert tab.view.rgba is not None and tab.view.rgba.shape[1] > tab.view.rgba.shape[0]  # side by side
    tab.style.setCurrentIndex(1)                               # pixel
    assert tab.view.rgba.shape[0] == 32
