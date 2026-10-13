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
    # neutral from the picture, expressions from the neutral, all one seed, no LoRA; no face
    # detector here, so no face found: whole-picture expressions
    ks = [next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"] for wf in fake.prompts]
    from pixel_pipeline.portraits import NEUTRAL_DENOISE, WHOLE_DENOISE
    assert [k["denoise"] for k in ks] == [NEUTRAL_DENOISE, WHOLE_DENOISE, WHOLE_DENOISE]
    assert asset.extra["face_found"] is False and asset.extra["face_only"] is False
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


FAKE_FACE = {"box": (380, 330, 264, 300), "eyes": [(430, 410), (594, 410)], "nose": (512, 480),
             "mouth": [(460, 560), (564, 560)], "score": 0.9}


def test_expressions_inpaint_only_the_face(comfy, tmp_path, monkeypatch):
    fake, client = comfy
    monkeypatch.setattr(portraits, "face_landmarks", lambda rgb, threshold=0.5: FAKE_FACE)
    ch = hero(tmp_path)
    asset = portraits.make_portraits(ch, ("happy", "sad"), client=client, painted_size=128, size=48)
    neutral_wf, *expr_wfs = fake.prompts
    assert not any(n["class_type"] == "SetLatentNoiseMask" for n in neutral_wf.values())
    for wf in expr_wfs:
        k = next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]
        noise = wf[k["latent_image"][0]]
        assert noise["class_type"] == "SetLatentNoiseMask" and k["denoise"] in (0.9, 0.95)
        assert wf[noise["inputs"]["mask"][0]]["class_type"] == "LoadImageMask"
        neg = next(n["inputs"]["text"] for nid, n in wf.items()
                   if nid == k["negative"][0])
        assert "neutral expression" in neg
        pos = next(n["inputs"]["text"] for nid, n in wf.items() if nid == k["positive"][0])
        assert pos.startswith("close-up face of a person, (")     # no costume words
    assert [next(n for n in wf.values() if n["class_type"] == "KSampler")["inputs"]["denoise"]
            for wf in expr_wfs] == [portraits.EXPRESSION_DENOISE, portraits.EXPRESSION_DENOISE_BY["sad"]]
    up = fake.uploads[next(n["inputs"]["image"] for n in expr_wfs[0].values()
                           if n["class_type"] == "LoadImage")]
    assert up.size == (1024, 1024)                    # the face crop, enlarged to fill the frame
    # hair/clothes/background outside the mask are exactly the neutral painting's pixels
    mask = np.asarray(Image.open(asset.folder / "face_mask.png"))
    raw_n = np.asarray(Image.open(asset.folder / "raw_neutral.png"))
    raw_h = np.asarray(Image.open(asset.folder / "raw_happy.png"))
    assert mask.shape == raw_n.shape[:2] and (mask == 255).any() and (mask == 0).mean() > 0.6
    assert (raw_n[mask == 0] == raw_h[mask == 0]).all()
    assert asset.extra["face_only"] is True and asset.extra["face_found"] is True


def test_person_words_and_crop_box():
    assert portraits.person_words("a young swordswoman in green leather armor") == "a young woman"
    assert portraits.person_words("an old wizard") == "an old man"
    assert portraits.person_words("Knight") == "a person"
    m = np.zeros((100, 100), np.float32)
    m[10:30, 80:95] = 1
    x0, y0, x1, y1 = portraits.face_crop_box(m)
    assert x1 - x0 == y1 - y0 and x1 <= 100 and x0 <= 80 and y0 <= 10 and y1 >= 30


def test_whole_picture_mode_and_replace(comfy, tmp_path):
    fake, client = comfy
    ch = hero(tmp_path)
    first = portraits.make_portraits(ch, ("angry",), client=client, painted_size=64, size=32)
    (first.folder / "raw_surprised.png").write_bytes(b"old")
    again = portraits.make_portraits(ch, ("smug",), client=client, painted_size=64, size=32,
                                     face_only=False, replace=first)
    assert again.folder == first.folder and len(list_assets("portrait")) == 1
    assert not (first.folder / "raw_surprised.png").exists()
    assert again.extra["expressions"] == ["neutral", "smug"]
    k = next(n for n in fake.prompts[-1].values() if n["class_type"] == "KSampler")["inputs"]
    assert k["denoise"] == portraits.WHOLE_DENOISE
    assert not any(n["class_type"] == "SetLatentNoiseMask" for n in fake.prompts[-1].values())
    with pytest.raises(ValueError):
        portraits.make_portraits(ch, client=client, replace=tmp_path)


def test_face_mask_from_landmarks():
    img = np.full((400, 400, 3), 255, np.uint8)
    face = {"box": (150, 120, 100, 130), "eyes": [(175, 170), (225, 170)], "nose": (200, 195),
            "mouth": [(182, 222), (218, 222)]}
    m = portraits.face_mask(img, face)
    assert m.dtype == np.float32 and m[195, 200] == 1.0           # nose: fully repainted
    assert m[150, 200] > 0.5 and m[245, 200] > 0.5                # brows and chin inside
    assert m[60, 200] == 0 and m[195, 300] == 0 and m[380, 200] == 0  # hair, ear, chest kept
    assert 0 < m[195, 245] < 1                                    # feathered edge
    # tilted head: the ellipse turns with the eye line
    tilt = dict(face, eyes=[(175, 160), (225, 180)], mouth=[(175, 212), (211, 226)])
    cx, cy, rx, ry, ang = portraits.face_ellipse(img, tilt)
    assert ang > 0.1 and ry > rx                                 # turned clockwise
    # no face: the bust's top
    bust = img.copy()
    bust[80:400, 120:280] = (200, 160, 130)
    cx, cy, *_ = portraits.face_ellipse(bust, None)
    assert abs(cx - 200) < 2 and 80 < cy < 260


def test_generate_images_inpaint(comfy):
    from pixel_pipeline.generate import composite_masked, generate_images, inpaint_workflow
    fake, client = comfy
    init = Image.new("RGB", (512, 512), (10, 200, 30))
    mask = np.zeros((512, 512), np.float32)
    mask[100:300, 150:350] = 1
    (img, _), = generate_images("a face", "", client=client, seed=3, init_image=init,
                                denoise=0.8, mask=mask, lora="none", width=512, height=512)
    a = np.asarray(img)
    assert (a[mask == 0] == (10, 200, 30)).all() and not (a[mask == 1] == (10, 200, 30)).all()
    names = [n["inputs"]["image"] for n in fake.prompts[0].values()
             if n["class_type"] == "LoadImageMask"]
    up = np.asarray(fake.uploads[names[0]].convert("L"))
    assert up[200, 250] == 255 and up[0, 0] == 0
    (img, _), = generate_images("a face", "", client=client, seed=3, init_image=init,
                                denoise=1.0, mask=mask, grow_mask_by=6, lora="none",
                                width=512, height=512)
    enc = [n for n in fake.prompts[1].values() if n["class_type"] == "VAEEncodeForInpaint"]
    assert enc and enc[0]["inputs"]["grow_mask_by"] == 6
    with pytest.raises(ValueError):
        generate_images("x", client=client, mask=mask)
    with pytest.raises(ValueError, match="mask is"):
        generate_images("x", client=client, init_image=init, mask=np.ones((8, 8), np.float32))
    wf = inpaint_workflow("p", "n", "i.png", "m.png", seed=1, lora=None)
    assert {n["class_type"] for n in wf.values()} >= {"LoadImageMask", "SetLatentNoiseMask"}
    half = composite_masked(Image.new("RGB", (4, 4), (0, 0, 0)), Image.new("RGB", (4, 4), (200,) * 3),
                            np.full((4, 4), 0.5, np.float32))
    assert np.asarray(half)[0, 0, 0] == 100


def test_refinish_and_white_key(comfy, tmp_path, monkeypatch):
    fake, client = comfy
    ch = hero(tmp_path)
    asset = portraits.make_portraits(ch, ("happy",), client=client, painted_size=64, size=32)
    (asset.folder / "portrait_happy_pixel.png").unlink()
    n = len(fake.prompts)
    portraits.refinish_portraits(asset)
    assert (asset.folder / "portrait_happy_pixel.png").exists() and len(fake.prompts) == n  # no GPU
    # a model that drops a soft red hood: the white key keeps what clearly isn't background
    from pixel_pipeline import pipeline
    monkeypatch.setattr(pipeline, "remove_background", lambda rgb, model="isnet-general-use":
                        np.zeros(rgb.shape[:2], np.uint8))
    img = Image.new("RGB", (64, 64), (255, 255, 255))
    img.paste((200, 30, 30), (16, 8, 48, 64))
    out = portraits._cut_out(img)
    assert out[30, 30, 3] == 255 and out[2, 2, 3] == 0
