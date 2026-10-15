"""Single-subject check (subjectcheck.py): sheets, several things, scenes, cropping; redraws and
best-first ordering in generate_single."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from pixel_pipeline import generate, subjectcheck as sc


def canvas(bg=(255, 255, 255), size=512):
    img = Image.new("RGB", (size, size), bg)
    return img, ImageDraw.Draw(img)


def one(size=512, colour=(150, 90, 40)):
    img, d = canvas(size=size)
    d.ellipse([size * .3, size * .15, size * .7, size * .9], fill=colour)
    d.rectangle([size * .42, size * .3, size * .48, size * .36], fill=(10, 10, 10))
    return img


def sheet(n=3):
    img, d = canvas()
    for r in range(n):
        for c in range(n):
            x, y = 40 + c * 150, 40 + r * 150
            d.rectangle([x, y, x + 90, y + 100], fill=(120, 80, 40))
    return img


def two():
    img, d = canvas()
    d.ellipse([40, 100, 230, 420], fill=(150, 90, 40))
    d.ellipse([290, 120, 470, 430], fill=(60, 90, 150))
    return img


def scene():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    img = Image.fromarray(a)
    ImageDraw.Draw(img).ellipse([150, 100, 360, 450], fill=(150, 90, 40))
    return img


def cropped():
    img, d = canvas()
    d.rectangle([-50, 200, 600, 700], fill=(150, 90, 40))
    return img


def test_one_subject_passes():
    v = sc.assess(one())
    assert v.ok and v.pieces == 1 and v.reason() == "one subject"
    # loose bits right next to it belong to it (a tree's leaves, a staff's tip)
    img = one()
    ImageDraw.Draw(img).ellipse([362, 250, 372, 262], fill=(150, 90, 40))
    assert sc.assess(img).ok
    # a coloured but plain background is fine too
    img, d = canvas((230, 210, 170))
    d.ellipse([150, 80, 360, 460], fill=(120, 60, 30))
    assert sc.assess(img).ok
    # RGBA (transparent) pictures are judged on white
    rgba = np.zeros((256, 256, 4), np.uint8)
    rgba[60:200, 90:170] = (200, 40, 40, 255)
    assert sc.assess(rgba).ok


@pytest.mark.parametrize("make, word", [(sheet, "sheet"), (two, "separate"), (scene, "scene"),
                                        (cropped, "cut off")])
def test_bad_pictures_fail(make, word):
    v = sc.assess(make())
    assert not v.ok and word in v.reason()
    assert v.score < sc.assess(one()).score


def test_sheet_flag():
    assert sc.assess(sheet(3)).sheet
    assert not sc.assess(two()).sheet


def test_rank_orders_best_first():
    pics = [sheet(), one(), two(), scene()]
    order = sc.rank(pics)
    assert order[0] == 1 and order[-1] in (0, 3)


def fake_generator(bad_seeds):
    calls = []

    def fake(positive, negative="", *, seed=None, count=1, **kw):
        calls.append((seed, count, kw))
        out = []
        for i in range(count):
            s = seed + i
            out.append((sheet() if s in bad_seeds else one(colour=(s % 200, 90, 40)), s))
        return out
    return fake, calls


def test_generate_single_redraws_failures_and_puts_best_first(monkeypatch):
    fake, calls = fake_generator(bad_seeds={10, 11})
    monkeypatch.setattr(generate, "generate_images", fake)
    said = []
    picks = sc.generate_single("barrel", "neg", count=3, seed=10, aspect=1.0, say=said.append)
    assert [p.verdict.ok for p in picks] == [True, True, True]
    assert sorted(p.seed for p in picks) == [12, 13, 14]
    assert calls[0][:2] == (10, 3) and calls[0][2]["aspect"] == 1.0   # first round: all three
    assert calls[1][:2] == (13, 2)                        # then only the two that failed
    assert said and "not one clean subject" in said[0]


def test_generate_single_budget_and_off_switch(monkeypatch):
    fake, calls = fake_generator(bad_seeds=set(range(100)))
    monkeypatch.setattr(generate, "generate_images", fake)
    picks = sc.generate_single("barrel", count=1, seed=1, redraws=2)
    assert len(calls) == 3 and len(picks) == 1 and not picks[0].verdict.ok   # gave up: best of 3
    calls.clear()
    monkeypatch.setenv(sc.CHECK_ENV, "0")
    picks = sc.generate_single("barrel", count=2, seed=1)
    assert len(calls) == 1 and [p.seed for p in picks] == [1, 2]


def test_generate_single_extra_check(monkeypatch):
    fake, calls = fake_generator(bad_seeds=set())
    monkeypatch.setattr(generate, "generate_images", fake)
    seen = []

    def check(img):
        seen.append(img)
        return "2 separate things after the cut-out" if len(seen) == 1 else None
    picks = sc.generate_single("knight", count=1, seed=5, check=check)
    assert picks[0].seed == 6 and picks[0].verdict.ok and len(calls) == 2


def test_clean_style_words_drop_broken_parts():
    from pixel_pipeline.assets import NEO16_PROMPT, _clean_terms
    out = _clean_terms(NEO16_PROMPT + ", HD-2D JRPG, detailed sprites")
    assert "era craft" not in out and "modern" not in out and "detailed" not in out
    assert "rich wide colour palette" in out and "HD-2D JRPG" in out
    assert _clean_terms("16-bit SNES JRPG, warm palette") == "JRPG, warm palette"



def test_start_blob_shape_colour_and_plumbing(monkeypatch):
    assert sc.shape_for("rusty iron sword") == "weapon" and sc.shape_for("golden key") == "weapon"
    assert sc.shape_for("tavern with a thatched roof") == "building"
    assert sc.shape_for("wooden barrel") == "object" and sc.shape_for("goblin", "person") == "person"
    assert sc.init_colour("red apple") == (185, 55, 45) and sc.init_colour("a thing") is None
    blob = np.asarray(sc.single_init("weapon", size=256, colour=(185, 55, 45)))
    assert tuple(blob[0, 0]) == (255, 255, 255)
    assert np.abs(blob[128, 128].astype(int) - (185, 55, 45)).max() <= 2
    assert blob[128, 20].tolist() == [255, 255, 255]                 # thin: sides stay white
    fake, calls = fake_generator(bad_seeds=set())
    monkeypatch.setattr(generate, "generate_images", fake)
    sc.generate_single("apple", count=1, seed=1, init="object", init_denoise=0.93,
                       init_words="red apple")
    kw = calls[0][2]
    centre = np.asarray(kw["init_image"])[512, 512].astype(int)
    assert kw["denoise"] == 0.93 and np.abs(centre - (185, 55, 45)).max() <= 2
    calls.clear()
    monkeypatch.setenv(sc.CHECK_ENV, "0")                            # off = plain txt2img
    sc.generate_single("apple", count=1, seed=1, init="object")
    assert "init_image" not in calls[0][2]
