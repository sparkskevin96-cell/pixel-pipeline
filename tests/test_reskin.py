import json

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline import library, reskin as R
from pixel_pipeline.teams import _lch
from test_exports import DARK_RED, FRAMES, GREY, RED, make_character


@pytest.fixture
def knight(tmp_path):
    return make_character(tmp_path)


def _px(path):
    return np.asarray(Image.open(path).convert("RGBA"))


def _hue(rgb):
    return float(_lch(np.array([rgb], np.uint8))[0, 2])


def test_red_cape_turns_blue(knight):
    blue = R.reskin(knight, "Blue Knight", target_colour="blue")
    a, b = _px(knight.sheet), _px(blue.sheet)
    assert a.shape == b.shape and (a[..., 3] == b[..., 3]).all()      # alpha exact
    cape = (a[..., :3] == RED).all(-1) & (a[..., 3] > 0)
    grey = (a[..., :3] == GREY).all(-1) & (a[..., 3] > 0)
    new_cape = np.unique(b[cape][:, :3], axis=0)
    assert len(new_cape) == 1 and 230 < _hue(new_cape[0]) < 320        # blue now
    assert (b[grey][:, :3] == GREY).all()                             # body untouched
    shade = (a[..., :3] == DARK_RED).all(-1) & (a[..., 3] > 0)
    # the shade stays darker than the cape
    assert _lch(b[shade][:1, :3])[0, 0] < _lch(new_cape)[0, 0]
    # normal map copied unchanged, files there, library + asset metadata
    assert (_px(blue.folder / "sheet_n.png") == _px(knight.folder / "sheet_n.png")).all()
    for f in ("sheet.json", "sheet.tres", "sheet_facing.gd", "thumb.png", "sheet.aseprite"):
        assert (blue.folder / f).exists(), f
    assert blue.folder != knight.folder and library.get("Blue Knight").folder == blue.folder
    meta = json.loads((blue.folder / "asset.json").read_text())
    assert meta["extra"]["reskin_of"]["name"] == "Knight" and meta["extra"]["body"] == "person"
    assert json.loads((blue.folder / "character.json").read_text())["name"] == "Blue Knight"
    blue.save()                                                       # extra survives a save
    assert "reskin_of" in json.loads((blue.folder / "asset.json").read_text())["extra"]


def test_aseprite_rewritten_from_recoloured_sheet(knight):
    from test_aseprite import read_ase
    blue = R.reskin(knight, "Blue", target_colour="#2f5fd8")
    ase = read_ase(blue.folder / "sheet.aseprite")
    assert len(ase["frames"]) == 2 * 4 * FRAMES and len(ase["tags"]) == 2 * 4
    pal = {tuple(p[:3]) for p in ase["palette"]}
    assert tuple(RED) not in pal                                      # the old red is gone


def test_hue_shift_saturation_and_palette_map(knight):
    shifted = R.reskin(knight, "Shifted", hue=180)
    a, b = _px(knight.sheet), _px(shifted.sheet)
    cape = (a[..., :3] == RED).all(-1) & (a[..., 3] > 0)
    gap = abs((_hue(b[cape][0, :3]) - _hue(RED) + 180) % 360 - 180)
    assert gap > 120                                                  # (sRGB gamut clips a little)
    grey = R.reskin(knight, "Grey", saturation=0.0)
    g = _px(grey.sheet)
    assert _lch(g[cape][:1, :3])[0, 1] < 3                            # no colour left
    mapped = R.reskin(knight, "Mapped", palette_map={"#c81e1e": "#00ff00"})
    m = _px(mapped.sheet)
    assert (m[cape][:, :3] == [0, 255, 0]).all()
    assert (m[~cape] == a[~cape]).all()                               # nothing else moved


def test_preview_and_team_variants(knight):
    pv = R.preview(knight, target_colour="green")
    assert pv.shape == _px(knight.thumb).shape
    teams = R.make_team_variants(knight, ["blue", "red", "#30b8d0"])
    assert [t.name for t in teams] == ["Knight Blue", "Knight Red", "Knight #30b8d0"]
    assert len(library.list_characters()) == 4
    with pytest.raises(ValueError):
        R.colour_hex("not-a-colour")
