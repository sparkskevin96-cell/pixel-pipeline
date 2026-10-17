"""Small PLAYABLE test games, one per style, built from a project's assets through the app's own
paths (make_hd2d for the HD-2D board, gameexport.export_to_game + manifest for the 2D games), so
whatever looks or plays wrong in them is the app's real issue.

    build_playtest("tactics", "C:/.../playtest/tactics")      # HD-2D tactics, real 3D board
    build_playtest("action", ...)                              # ALttP-style action on the board
    build_playtest("platformer", ...)                          # side-scroller (2 directions)
    build_playtest("iso", ...)                                 # isometric 2D tactics
    build_playtest("cards", ...)                               # a small deck-builder
    build_all("C:/.../playtest")                               # all five + launchers

Every game gets the godot_runtime helpers: PPDebug (F1 controls hint, F3 debug overlay with FPS
and game state + collision shapes, F12 report = screenshot + state into <game>/reports/, F5
restart), a README.txt with the controls, and Play_<style>.bat / Edit_<style>.bat.

Assets come from `Sources` (characters by role, the tilesets' project, a UI kit, cards, items,
props). `Sources.from_project(p)` takes a project's own; the defaults are the sample projects
("QA 2026-10-15", "Neo16 test") when they exist. Sources are only READ: what the builders make
on the way (re-skins, effects, cards, side-scroller re-renders) goes into a staging home
(<out>/_studio, PIXEL_HOME-style projects) that the export reads from.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import godot_runtime

STYLES = ("tactics", "action", "platformer", "iso", "cards")
TITLES = {"tactics": "HD-2D tactics test", "action": "Top-down action test",
          "platformer": "Side-scroller test", "iso": "Isometric tactics test",
          "cards": "Deck-builder test"}
SAMPLE_PROJECTS = Path.home() / "Pixel Pipeline" / "Projects"

CONTROLS = {
    "tactics": ("Click a blue unit -> click a purple cell (or the unit to stay) -> click a red enemy to "
                "attack / cast   Space: wait / end turn   Right click / Esc: cancel\n"
                "Q/E: turn camera   WASD: pan   Wheel: zoom   L: light   B: pixel bands   Esc (nothing "
                "selected): looks menu"),
    "action": ("WASD / arrows: walk (8 ways)   Space / J: sword   K: shoot   R: restart   Esc: quit\n"
               "Cut bushes and pots, pick up hearts and coins, beat the wolves and the bandit"),
    "platformer": ("A/D or arrows: run   Shift: sprint   Space / W / Up: jump (hold = higher)   J / K: shoot   "
                   "R: restart   Esc: quit\nStomp or shoot the wolves, grab the coins, reach the flag"),
    "iso": ("Click a blue unit -> click a purple cell (or the unit to stay) -> click a red enemy   "
            "Space/Tab: wait / end turn   Right click: cancel   Wheel: zoom   WASD/arrows: pan"),
    "cards": ("Click (or drag) a card onto the enemy to play it; skills/powers: click or drag up   "
              "E / End turn button: end turn   V: deck   Esc: quit\n"
              "Map: fight -> fight or rest -> boss; pick 1 of 3 cards after each win"),
}


def _sample(*parts) -> Path:
    return SAMPLE_PROJECTS.joinpath(*parts)


@dataclass
class Sources:
    """Where a playtest takes its art from (read only)."""
    characters: dict[str, Path] = field(default_factory=dict)   # role -> library character folder
    tiles_project: Path | None = None        # a project whose tilegen sets texture the HD-2D board
    ui_kit: Path | None = None               # a uikit.py kit folder
    cards: list[Path] = field(default_factory=list)              # card asset folders
    items: list[Path] = field(default_factory=list)              # item folders (card art)
    props: list[Path] = field(default_factory=list)              # prop folders (card art)

    @classmethod
    def samples(cls) -> "Sources":
        qa, neo = _sample("QA 2026-10-15"), _sample("Neo16 test")
        chars = {"knight": neo / "characters/knight-v3", "mage": neo / "characters/mage-v3",
                 "wolf": neo / "characters/wolf-v3", "archer": qa / "characters/elf-archer",
                 "dwarf": qa / "characters/dwarf-front-first", "small_wolf": qa / "characters/grey-wolf"}
        return cls({k: v for k, v in chars.items() if (v / "sheet.png").exists()},
                   qa if (qa / "tilesets").exists() else None,
                   qa / "ui/wood-ui-kit" if (qa / "ui/wood-ui-kit/kit.json").exists() else None,
                   sorted(p for p in (qa / "cards").glob("*") if (p / "card.json").exists()),
                   sorted(p for p in (qa / "items").glob("*") if (p / "asset.json").exists()),
                   sorted(p for p in (qa / "props").glob("*") if (p / "asset.json").exists()))

    @classmethod
    def from_project(cls, project=None) -> "Sources":
        """A project's own characters (first ones by name for each role), sets, kit, cards."""
        from . import assets
        p = project or assets.current_project()
        folder = Path(p.folder)
        chars = sorted(c for c in (folder / "characters").glob("*") if (c / "sheet.png").exists())
        people = [c for c in chars if json.loads((c / "character.json").read_text(encoding="utf-8")).get("kind") != "animal"]
        animals = [c for c in chars if c not in people]
        roles = {}
        for role, pool, i in (("knight", people, 0), ("mage", people, 1), ("archer", people, 2),
                              ("dwarf", people, 3), ("wolf", animals, 0), ("small_wolf", animals, 1)):
            if pool:
                roles[role] = pool[i % len(pool)]
        kits = sorted(k for k in (folder / "ui").glob("*") if (k / "kit.json").exists())
        return cls(roles, folder if (folder / "tilesets").exists() else None, kits[0] if kits else None,
                   sorted(c for c in (folder / "cards").glob("*") if (c / "card.json").exists()),
                   sorted(c for c in (folder / "items").glob("*") if (c / "asset.json").exists()),
                   sorted(c for c in (folder / "props").glob("*") if (c / "asset.json").exists()))

    def character(self, role: str, *fallbacks: str) -> Path:
        for r in (role,) + fallbacks:
            if r in self.characters:
                return Path(self.characters[r])
        if self.characters:
            return Path(next(iter(self.characters.values())))
        raise ValueError("No characters to build a playtest with (add some to the project's library)")


# --------------------------------------------------------------------------- shared bits

def _godot_bin(godot: str | None = None) -> str | None:
    if godot:
        return godot
    from .gameexport import find_godot
    return find_godot()


def _gui_exe(godot: str) -> str:
    """The windowed exe beside the console one (no console window when playing)."""
    p = Path(godot)
    gui = p.with_name(p.name.replace("_console", ""))
    return str(gui if gui.exists() else p)


def write_launchers(game_dir, style: str, godot: str | None = None, also_in=None) -> list[Path]:
    """Play_<style>.bat (plays) and Edit_<style>.bat (opens the editor) in the game folder (and
    in also_in, e.g. the playtest root). %GODOT_BIN% wins over the exe found now."""
    game_dir = Path(game_dir)
    exe = _gui_exe(_godot_bin(godot) or "godot")
    out = []
    for folder in [game_dir] + ([Path(also_in)] if also_in else []):
        rel = "%~dp0" if folder == game_dir else f"%~dp0{game_dir.name}\\"
        for verb, args in (("Play", ""), ("Edit", " -e")):
            bat = folder / f"{verb}_{style}.bat"
            bat.write_text("@echo off\r\n"
                           f'set "GODOT={exe}"\r\n'
                           'if defined GODOT_BIN set "GODOT=%GODOT_BIN%"\r\n'
                           f'start "" "%GODOT%"{args} --path "{rel}."\r\n', encoding="utf-8")
            out.append(bat)
    return out


def write_readme(game_dir, style: str, about: str, extra: str = "") -> Path:
    p = Path(game_dir) / "README.txt"
    text = (f"{TITLES[style]} (pixel-pipeline playtest)\r\n{'=' * 40}\r\n\r\n{about}\r\n\r\n"
            f"CONTROLS\r\n{CONTROLS[style]}\r\n\r\n"
            "TEST HELPERS (every playtest game)\r\n"
            "  F1   show / hide the controls hint\r\n"
            "  F3   debug overlay: FPS, positions, cell + height, facing, animation + frame,\r\n"
            "       collision shapes / blocked cells\r\n"
            "  F12  report: saves a screenshot + the game state (and the last 3 seconds of it,\r\n"
            "       and your recent keys) to reports\\<style>_<date>_<time>.png / .txt.\r\n"
            "       Open the .txt and type what looks wrong under YOUR NOTE.\r\n"
            "  F5   restart the scene\r\n\r\n"
            f"Play: Play_{style}.bat   Editor: Edit_{style}.bat\r\n"
            f"{extra}")
    p.write_text(text, encoding="utf-8")
    return p


def _playtest_json(game_dir, style: str, extra: dict | None = None) -> Path:
    p = Path(game_dir) / "playtest.json"
    p.write_text(json.dumps({"style": style, "title": TITLES[style],
                             "controls": CONTROLS[style], **(extra or {})}, indent=1), encoding="utf-8")
    return p


def install_runtime(game_dir, names, autoloads: dict[str, str]) -> list[Path]:
    game_dir = Path(game_dir)
    files = godot_runtime.install(game_dir, names)
    godot_runtime.add_autoloads(game_dir / "project.godot", autoloads)
    return files


def _window(game_dir, w: int = 1600, h: int = 900) -> None:
    pg = Path(game_dir) / "project.godot"
    godot_runtime.set_setting(pg, "display", "window/size/window_width_override", str(w))
    godot_runtime.set_setting(pg, "display", "window/size/window_height_override", str(h))


def godot_import(game_dir, godot: str | None = None, timeout: float = 600) -> str:
    """`godot --headless --import` once (the iso look and the 2D games load imported textures)."""
    exe = _godot_bin(godot)
    if not exe:
        return "no Godot found"
    r = subprocess.run([exe, "--headless", "--path", str(game_dir), "--import"], capture_output=True,
                       text=True, timeout=timeout, creationflags=_low_priority())
    return (r.stdout + r.stderr)[-4000:]


def _low_priority() -> int:
    return 0x00004000 if os.name == "nt" else 0          # BELOW_NORMAL_PRIORITY_CLASS


def smoke_test(game_dir, godot: str | None = None, frames: int = 240, scene: str | None = None,
               timeout: float = 300) -> dict:
    """Run the game headless for some frames; report script errors (SCRIPT ERROR / Parse Error)."""
    exe = _godot_bin(godot)
    if not exe:
        return {"ok": False, "log": "no Godot found"}
    cmd = [exe, "--headless", "--path", str(game_dir), "--quit-after", str(frames)]
    if scene:
        cmd.append(scene)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, creationflags=_low_priority())
    log = r.stdout + r.stderr
    bad = [ln for ln in log.splitlines() if any(k in ln for k in ("SCRIPT ERROR", "Parse Error", "ERROR:",
                                                                     "Invalid call", "Invalid access"))]
    return {"ok": not bad, "errors": bad[:40], "log": log[-6000:]}


# --------------------------------------------------------------------------- staging

def _studio(out: Path) -> Path:
    s = out.parent / "_studio" if out.name in STYLES else out / "_studio"
    s.mkdir(parents=True, exist_ok=True)
    return s


def stage_character(src: Path, project_folder: Path, *, keys: bool = False, name: str | None = None) -> Path:
    """A copy of a library character inside a staging project (sheets + metadata; keys/ and the
    rigged model only when keys=True, for CPU re-renders)."""
    dst = project_folder / "characters" / (name or src.name)
    if dst.exists():
        return dst
    skip = ["attempts", "agent_preview", "*.aseprite", "old"] + ([] if keys else ["keys", "*.glb"])
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*skip))
    if keys and (dst / "keys").exists():
        for junk in ("old", "attempts"):
            shutil.rmtree(dst / "keys" / junk, ignore_errors=True)
    return dst


def reskin_sheet(src_folder: Path, dst_folder: Path, colour: str) -> Path:
    """A recoloured copy of a character's sheet (reskin.py), normals/json copied."""
    from .reskin import image_colours, recolour_image, reskin_palette
    dst_folder.mkdir(parents=True, exist_ok=True)
    rgba = np.asarray(Image.open(src_folder / "sheet.png").convert("RGBA"))
    cols, counts = image_colours(rgba)
    new = reskin_palette(cols, counts, target_colour=colour)
    Image.fromarray(recolour_image(rgba, cols, new)).save(dst_folder / "sheet.png")
    for f in ("sheet.json", "sheet_n.png"):
        if (src_folder / f).exists():
            shutil.copyfile(src_folder / f, dst_folder / f)
    return dst_folder / "sheet.png"


FX_SPECS = {
    "fireball": ("spell", "fireball", {}), "frost": ("spell", "frost_nova", {}),
    "ice": ("spell", "ice_shard", {}), "lightning": ("spell", "lightning_bolt", {}),
    "heal": ("spell", "holy_light", {}), "slash": ("spell", "sword_combo", {"hits": 1}),
    "spark": ("spell", "hit_spark", {}), "explosion": ("spell", "big_explosion", {}),
    "levelup": ("spell", "level_up_burst", {}), "poof": ("classic", "smoke", {}),
    "sparkle": ("classic", "sparkle", {}), "dark": ("spell", "dark_magic", {}),
    "poison": ("spell", "poison_cloud", {}), "buff": ("spell", "buff_aura", {}),
}


def effect_sheets(folder: Path, names) -> dict[str, Path]:
    """The app's procedural effects (spellfx / effects, CPU) as sheet.png folders, cached."""
    from .effects import render_effect, write_effect
    from .spellfx import render_spell
    out = {}
    for n in names:
        d = folder / n
        if not (d / "sheet.png").exists():
            kind, k, opts = FX_SPECS[n]
            fx = render_spell(k, seed=3, **opts) if kind == "spell" else render_effect(k, seed=3)
            write_effect(fx, d, "sheet")
        out[n] = d / "sheet.png"
    return out


# --------------------------------------------------------------------------- footsteps

LOCO_MARK = ".playtest_loco"


def locomotion_copies(src: Sources, studio: Path, frames: int = 12, progress=None) -> Sources:
    """Copies of the source characters (studio/projects/<their project>/characters/<name>, style
    copied) with planted-feet walk/run at `frames` frames, bent again from their existing key
    drawings on the CPU (library.rerender_locomotion; 3D-rendered ones re-rendered) + their
    stride measured. The sources are never touched; a copy is made once (LOCO_MARK)."""
    from . import assets, library
    say = progress or (lambda t: None)
    out = {}
    for role, folder in src.characters.items():
        folder = Path(folder)
        base = folder.parent.parent
        proj = staging_project(studio, f"loco-{base.name}", base=base)
        dst = Path(proj.folder) / "characters" / folder.name
        mark = dst / LOCO_MARK
        if not (mark.exists() and mark.read_text(encoding="utf-8") == str(frames)):
            shutil.rmtree(dst, ignore_errors=True)
            stage_character(folder, Path(proj.folder), keys=True)
            t = time.time()
            with assets.using_project(proj):
                ch = library.from_folder(dst)
                try:
                    library.rerender_locomotion(ch, frames, progress=say)
                    note = f"{frames} frames"
                except Exception as e:  # noqa: BLE001 - keep the copy as it was, stride measured
                    note = f"kept ({type(e).__name__}: {e})"
                    library.annotate_stride(ch.sheet, ch.rigged, None)
            say(f"{folder.name}: walk/run {note} in {time.time() - t:.0f} s")
            shutil.rmtree(dst / "keys" / "old", ignore_errors=True)
            mark.write_text(str(frames), encoding="utf-8")
        out[role] = dst
    return Sources(out, src.tiles_project, src.ui_kit, src.cards, src.items, src.props)


def px_scale_of(sheet_png, name: str) -> float:
    """How much bigger than the library's 64 px-person scale a unit's sheet is (motion.px_scale)."""
    from .motion import px_scale
    meta = json.loads(Path(sheet_png).with_suffix(".json").read_text(encoding="utf-8"))
    fr = meta["frames"]
    fh = (fr[0] if isinstance(fr, list) else next(iter(fr.values())))["frame"]["h"]
    return px_scale(fh, "animal" if name in ("wolf", "dog") else "person")


# --------------------------------------------------------------------------- HD-2D board games

def tactics_map():
    """10x8: a river with a plank bridge, a raised east bank reached by stone stairs, torches."""
    from .diorama import Diorama, Unit
    C, R = 10, 8
    h = np.full((R, C), 2, int)
    t = np.full((R, C), "grass", object)
    for r in range(R):                                    # the river (col 4-5), sand on the west bank
        h[r, 4], t[r, 4] = 0, "water"
        if r not in (3, 4):
            h[r, 5], t[r, 5] = 0, "water"
        t[r, 3] = "sand" if r not in (3, 4) else "cobble"
        h[r, 3] = 1 if r not in (3, 4) else 2
    for r in (3, 4):                                      # bridge
        h[r, 4], t[r, 4] = 2, "planks"
        h[r, 5], t[r, 5] = 2, "planks"
    h[0:4, 7:10], t[0:4, 7:10] = 4, "grass"               # east plateau (2 levels up)
    t[1:3, 8:10] = "stone"
    h[4, 7], t[4, 7] = 3, "stone"                         # stairs: 2 -> 3 -> 4
    h[4, 8], t[4, 8] = 3, "stone"
    t[3:5, 0:3] = "cobble"
    t[3:5, 6] = "cobble"
    props = [("tree", 0, 0), ("tree", 1, 0), ("tree", 0, 7), ("tree", 9, 7), ("tree", 6, 0),
             ("rock", 2, 6), ("bush", 1, 6), ("bush", 8, 6), ("torch", 2, 2), ("torch", 2, 5),
             ("torch", 6, 2), ("brazier", 9, 0), ("crate", 6, 6), ("barrel", 7, 6), ("lantern", 6, 5),
             ("fence", 0, 2), ("fence", 1, 2)]
    units = [Unit("knight", 1, 3, 0, "E", hp=34), Unit("mage", 0, 4, 0, "E", hp=22),
             Unit("archer", 1, 5, 0, "E", hp=24),
             Unit("wolf", 6, 4, 1, "W", hp=20), Unit("wolf", 7, 5, 1, "W", hp=20),
             Unit("warlock", 8, 2, 1, "SW", hp=18, attack=12, defence=3, move=3, jump=2, reach=3)]
    return Diorama(C, R, h, t, props, units, name="river crossing", sea=False)


def action_map():
    """18x14: a cliff (north) with stone stairs and a cave, a pond with a jetty, a village corner."""
    from .diorama import Diorama, Unit
    C, R = 18, 14
    h = np.full((R, C), 2, int)
    t = np.full((R, C), "grass", object)
    h[0:3, :], t[0:3, :] = 6, "grass"                     # plateau
    t[1, 2:6] = "dirt"
    for r, lv in ((3, 5), (4, 4), (5, 3)):                # stairs down from the plateau
        h[r, 12:14], t[r, 12:14] = lv, "stone"
    for r in range(7, 11):                                # pond
        for c in range(12, 16):
            if (r, c) not in ((7, 12), (7, 15), (10, 15)):
                h[r, c], t[r, c] = 0, "water"
    for r in range(6, 12):
        for c in range(11, 17):
            if t[r, c] == "grass" and any(0 <= r + dr < R and 0 <= c + dc < C and t[r + dr, c + dc] == "water"
                                          for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
                h[r, c], t[r, c] = 1, "sand"
    h[9, 12], t[9, 12] = 1, "planks"                      # jetty
    for c in range(0, 12):
        t[12, c] = "cobble"
    for r in range(6, 13):
        t[r, 10] = "cobble"
    t[6, 10:12] = "cobble"
    t[6, 12:14] = "cobble"
    t[8:14, 0:5] = "dirt"
    h[8:10, 1:4], t[8:10, 1:4] = 5, "planks"              # a log cabin, its door on the south wall
    props = [("cave", 6, 3), ("torch", 5, 3), ("torch", 7, 3), ("cave", 2, 10), ("lantern", 4, 10),
             ("well", 6, 10), ("lantern", 11, 6), ("fence", 0, 7), ("fence", 1, 7), ("fence", 2, 7),
             ("bush", 8, 9), ("bush", 9, 11), ("bush", 7, 7), ("bush", 15, 12), ("bush", 16, 5),
             ("bush", 3, 5), ("bush", 9, 5), ("pot", 0, 11), ("pot", 0, 12), ("pot", 5, 13),
             ("crate", 4, 13), ("barrel", 3, 13), ("rock", 14, 4), ("rock", 1, 4),
             ("tree", 0, 5), ("tree", 17, 7), ("tree", 17, 10), ("tree", 16, 13), ("tree", 8, 13),
             ("tree", 1, 1), ("tree", 10, 0), ("tree", 16, 1), ("crystal", 4, 1), ("brazier", 15, 2)]
    units = [Unit("knight", 7, 12, 0, "N", hp=24),
             Unit("wolf", 6, 7, 1, "W", hp=6), Unit("wolf", 14, 12, 1, "W", hp=6),
             Unit("wolf", 4, 6, 1, "E", hp=6), Unit("wolf", 8, 1, 1, "S", hp=6),
             Unit("bandit", 3, 1, 1, "S", hp=6)]
    return Diorama(C, R, h, t, props, units, name="hillside corner", sea=False)


AFTERNOON = {
    "sun": [1.0, 0.86, 0.64], "sun_energy": 1.45, "sun_angle": [-38, -30],
    "fill": [0.55, 0.65, 1.0], "fill_energy": 0.32, "fill_angle": [-55, 150],
    "ambient": [0.6, 0.62, 0.8], "ambient_energy": 0.55, "background": [0.18, 0.3, 0.16],
    "fog": [0.7, 0.68, 0.75], "fog_density": 0.0012, "volumetric": 0.0,
    "torches": 0.55, "rim": 0.4, "blob": 0.5, "exposure": 0.97, "white": 1.6, "glow": 0.6,
    "saturation": 1.04, "contrast": 1.06,
    "water_deep": [0.04, 0.17, 0.36], "water_shallow": [0.12, 0.42, 0.66], "sparkle": 1.2,
    "water_spec": 0.3, "glare": 0.3}


def _unit_sheets(src: Sources, studio: Path, names: set[str]) -> tuple[dict, dict]:
    """{map unit name: sheet.png} + pixels-per-tile per unit (one scale: a 64 px person ~1.9
    tiles, frames sized like the library's)."""
    sheets, ppu = {}, {}
    plain = {"knight": ("knight",), "mage": ("mage",), "archer": ("archer", "dwarf"),
             "wolf": ("wolf", "small_wolf"), "dwarf": ("dwarf", "archer")}
    skins = {"warlock": (("mage", "knight"), "red"), "bandit": (("dwarf", "archer"), "red"),
             "shadowknight": (("knight",), "purple")}
    for n in names:
        if n in plain:
            folder = src.character(*plain[n])
            sheets[n] = folder / "sheet.png"
        elif n in skins:
            roles, colour = skins[n]
            folder = src.character(*roles)
            sheets[n] = reskin_sheet(folder, studio / "reskins" / f"{n}-{folder.name}", colour)
        else:
            folder = src.character(n)
            sheets[n] = folder / "sheet.png"
        meta = json.loads(Path(sheets[n]).with_suffix(".json").read_text(encoding="utf-8"))
        fr = meta["frames"]
        fh = (fr[0] if isinstance(fr, list) else next(iter(fr.values())))["frame"]["h"]
        kind = "animal" if n == "wolf" else "person"
        ppu[n] = 42.0 * fh / 51.0 if kind == "animal" else 34.0 * fh / 64.0
    return sheets, ppu


def _board_project(style: str, out: Path, src: Sources, d, script: dict, light: str,
                   fx: list[str], apron: int = 0) -> list[Path]:
    from .hd2d import make_hd2d
    from .maplib import install_board_extras
    studio = _studio(out)
    used = {u.name for u in d.units}
    sheets, ppu = _unit_sheets(src, studio, used)
    effects = effect_sheets(studio / "fx", fx)
    tiles = None
    if src.tiles_project is not None:
        from . import assets
        tiles = assets.Project(Path(src.tiles_project).name, Path(src.tiles_project))
    keep = {n: out / n for n in ("reports",) if (out / n).exists()}
    tmp_keep = {}
    for n, p in keep.items():                              # Kevin's reports survive a rebuild
        tmp_keep[n] = studio / f"_keep_{style}_{n}"
        shutil.rmtree(tmp_keep[n], ignore_errors=True)
        shutil.move(str(p), str(tmp_keep[n]))
    if out.exists():
        shutil.rmtree(out)
    files = make_hd2d(out, sheets, d, name=TITLES[style], iso=(style == "iso"), lighting=light,
                      effects=effects, tiles=tiles if tiles is not None else False,
                      autoplay=script, ui_kit=src.ui_kit, unit_ppu=ppu, apron=apron,
                      iso_scale={n: 1.0 / px_scale_of(sh, n) for n, sh in sheets.items()})
    files += install_board_extras(out, 32) or []
    for n, p in tmp_keep.items():
        shutil.move(str(p), str(out / n))
    return [Path(f) for f in files if f]


def build_tactics(out, src: Sources | None = None, godot: str | None = None) -> dict:
    out = Path(out)
    src = src or Sources.samples()
    script = {"mode": "tactics", "steps": []}               # (autoplay.gd is installed; not run)
    _board_project("tactics", out, src, tactics_map(), script, "sunset",
                   ["lightning", "dark", "slash", "spark", "fireball", "heal"])
    # the mage and the warlock cast: board.gd plays the effect named after the unit
    for unit, fx in (("mage", "lightning"), ("warlock", "dark")):
        if (out / "effects" / fx).exists():
            shutil.copytree(out / "effects" / fx, out / "effects" / unit, dirs_exist_ok=True)
            for f in (out / "effects" / unit).glob(f"{fx}*"):
                f.rename(f.with_name(f.name.replace(fx, unit, 1)))
    (out / "autoplay.json").unlink(missing_ok=True)
    pg = out / "project.godot"
    godot_runtime.set_setting(pg, "application", "run/main_scene", '"res://real3d.tscn"')
    install_runtime(out, ["pp_debug.gd", "pp_tactics.gd"],
                    {"PPDebug": "res://pp_runtime/pp_debug.gd", "PPTactics": "res://pp_runtime/pp_tactics.gd"})
    _window(out)
    _playtest_json(out, "tactics")
    write_readme(out, "tactics", "A 10x8 river crossing: your knight, mage and archer (blue) against two "
                 "wolves and a warlock on the raised east bank (red, played by a simple AI).\r\n"
                 "Board: real 3D (hd2d.py board.gd), classic iso camera, sunset + torches, the tiles of "
                 "the project's tilegen sets. Esc with nothing selected opens the looks menu (real 3D / "
                 "fake 3D / iso).")
    return {"style": "tactics", "folder": str(out)}


def build_iso(out, src: Sources | None = None, godot: str | None = None) -> dict:
    out = Path(out)
    src = src or Sources.samples()
    _board_project("iso", out, src, tactics_map(), {"mode": "tactics", "steps": []}, "sunset",
                   ["slash", "spark"])
    (out / "autoplay.json").unlink(missing_ok=True)
    pg = out / "project.godot"
    godot_runtime.set_setting(pg, "application", "run/main_scene", '"res://iso/iso.tscn"')
    install_runtime(out, ["pp_debug.gd", "pp_tactics.gd"],
                    {"PPDebug": "res://pp_runtime/pp_debug.gd", "PPTactics": "res://pp_runtime/pp_tactics.gd"})
    _window(out, 1280, 720)
    _playtest_json(out, "iso", {"hint_at": "top-right"})
    write_readme(out, "iso", "The same 10x8 river crossing in the isometric 2D look (iso_scene.py: "
                 "pre-rendered diamond columns, 640x360 pixel-perfect, painter's order). Blue = you, "
                 "red = a simple AI. The window scales the 640x360 picture by whole numbers.")
    return {"style": "iso", "folder": str(out)}


def build_action(out, src: Sources | None = None, godot: str | None = None, shot: str = "fireball") -> dict:
    out = Path(out)
    src = src or Sources.samples()
    script = {"mode": "action", "seed": 5, "hero": "knight", "coins": 0, "item": "sword", "light": "afternoon",
              "presets": {"afternoon": AFTERNOON}, "poof": "poof", "poof_scale": 1.4, "lean": 0.85,
              "fade_in": 0.0, "camera": {"yaw": 0, "pitch": 58, "dist": 15.5, "rate": 5.0, "dof": 0.015,
                                         "follow": "knight"},
              "shot": shot, "ranged": ["bandit"], "sword": 3, "shot_damage": 2, "bite": 2,
              "aggro": 4.5, "foe_speed": 2.4, "speed": 3.1, "steps": []}
    _board_project("action", out, src, action_map(), script, "noon",
                   ["slash", "spark", "poof", "sparkle", "fireball", "explosion"], apron=8)
    (out / "autoplay.json").replace(out / "action.json")
    (out / "play_action.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n\n[ext_resource type="Script" path="res://pp_runtime/action_play.gd" '
        'id="1"]\n\n[node name="Play" type="Node"]\nscript = ExtResource("1")\n', encoding="utf-8")
    pg = out / "project.godot"
    godot_runtime.set_setting(pg, "application", "run/main_scene", '"res://play_action.tscn"')
    install_runtime(out, ["pp_debug.gd", "action_play.gd"], {"PPDebug": "res://pp_runtime/pp_debug.gd"})
    _window(out)
    _playtest_json(out, "action")
    write_readme(out, "action", "A small hillside corner on the HD-2D board with a steep follow camera "
                 "(Link to the Past style, HD-2D lighting): a cliff with stone stairs and a cave, a pond "
                 "with a jetty and sand rim, a log cabin, bushes and pots to cut, four wolves that chase "
                 "you and a bandit on the cliff who shoots arrows. HUD = the UI kit's hearts, coins and "
                 "item box.")
    return {"style": "action", "folder": str(out)}


# --------------------------------------------------------------------------- 2D games (export path)

def staging_project(studio: Path, name: str, type_: str | None = None, base: Path | None = None,
                    **style_over):
    """A project inside the staging home (never the user's projects list): the type's house
    style, or a copy of `base`'s style.json, with overrides."""
    from dataclasses import asdict

    from . import assets
    folder = studio / "projects" / name
    folder.mkdir(parents=True, exist_ok=True)
    proj = assets.Project(name, folder)
    if base is not None and (Path(base) / "style.json").exists():
        st = assets.Project("base", Path(base)).style
        data = asdict(st)
        if type_:
            preset = assets.style_for_type(type_)
            data.update(project_type=type_, camera=preset.camera, tile_style=preset.tile_style,
                        export=preset.export, animations=list(preset.animations))
    else:
        data = asdict(assets.style_for_type(type_ or "custom"))
    data.update(style_over)
    known = {k: v for k, v in data.items() if k in assets.Style.__dataclass_fields__}
    proj.save_style(assets.Style(**known))
    return proj


class _NoGPU:
    """A ComfyUI client stand-in: any drawing request fails (the playtests never draw on the GPU)."""

    def __getattr__(self, name):
        raise RuntimeError("playtest: this re-render needs new GPU drawings (not allowed here)")


def side_character(src_folder: Path, proj, clips=("jump",), progress=None) -> tuple[Path, str]:
    """The character re-rendered for a side-scroller project: 2 directions (E, W) bent on the CPU
    from its existing drawn keys (drawnrig), plus side-scroller clips motions.py knows (jump).
    The camera stays the one the keys were drawn at (needs no new drawings). Falls back to a
    plain copy of the 16-direction sheet (the game uses its E/W rows) when that isn't possible.
    Returns (character folder, note)."""
    from . import assets, library
    say = progress or (lambda t: None)
    folder = Path(proj.folder) / "characters" / Path(src_folder).name
    if (folder / "sheet.json").exists() and (folder / ".playtest_side2").exists():
        return folder, (folder / ".playtest_side2").read_text(encoding="utf-8")
    shutil.rmtree(folder, ignore_errors=True)
    has_keys = (Path(src_folder) / "keys" / "keys.json").exists() and (Path(src_folder) / "rigged.glb").exists()
    stage_character(Path(src_folder), Path(proj.folder), keys=has_keys)
    note = "16-direction sheet as made (E/W rows used)"
    if has_keys:
        try:
            from .drawnrig import drawn_sheets
            from .motions import add_animations
            with assets.using_project(proj):
                ch = library.from_folder(folder)
                t = time.time()
                add_animations(ch.rigged, list(clips))
                add_animations(ch.rigged, ["walk", "run"], replace=True)     # planted feet
                from .drawnrig import no_new_drawings
                with no_new_drawings():
                    drawn_sheets(ch, directions=2, client=_NoGPU(), progress=say,
                                 frames=library.frames_for(None, locomotion=12))
                library.make_thumb(ch.sheet, ch.thumb)
            note = (f"re-rendered for the side-scroller project: E + W, clips + {', '.join(clips)}, bent on the "
                    f"CPU from the existing drawn keys in {time.time() - t:.0f} s (camera kept at the keys' "
                    "30 deg; a true 0 deg side view would need new GPU drawings)")
        except Exception as e:  # noqa: BLE001 - fall back to the sheet as it is
            shutil.rmtree(folder, ignore_errors=True)
            stage_character(Path(src_folder), Path(proj.folder))
            note = f"re-render not possible ({type(e).__name__}: {e}); 16-direction sheet as made"
    shutil.rmtree(folder / "keys", ignore_errors=True)    # big; the export doesn't need them
    for big in ("model.glb",):
        (folder / big).unlink(missing_ok=True)
    (folder / ".playtest_side2").write_text(note, encoding="utf-8")
    return folder, note


def _new_godot_project(out: Path, title: str, w: int, h: int, extra: str = "") -> Path:
    out.mkdir(parents=True, exist_ok=True)
    pg = out / "project.godot"
    pg.write_text(f"""config_version=5

[application]

config/name="{title}"
config/features=PackedStringArray("4.6", "GL Compatibility")

[display]

window/size/viewport_width={w}
window/size/viewport_height={h}
window/size/window_width_override={w * 2}
window/size/window_height_override={h * 2}
window/stretch/mode="canvas_items"
window/stretch/scale_mode="integer"

[rendering]

renderer/rendering_method="gl_compatibility"
renderer/rendering_method.mobile="gl_compatibility"
textures/canvas_textures/default_texture_filter=0
2d/snap/snap_2d_transforms_to_pixel=true
{extra}""", encoding="utf-8")
    return pg


def platformer_level(cols: int = 72, rows: int = 12, seed: int = 4) -> dict:
    """One short level: ground with steps, pits and a hill, one-way platforms, coins, hearts,
    enemies, a flag. Cells: [x, y] with y down; ground = solid cells."""
    rng = np.random.default_rng(seed)
    top = np.full(cols, 9)                         # the ground's top row per column (rows 9..11 solid)
    top[14:18] = 8
    top[18:22] = 7
    top[30:34] = 8
    top[46:52] = 7
    top[52:56] = 6
    top[62:] = 8
    pits = list(range(24, 27)) + list(range(39, 42))
    solid = []
    for x in range(cols):
        if x in pits:
            continue
        for y in range(int(top[x]), rows):
            solid.append([x, y])
    for x, y in ((8, 5), (9, 5)):                  # a floating block
        solid.append([x, y])
    platforms = [[24, 6, "left"], [25, 6, "middle"], [26, 6, "right"], [35, 5, "left"], [36, 5, "middle"],
                 [37, 5, "middle"], [38, 5, "right"], [40, 7, "single"], [57, 3, "left"], [58, 3, "right"]]
    coins = [[5, 7], [6, 7], [8, 4], [9, 4], [16, 6], [20, 5], [21, 5], [25, 5], [26, 5], [36, 4], [37, 4],
             [40, 6], [44, 6], [45, 6], [53, 4], [57, 2], [58, 2], [66, 6], [67, 6]]
    hearts = [[38, 4], [58, 1]]
    enemies = [{"x": 12, "y": 8, "kind": "wolf"}, {"x": 29, "y": 8, "kind": "wolf"},
               {"x": 44, "y": 8, "kind": "wolf"}, {"x": 49, "y": 6, "kind": "wolf"}, {"x": 64, "y": 7, "kind": "wolf"}]
    return {"cols": cols, "rows": rows, "solid": solid, "platforms": platforms, "coins": coins,
            "hearts": hearts, "enemies": enemies, "start": [2, 8], "flag": [69, 7], "seed": int(rng.integers(0, 99))}


def build_platformer(out, src: Sources | None = None, godot: str | None = None, progress=None) -> dict:
    """Side-scroller: a side-scroller staging project (tiles, background, effects, the hero
    re-rendered E/W), exported with gameexport.export_to_game into a fresh Godot project whose
    runtime (side_game.gd) reads the manifest by name."""
    from . import assets, gameexport, tilegen
    from .backgrounds import make_background
    from .effects import make_effect
    out = Path(out)
    src = src or Sources.samples()
    studio = _studio(out)
    hero_src = src.character("archer", "dwarf", "knight")
    foe_src = src.character("small_wolf", "wolf")
    hero_base = hero_src.parent.parent                  # the hero's own project: its scale + palette
    proj = staging_project(studio, "platformer", "side-scroller", base=hero_base, directions=2,
                           elevation=assets.Project("b", hero_base).style.elevation)
    st = proj.style
    made = []
    hero, note = side_character(hero_src, proj, progress=progress)
    made.append(library_character(hero))
    foe = stage_character(foe_src, Path(proj.folder))
    made.append(library_character(foe))
    with assets.using_project(proj):
        sets = assets.list_assets("tileset", project=proj)
        if not sets:
            sw = tilegen.builtin_swatch("grass", 32, seed=3, style=st)
            gs = tilegen.build_tileset("grass", sw.texture, base="dirt", camera="side", style=st, project=proj,
                                       sources=sw.sources, variants=3)
            sets = [tilegen.save_tileset(gs, "Grass ledge tiles", project=proj, swatch=sw)]
        made += sets[:1]
        bgs = assets.list_assets("background", project=proj)
        if not bgs:
            bgs = [make_background("forest", "sunset", "Forest sunset", project=proj, width=640, height=384, seed=2)]
        made += bgs[:1]
        fx = {a.name: a for a in assets.list_assets("effect", project=proj)}
        for kind, name in (("smoke", "Poof"), ("hit_spark", "Hit spark"), ("sparkle", "Sparkle")):
            if name not in fx:
                fx[name] = make_effect(kind, name, project=proj, seed=3)
        made += [fx["Poof"], fx["Hit spark"], fx["Sparkle"]]
    if src.ui_kit is not None:
        kit_dst = Path(proj.folder) / "ui" / Path(src.ui_kit).name
        if not kit_dst.exists():
            shutil.copytree(src.ui_kit, kit_dst)
        made += [a for a in assets.list_assets("ui", project=proj)][:1]
    keep = studio / "_keep_platformer_reports"
    if (out / "reports").exists():
        shutil.rmtree(keep, ignore_errors=True)
        shutil.move(str(out / "reports"), str(keep))
    if out.exists():
        shutil.rmtree(out)
    _new_godot_project(out, TITLES["platformer"], 640, 360)
    rep = gameexport.export_to_game(made, "godot", str(out), "art", project=proj)
    if keep.exists():
        shutil.move(str(keep), str(out / "reports"))
    lv = platformer_level()
    lv["hero"] = gameexport.asset_id(made[0])
    lv["enemy_kinds"] = {"wolf": gameexport.asset_id(made[1])}
    lv["tileset"] = gameexport.asset_id(made[2])
    lv["background"] = gameexport.asset_id(made[3])
    lv["effects"] = {"poof": gameexport.asset_id(fx["Poof"]), "spark": gameexport.asset_id(fx["Hit spark"]),
                     "sparkle": gameexport.asset_id(fx["Sparkle"])}
    lv["ui"] = gameexport.asset_id(made[-1]) if src.ui_kit is not None else ""
    lv["tile"] = 32
    tl = json.loads((Path(made[2].folder) / "tileset.json").read_text(encoding="utf-8"))
    lv["platform_atlas"] = _side_platform_atlas(tl)
    (out / "level.json").write_text(json.dumps(lv, indent=1), encoding="utf-8")
    (out / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n\n[ext_resource type="Script" path="res://pp_runtime/side_game.gd" '
        'id="1"]\n\n[node name="Game" type="Node2D"]\nscript = ExtResource("1")\n', encoding="utf-8")
    pg = out / "project.godot"
    godot_runtime.set_setting(pg, "application", "run/main_scene", '"res://main.tscn"')
    install_runtime(out, ["pp_debug.gd", "side_game.gd", "side_player.gd", "side_enemy.gd", "pickup_2d.gd",
                          "pp_manifest.gd"], {"PPDebug": "res://pp_runtime/pp_debug.gd"})
    _playtest_json(out, "platformer", {"hint_at": "bottom-left"})
    write_readme(out, "platformer", "One short level (72 x 12 tiles of 32 px) through the app's export path: "
                 "a side-scroller project's grass-ledge tileset (tilegen, side camera: Godot terrain painting + "
                 "one-way platforms), a forest-sunset parallax background (backgrounds.py), effects (smoke, "
                 "hit spark, sparkle), the UI kit HUD, and the hero " + f"({note}). Enemies: wolves (E/W rows "
                 "of their 16-direction sheet). Everything is loaded by name from "
                 "art/pixel_pipeline_manifest.json.")
    return {"style": "platformer", "folder": str(out), "hero": note, "export": rep.get("counts"),
            "warnings": rep.get("warnings")}


def library_character(folder: Path):
    from . import library
    ch = library.from_folder(folder)
    if ch is None:
        raise ValueError(f"{folder} is not a library character")
    return ch


def _side_platform_atlas(layout: dict) -> dict:
    """{left|middle|right|single: [col, row]} from a side tileset's layout json (row 4 by the
    tilesets.build_side_set layout when the json doesn't say)."""
    out = {"left": [0, 4], "middle": [1, 4], "right": [2, 4], "single": [3, 4]}
    for t in layout.get("tiles", []) if isinstance(layout, dict) else []:
        k = t.get("platform")
        if k in out:
            out[k] = [int(t["col"]), int(t["row"])]
    return out


# --------------------------------------------------------------------------- deck-builder

# id: (name, cost, type, rarity, faction, art, ops, target, fx, anim)
#   art: "char:<role>" (the character's painted picture), "item:<n>" / "prop:<n>" (n-th picture
#   of the sources), "fx:<effect>" (the effect's brightest frame on a dark backdrop), "card:<n>"
#   (an existing card asset's own art, re-texted). ops: [op, value(, times)] run in order.
CARD_DEFS = {
    "strike": ("Strike", 1, "Attack", "common", "neutral", "char:knight", [["dmg", 6]], "enemy", "slash", "attack"),
    "defend": ("Defend", 1, "Skill", "common", "neutral", "char:dwarf", [["block", 5]], "self", "buff", "idle"),
    "fireball": ("Fireball", 2, "Attack", "common", "fire", "fx:fireball", [["dmg", 12]], "enemy", "fireball", "attack"),
    "fire": ("Fire", 2, "Attack", "common", "fire", "card:0", [["dmg", 8], ["burn", 3]], "enemy", "explosion", "attack"),
    "ice": ("Ice Shard", 1, "Attack", "common", "water", "fx:ice", [["dmg", 7], ["block", 3]], "enemy", "ice", "attack"),
    "lightning": ("Chain Bolt", 1, "Attack", "rare", "water", "fx:lightning", [["dmg", 3, 3]], "enemy", "lightning", "attack"),
    "holy": ("Holy Light", 1, "Skill", "rare", "holy", "fx:heal", [["heal", 6], ["exhaust", 1]], "self", "heal", "idle"),
    "potion": ("Healing Potion", 0, "Skill", "common", "nature", "item:healing", [["heal", 4], ["exhaust", 1]], "self", "heal", "idle"),
    "sword": ("Rusty Sword", 1, "Attack", "common", "neutral", "item:sword", [["dmg", 9]], "enemy", "slash", "attack"),
    "volley": ("Volley", 1, "Attack", "common", "nature", "char:archer", [["dmg", 4, 2]], "enemy", "spark", "attack"),
    "insight": ("Arcane Insight", 0, "Skill", "rare", "shadow", "char:mage", [["draw", 2]], "self", "levelup", "idle"),
    "battlecry": ("Battle Cry", 1, "Power", "rare", "fire", "fx:buff", [["str", 2]], "self", "buff", "idle"),
    "poison": ("Poison Cloud", 1, "Skill", "common", "nature", "fx:poison", [["poison", 5]], "enemy", "poison", "idle"),
    "frost": ("Frost Nova", 2, "Attack", "rare", "water", "fx:frost", [["dmg", 8], ["block", 8]], "enemy", "frost", "attack"),
    "explosion": ("Big Bang", 3, "Attack", "epic", "fire", "fx:explosion", [["dmg", 24]], "enemy", "explosion", "attack"),
    "barrel": ("Supply Barrel", 1, "Skill", "common", "neutral", "prop:barrel", [["block", 6], ["draw", 1]], "self", "buff", "idle"),
    "dark": ("Shadow Pact", 0, "Skill", "epic", "shadow", "fx:dark", [["lose_hp", 3], ["energy", 2]], "self", "dark", "idle"),
}
STARTER = ["strike"] * 5 + ["defend"] * 4 + ["fireball"]
REWARD_POOL = [k for k in CARD_DEFS if k not in ("strike", "defend")]


def card_text(ops) -> str:
    words = []
    for op in ops:
        k, v = op[0], op[1]
        n = op[2] if len(op) > 2 else 1
        words.append({"dmg": f"Deal {v} damage" + (f" {n} times" if n > 1 else ""),
                      "block": f"Gain {v} Block", "heal": f"Heal {v} HP", "draw": f"Draw {v} card" + ("s" if v > 1 else ""),
                      "str": f"Gain {v} Strength", "poison": f"Apply {v} Poison", "burn": f"Apply {v} Burn",
                      "energy": f"Gain {v} Energy", "lose_hp": f"Lose {v} HP", "exhaust": "Exhaust"}[k] + ".")
    return " ".join(words)


def _backdrop(w: int, h: int, colour, seed: int = 0) -> np.ndarray:
    """A dark vignette in a faction colour (card art behind an effect)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h * 0.55) / (h / 2)) ** 2)
    k = np.clip(1.0 - r * 0.75, 0.12, 1.0)[..., None]
    c = np.array(colour, np.float32)[None, None, :3]
    img = np.concatenate([np.clip(c * 0.45 * k + 10, 0, 255), np.full((h, w, 1), 255, np.float32)], -1)
    rng = np.random.default_rng(seed)
    for _ in range(40):                                   # a few faint stars / motes
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        img[y, x, :3] = np.clip(img[y, x, :3] + 60, 0, 255)
    return img.astype(np.uint8)


def _fx_art(name: str, colour) -> np.ndarray:
    from .effects import peak_frame, render_effect
    from .spellfx import render_spell
    kind, k, opts = FX_SPECS[name]
    fx = render_spell(k, seed=3, **opts) if kind == "spell" else render_effect(k, seed=3)
    fr = peak_frame(fx)
    ys, xs = np.nonzero(fr[..., 3])
    if len(xs):
        fr = fr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    W, H = 324, 216
    s = max(1, int(min(W * 0.8 / fr.shape[1], H * 0.8 / fr.shape[0])))
    big = np.asarray(Image.fromarray(fr).resize((fr.shape[1] * s, fr.shape[0] * s), Image.NEAREST))
    bg = Image.fromarray(_backdrop(W, H, colour))
    bg.alpha_composite(Image.fromarray(big), ((W - big.shape[1]) // 2, (H - big.shape[0]) // 2))
    return np.asarray(bg)


def build_cards(out, src: Sources | None = None, godot: str | None = None, progress=None) -> dict:
    """Deck-builder: cards made with cards.py from the app's own pictures (characters' painted
    pictures, items, props, effect frames, an existing card) - CPU only - plus enemies (library
    characters + re-skins), effects and the UI kit, exported with gameexport into a fresh Godot
    project that card_game.gd runs."""
    from . import assets, cards, gameexport
    from .cards import FACTION_COLOURS
    from .effects import make_effect
    out = Path(out)
    src = src or Sources.samples()
    studio = _studio(out)
    base = src.character("knight").parent.parent
    proj = staging_project(studio, "cards", "cards-ui", base=Path(src.ui_kit).parent.parent if src.ui_kit else base)
    pf = Path(proj.folder)
    cards.register_kind()
    have = {a.folder.name: a for a in cards.list_cards(project=proj)}
    made_cards = {}
    art_dir = studio / "card_art"
    art_dir.mkdir(exist_ok=True)
    items = {p.name: p for p in src.items}
    props_ = {p.name: p for p in src.props}

    def pick(pool: dict, word: str):
        for k, p in pool.items():
            if word in k:
                return p
        return next(iter(pool.values()), None)

    for cid, (name, cost, typ, rarity, faction, art, ops, *_rest) in CARD_DEFS.items():
        slug = f"pt-{cid}"
        if slug in have:
            made_cards[cid] = have[slug]
            continue
        info = cards.CardInfo(name=name, cost=cost, type=typ, text=card_text(ops), rarity=rarity,
                              faction=faction, size="small")
        a = cards.new_card(info, project=proj)
        new_folder = a.folder.parent / slug                 # stable ids: cards/pt-<id>
        shutil.move(str(a.folder), str(new_folder))
        a = cards.open_card(new_folder)
        what, _, arg = art.partition(":")
        pic = None
        if what == "char":
            f = src.character(arg) / "picture.png"
            pic = f if f.exists() else None
        elif what == "item" and items:
            p = pick(items, arg)
            pic = next((p / n for n in ("picture_v1.png", "icon_4x.png") if (p / n).exists()), None)
        elif what == "prop" and props_:
            p = pick(props_, arg)
            pic = next((p / n for n in ("drawn_v1.png", "sprite_v1.png", "model.front.png") if (p / n).exists()), None)
        elif what == "fx":
            hexs = FACTION_COLOURS.get(faction, {}).get("banner", ["5a4a70"] * 4)[3]
            colour = [int(hexs[i:i + 2], 16) for i in (0, 2, 4)]
            pic = art_dir / f"{cid}.png"
            Image.fromarray(_fx_art(arg, colour)).save(pic)
        elif what == "card" and src.cards:
            c = src.cards[int(arg) % len(src.cards)]
            for n in ("art_raw_v1.png", "art_hd.png", "art.png"):
                if (c / n).exists():
                    pic = c / n
                    break
        if pic is not None:
            cards.set_art(a, pic, project=proj)
        cards.render_card(a, project=proj)
        made_cards[cid] = a
    # enemies + the hero: library characters (re-skins staged as their own characters)
    chars = {}
    for role, folder_role, colour, label in (("hero", "knight", None, None), ("wolf", "small_wolf", None, None),
                                             ("bandit", "dwarf", "red", "Bandit"),
                                             ("shadow", "knight", "purple", "Shadow Knight"),
                                             ("warlock", "mage", "red", "Blood Warlock")):
        srcf = src.character(folder_role, "knight")
        name = None if colour is None else f"pt-{role}"
        dst = stage_character(srcf, pf, name=name)
        if colour is not None and not (dst / ".reskinned").exists():
            reskin_sheet(srcf, dst, colour)
            cj = json.loads((dst / "character.json").read_text(encoding="utf-8"))
            cj["name"] = label
            (dst / "character.json").write_text(json.dumps(cj, indent=1), encoding="utf-8")
            for stale in ("sheet.tres", "sheet_facing.gd", "sheet_glow.png", "thumb.png"):
                (dst / stale).unlink(missing_ok=True)
            (dst / ".reskinned").write_text(colour)
        chars[role] = library_character(dst)
    fxs = {a.name: a for a in assets.list_assets("effect", project=proj)}
    fx_kinds = {"slash": "sword_combo", "fireball": "fireball", "ice": "ice_shard", "lightning": "lightning_bolt",
                "heal": "holy_light", "spark": "hit_spark", "levelup": "level_up_burst", "buff": "buff_aura",
                "poison": "poison_cloud", "frost": "frost_nova", "explosion": "big_explosion", "dark": "dark_magic",
                "claw": "hit_spark"}
    for n, kind in fx_kinds.items():
        label = f"pt {n}"
        if label not in fxs:
            fxs[label] = make_effect(kind, label, project=proj, seed=3, **({"hits": 1} if kind == "sword_combo" else {}))
    made = list(made_cards.values()) + list(chars.values()) + [fxs[f"pt {n}"] for n in fx_kinds]
    if src.ui_kit is not None:
        kit_dst = pf / "ui" / Path(src.ui_kit).name
        if not kit_dst.exists():
            shutil.copytree(src.ui_kit, kit_dst)
        kit = assets.list_assets("ui", project=proj)[:1]
        made += kit
    keep = studio / "_keep_cards_reports"
    if (out / "reports").exists():
        shutil.rmtree(keep, ignore_errors=True)
        shutil.move(str(out / "reports"), str(keep))
    if out.exists():
        shutil.rmtree(out)
    _new_godot_project(out, TITLES["cards"], 1280, 720)
    godot_runtime.set_setting(out / "project.godot", "display", "window/size/window_width_override", "1280")
    godot_runtime.set_setting(out / "project.godot", "display", "window/size/window_height_override", "720")
    rep = gameexport.export_to_game(made, "godot", str(out), "art", project=proj)
    if keep.exists():
        shutil.move(str(keep), str(out / "reports"))
    game = {"cards": {}, "starter": STARTER, "reward_pool": REWARD_POOL,
            "chars": {k: gameexport.asset_id(v) for k, v in chars.items()},
            "fx": {n: gameexport.asset_id(fxs[f"pt {n}"]) for n in fx_kinds},
            "ui": gameexport.asset_id(made[-1]) if src.ui_kit is not None else ""}
    for cid, a in made_cards.items():
        name, cost, typ, rarity, faction, art, ops, target, fx, anim = CARD_DEFS[cid]
        game["cards"][cid] = {"asset": gameexport.asset_id(a), "folder": a.folder.name, "name": name, "cost": cost,
                              "type": typ, "rarity": rarity, "ops": ops, "target": target, "fx": fx, "anim": anim,
                              "text": card_text(ops)}
    pic = src.character("knight") / "picture.png"       # the hero's painted portrait (HUD)
    if pic.exists():
        (out / "portraits").mkdir(exist_ok=True)
        im = Image.open(pic).convert("RGBA")
        box = im.getbbox() or (0, 0, im.width, im.height)
        im = im.crop((box[0], box[1], box[2], box[1] + min(box[3] - box[1], box[2] - box[0])))  # head + chest
        im.thumbnail((192, 192), Image.LANCZOS)
        im.save(out / "portraits" / "hero.png")
        game["portrait"] = "res://portraits/hero.png"
    (out / "deck.json").write_text(json.dumps(game, indent=1), encoding="utf-8")
    (out / "main.tscn").write_text(
        '[gd_scene load_steps=2 format=3]\n\n[ext_resource type="Script" path="res://pp_runtime/card_game.gd" '
        'id="1"]\n\n[node name="Game" type="Control"]\nlayout_mode = 3\nanchors_preset = 15\nanchor_right = 1.0\n'
        'anchor_bottom = 1.0\ngrow_horizontal = 2\ngrow_vertical = 2\nscript = ExtResource("1")\n', encoding="utf-8")
    godot_runtime.set_setting(out / "project.godot", "application", "run/main_scene", '"res://main.tscn"')
    install_runtime(out, ["pp_debug.gd", "card_game.gd", "pp_manifest.gd"], {"PPDebug": "res://pp_runtime/pp_debug.gd"})
    _playtest_json(out, "cards", {"hint_at": "under-bar"})
    write_readme(out, "cards", "A tiny Slay-the-Spire-style run: a 10-card starting deck (5 Strike, 4 Defend, "
                 "1 Fireball), 3 energy a turn, 5 cards drawn, draw / discard piles that reshuffle. Map: fight -> "
                 "(fight | rest | shop) -> (elite | rest) -> boss. After each win pick 1 of 3 cards for your deck. "
                 "The cards are the app's own (cards.py frames, fonts, layout; Card.tscn from the export) with art "
                 "from the sample characters' painted pictures, items, a prop, effect frames and the QA project's "
                 "Fire card. Enemies are library characters (and re-skins); hits play the app's spell effects; the "
                 "UI uses the UI kit's theme, panels, icons and font.")
    return {"style": "cards", "folder": str(out), "export": rep.get("counts"), "warnings": rep.get("warnings"),
            "skipped": rep.get("skipped")}


BUILDERS = {"tactics": build_tactics, "action": build_action, "iso": build_iso,
            "platformer": build_platformer, "cards": build_cards}


def build_playtest(style: str, out_dir, sources: Sources | None = None, *, godot: str | None = None,
                   launchers_root=None, import_: bool = True, loco_frames: int | None = 12) -> dict:
    """Build one playtest game into out_dir (replaced; its reports/ kept). Returns a report.
    loco_frames: the characters are first copied with planted-feet walk/run at this many frames
    (locomotion_copies; None = as they are)."""
    if style not in STYLES:
        raise ValueError(f"style must be one of {', '.join(STYLES)}")
    t = time.time()
    if loco_frames:
        sources = locomotion_copies(sources or Sources.samples(), _studio(Path(out_dir)), loco_frames)
    rep = BUILDERS[style](Path(out_dir), sources, godot)
    write_launchers(out_dir, style, godot, launchers_root)
    if import_:
        rep["import_log"] = godot_import(out_dir, godot)[-1500:]
    rep["seconds"] = round(time.time() - t, 1)
    return rep


def build_all(root, sources: Sources | None = None, styles=STYLES, godot: str | None = None) -> list[dict]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    return [build_playtest(s, root / s, sources, godot=godot, launchers_root=root) for s in styles]
