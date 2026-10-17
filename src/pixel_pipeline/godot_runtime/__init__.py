"""GDScript runtime the app can ship into games (playtest maps first, see playtest.py).

The .gd files live beside this module (package data) and are copied into a Godot project by
`install(project_dir, names)`; `add_autoloads(project_godot, {name: res_path})` registers the
autoload ones in project.godot.

    pp_debug.gd      autoload PPDebug: F1 controls hint, F3 debug overlay (FPS, game state from
                     nodes in group "pp_debug", collision shapes), F12 report (screenshot +
                     state -> res://reports/), F5 restart
    pp_tactics.gd    autoload PPTactics: enemy AI for the HD-2D board (board.gd) and the iso
                     scene (iso.gd) + their debug state
    action_play.gd   live ALttP-style controls on the HD-2D board (extends res://autoplay.gd:
                     reuses its HUD, swing/hit/die/drops/pickups/projectiles)
    side_game.gd     side-scroller level: tiles from a manifest tileset, parallax, player,
                     enemies, pickups, HUD
    side_player.gd   platformer CharacterBody2D (run, jump with coyote time + buffer, attack)
    side_enemy.gd    patrol/chase walker (turns at walls and ledges)
    pickup_2d.gd     a bobbing coin/heart Area2D
    card_game.gd     a small deck-builder (map of nodes, fights, rewards, deck viewer, rest, shop)
    pp_manifest.gd   reads pixel_pipeline_manifest.json (sprite frames, anchors, by name)
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

HERE = Path(__file__).parent
FILES = tuple(sorted(p.name for pat in ("*.gd", "*.gdshader") for p in HERE.glob(pat)))
SUBDIR = "pp_runtime"


def source(name: str) -> str:
    """The text of one runtime script (e.g. "pp_debug.gd")."""
    p = HERE / name
    if not p.exists():
        raise ValueError(f"No runtime script {name} (have: {', '.join(FILES)})")
    return p.read_text(encoding="utf-8")


def install(project_dir, names=None, subdir: str = SUBDIR) -> list[Path]:
    """Copy runtime scripts into <project>/<subdir>/ (all when names is None)."""
    out = Path(project_dir) / subdir
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for n in (names or FILES):
        dst = out / n
        shutil.copyfile(HERE / n, dst)
        written.append(dst)
    return written


def add_autoloads(project_godot, autoloads: dict[str, str]) -> Path:
    """Add {name: "res://..."} autoloads to project.godot (kept if already there)."""
    p = Path(project_godot)
    text = p.read_text(encoding="utf-8")
    lines = [f'{k}="*{v}"' for k, v in autoloads.items() if not re.search(rf"^{k}=", text, re.M)]
    if not lines:
        return p
    if "[autoload]" in text:
        text = text.replace("[autoload]\n", "[autoload]\n\n" + "\n".join(lines) + "\n", 1)
    else:
        text = text.rstrip() + "\n\n[autoload]\n\n" + "\n".join(lines) + "\n"
    p.write_text(text, encoding="utf-8")
    return p


def set_setting(project_godot, section: str, key: str, value: str) -> Path:
    """Set `key=value` under [section] in project.godot (value is written as given)."""
    p = Path(project_godot)
    text = p.read_text(encoding="utf-8")
    pat = re.compile(rf"^{re.escape(key)}=.*$", re.M)
    head = f"[{section}]"
    if head in text:
        start = text.index(head)
        end = text.find("\n[", start + 1)
        end = len(text) if end < 0 else end
        block = text[start:end]
        if pat.search(block):
            block = pat.sub(f"{key}={value}", block)
        else:
            block = block.rstrip("\n") + f"\n{key}={value}\n"
        text = text[:start] + block + ("\n" if end < len(text) and not block.endswith("\n\n") else "") + text[end:]
    else:
        text = text.rstrip() + f"\n\n{head}\n\n{key}={value}\n"
    p.write_text(text, encoding="utf-8")
    return p
