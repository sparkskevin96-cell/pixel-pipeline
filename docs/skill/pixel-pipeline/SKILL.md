---
name: pixel-pipeline
description: Make game art with Kevin's Pixel Pipeline asset studio (MCP server "pixel-pipeline") and put it into this game - characters (drawn, front first), items, props, tilesets, maps, effects, weather/water/skies, portraits, cards, UI kits - then export into the Godot/Unity project with a manifest. Use when the game needs sprites, tiles, icons, effects or UI art, or when assets must be (re)exported.
---

# Pixel Pipeline in this game

Pixel Pipeline runs on this PC (MCP server `pixel-pipeline`; full guide:
C:\Users\spark\pixel-pipeline\docs\FOR_CLAUDE.md). One studio **project** per game holds the
house style; every asset is made in it and exported into this repo.

## Before making anything
1. `list_projects` -> is there a project for this game? If not: `create_project(name, type, engine)`
   (type hd2d / topdown-rpg / side-scroller / isometric / cards-ui) and set the look with
   `set_house_style({...})`. Otherwise `switch_project`.
2. `list_assets` to reuse what exists. `plan_asset("...")` tells kind + GPU minutes before spending them.

## Making
- `make(description)` routes anything; or the specific tools (`make_character`, `make_item`,
  `make_prop`, `make_terrain_tiles`, `make_effect`, `make_atmosphere`, `make_portrait`,
  `make_card`, `make_ui_kit`, `make_map`).
- GPU work is a background job: poll `job_status(job, wait_s=60)`. ONE GPU job at a time; don't
  queue more than ~30-45 min of GPU work without checking results with the user.
- Characters stop at the front drawing. LOOK at it, say what you see, then `character_approve`
  (~13 min) or `character_redraw_front` / `character_change_words` (new_model=true if the shape is
  wrong). Ask the user before approving if the look matters to them.
- Always look at the attached previews and `quality.warnings`; `pick_variation` for another
  icon / drawing / card art / tile texture.

## Exporting into this repo
- `export_to_game(project_path=<this repo's Godot/Unity project folder>, engine=..., subdir="art",
  verify=true)` -> files + `art/pixel_pipeline_manifest.json` (asset id -> files, res:// paths,
  frame size, animations with directions/frames/ms/loop, feet anchor, ppu, tile size; `by_name`).
- Load assets in game code through the manifest (by name), never by guessing file names.
- Re-export after any change (incremental). Don't hand-edit exported art: re-make or re-skin it in
  the studio. Drafts aren't exported until approved.

## Godot loading snippet
```gdscript
var man := JSON.parse_string(FileAccess.get_file_as_string("res://art/pixel_pipeline_manifest.json"))
func frames_of(key: String) -> SpriteFrames:
    return load(man.assets[man.by_name[key]].res.sprite_frames)
```
Character sheets have one animation per `action_DIR` (e.g. `walk_S`); the exported `_facing.gd`
gives `face(velocity)` and `act(&"walk")`.
