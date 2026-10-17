# Driving Pixel Pipeline from Claude (MCP / CLI)

Pixel Pipeline is Kevin's asset studio: characters, items, props/buildings, tilesets, effects,
weather/water/skies, portraits, cards, UI kits and maps, all in one project's **house style**,
exported straight into a Godot or Unity game. This page is for Claude (Claude Code in a game
repo, or scripts) driving it without the desktop app.

Everything the app's Make / Characters / Library pages do is a tool here. The logic is in
`src/pixel_pipeline/agent.py` (plain functions returning JSON); the MCP tools
(`mcp_server.py`, "asset studio" section) and the CLI (`cli_agent.py`, `--json`) wrap it.

## Set it up in a game repo

The server runs from the pixel-pipeline repo's venv (the installed Windows app does not ship the
MCP server). One time, from the game repo:

```powershell
claude mcp add --transport stdio --scope project --env PIXEL_GPU_LOCK=C:\AI\gpu.lock `
  pixel-pipeline -- C:\Users\spark\pixel-pipeline\.venv\Scripts\pixel-pipeline-mcp.exe
```

That writes `.mcp.json` in the game repo; or write it yourself:

```json
{
  "mcpServers": {
    "pixel-pipeline": {
      "command": "C:\\Users\\spark\\pixel-pipeline\\.venv\\Scripts\\pixel-pipeline-mcp.exe",
      "args": [],
      "env": {
        "PIXEL_GPU_LOCK": "C:\\AI\\gpu.lock",
        "GODOT_BIN": "C:\\Users\\spark\\AppData\\Local\\Microsoft\\WinGet\\Packages\\GodotEngine.GodotEngine_Microsoft.Winget.Source_8wekyb3d8bbwe\\Godot_v4.6.1-stable_win64_console.exe"
      }
    }
  }
}
```

Without the exe: `"command": "C:\\Users\\spark\\pixel-pipeline\\.venv\\Scripts\\python.exe", "args":
["-m", "pixel_pipeline.mcp_server"]`. Restart Claude Code, `/mcp` should list `pixel-pipeline`
with ~64 tools. `PIXEL_GPU_LOCK` shares one GPU lock with Kevin's other scripts/agents
(`tools/gpu_lock.py` uses the same file protocol); without it the lock is
`~/Pixel Pipeline/gpu.lock`. `GODOT_BIN` lets `verify=true` import the game headless.

A Claude Code skill for game repos is in `docs/skill/pixel-pipeline/SKILL.md`: copy the folder to
`<game repo>/.claude/skills/pixel-pipeline/`.

The CLI (same operations, for scripts or when MCP isn't set up):
`C:\Users\spark\pixel-pipeline\.venv\Scripts\pixel-pipeline.exe <command> --json`.

## The typical game workflow

1. **Project + house style** (once per game)
   - `list_projects`; `create_project(name, type, engine)`: types `hd2d` (sprites in a 3D world,
     16 directions, Godot HD-2D), `topdown-rpg` (4 directions, 16 px tiles), `side-scroller`
     (2 directions), `isometric`, `cards-ui`, `custom`. The type sets sizes, camera, tile size,
     ppu, clip set and exports.
   - `set_house_style({"style_prompt": "...", "shading": "cel", "palette": "lospec:resurrect-64"})`.
     Every generator appends these words; `get_house_style` shows the words actually added.
   - After a few good assets: tag them `approved` and `lock_palette(32)` so everything snaps to one palette.
2. **Characters, front first**
   - `plan_asset("a dwarf warrior with a big axe")` = dry run (kind, settings, GPU minutes).
   - `make_character(...)` -> job. It makes the picture, 3D model, rig, all clips, and only the
     FRONT drawing + idle, then stops: a **draft**.
   - `job_status(job, wait_s=60)` until done; the result shows the front + idle. LOOK at it
     (`character_review` again any time). Then one of:
     `character_approve` (draws the other directions, ~13 min for 16) /
     `character_redraw_front` (new seed, ~40 s) / `character_change_words(words, new_model)` /
     `character_pick_try(n)`.
   - After approval: `get_asset` shows all directions + every animation; `quality.drawn.worst`
     names weak directions -> `character_redraw(ref, ["NE"])`.
   - More clips: `list_motions`, `add_animation(ref, ["cast", "jump"])`, `add_mocap`, `add_springs`.
     Re-skins: `reskin_character(ref, "Blue Knight", colour="blue")` (no GPU).
3. **World**: `make_terrain_tiles("cobblestone road tiles")` (one per terrain: grass, dirt, sand,
   snow, stone, road, water, cliff, fence, wall), `make_map(template)` + `map_info` / `map_edit`
   (maps draw with the project's newest tilesets), `make_prop("thatched cottage")`,
   `make_atmosphere(kind="water"|"weather"|"background", ...)`.
4. **Items, effects, UI**: `make_item("rusty iron sword")` (3 icons + world sprite;
   `pick_variation(2, ref)`), `make_effect("ice nova")`, `make_ui_kit("stone")`,
   `make_portrait("Knight portrait, angry and happy")`, `make_card(...)`.
5. **Into the game**: `export_to_game(project_path="C:/Games/MyGame", engine="godot",
   subdir="art", verify=true)`. Re-run it whenever assets change: unchanged ones are skipped.

`make(description)` is the one-box router (same as the Make tab) when the kind is obvious.
Assets are referred to by id `kind/folder` (`character/dwarf`, `item/rusty-iron-sword`), a name
works if unique. Use `list_assets(kind=, query=, tag=, collection=)` to find them.

## Jobs

- Anything with GPU work returns `{"job": id, "status": "queued"|"running", "estimate_min": ...}`
  at once (`wait_s` makes it wait). CPU-only work (UI kits, maps, effects, weather, water,
  procedural backgrounds, animations, exports) waits up to 300 s and usually returns done.
- `job_status(job, wait_s=60)`: percent, stage, what it is doing, eta, `minutes` so far; done ->
  `result` with the asset summary, previews (attached as images) and quality checks.
- **One GPU job at a time** (queue in this server; the lock file across processes); 20 s rest
  between GPU jobs (`PIXEL_AGENT_PAUSE`). `list_jobs` shows the queue and who has the GPU.
- `cancel_job`: queued -> dropped; running -> stops at its next step, finished work is kept
  (drawn directions, variations).
- Records: `<project>/agent_jobs/<id>.json`. If the server dies, its jobs show as
  `interrupted`; `resume_jobs()` queues them again (nothing restarts by itself). A drawn character
  carries on from its kept drawings; other makers start that job over.

## Costs and timings (RTX 4070 laptop, 8 GB; makeany.GPU, measured 2026-10-14/15)

| What | GPU | CPU | Notes |
|---|---|---|---|
| Cold ComfyUI start | ~80 s once | | warm ~20 s; the app/server start it hidden |
| SDXL picture | ~15 s each | | 0.25 min |
| Character, front first | ~3.3 min | ~2 min rig | model 2.5 min + the front drawing |
| ... after `character_approve` | ~0.75 min / direction | ~1.8 min bending | 16 dirs: 12-19 min measured |
| Character without review | ~11-13 min | ~4 min | 3D-rendered (`drawn=false`): ~2.5 min GPU |
| Item | ~0.8 min (3 icons) | | potion/food/armour + 1.2 min 3D world sprite |
| Prop / building, drawn | ~1.3 min (3 drawings) | | 4 sides ~3.8 min |
| Tileset | ~0.7 min (3 textures) | | road / cliff (painted) ~2.1 min |
| Portrait set | ~0.5 + 0.3 per expression | | + ~0.45 when not a library character |
| Card | ~0.6 min (3 arts) | | |
| Background from words | ~0.35 min | seconds | procedural scenes: no GPU |
| Effect, UI kit, map, weather, water | none | seconds | effect `paint=true`: +0.6 min |
| Add animation / mocap / springs | none | 20-120 s | drawn characters bend the same drawings |
| Game 3D model | none (+1.5 min for drawn props/items) | 1.5-2.5 min | |
| Export into a game | none | seconds; Godot verify ~20-60 s | |

`plan_asset` gives the per-plan numbers (`gpu_minutes`, `after_ok_gpu_minutes`, `cpu_minutes`).

## Judging results (do it every time)

- **Look at the previews** the tools attach: `preview` (characters: one frame per direction;
  sheets: the frames), `animations` (each clip facing the viewer), `variations` (pick the best),
  `tries` (front drawings so far). Paths are in `previews` (under `<project>/.agent/previews`) and
  `preview_image(path)` shows any file larger.
- `quality.warnings` lists what to look at. `quality.subject_check` = the one-subject check of the
  chosen picture (ok / why: sheet, several things, scene, cut off). `quality.drawn` (drawn
  characters): IoU of each drawing against the model, colour distance to the front, sharpness;
  `worst` = the directions to redraw. Tilesets: `variations.checks` colour_miss / pattern_miss
  (0 = fits the terrain; bricks in a road, grids in water score high).
- The detectors can't judge meaning: a hut vs a barrel, leather boots drawn as a chest piece, a
  staff missing from one direction. Say what you see and redraw / change the words.
- A front that is wrong in SHAPE (missing weapon, wrong body) needs
  `character_change_words(..., new_model=true)`; a wrong colour or pose only needs
  `character_redraw_front`.

## What lands in the game (export_to_game)

- Godot: `res://<subdir>/<kind folder>/<asset>/` with SpriteFrames `.tres` (an animation per
  `action_DIR`, loop flags, per-frame ms), the sheet, `_n.png` normal map, `_glow.png`,
  `_facing.gd` (characters: `face(velocity)` + `act(&"walk")`), TileSets, theme.tres, card and map
  scenes, weather/water/parallax scenes. PNGs get a pixel-art `.import` (lossless, no mipmaps, no
  VRAM compression) unless Godot made one; `project.godot` gets the Nearest canvas filter if
  unset. Maps are standalone scenes re-rooted under the subdir.
- Unity: `Assets/<subdir>/...` sliced sheets with `.meta` (point filter, no compression), `.anim`
  clips, UI 9-slices, card prefab script; the manifest gets a `.meta` (TextAsset).
- `<subdir>/pixel_pipeline_manifest.json`:
  `assets[id] = {name, key, kind, folder, files{texture, sprite_frames, normal_map, glow_map,
  facing_script, tileset, theme, scene, model...}, res{...} (Godot res:// paths), frame{w,h},
  animations{action: {directions, frames, ms, loop}}, directions, anchor{x,y} (feet), ppu, tile,
  tileset{terrain, set...}, map{cols, rows}, source_hash}`; `by_name[key] = id`.

Load by name in Godot:

```gdscript
var man := JSON.parse_string(FileAccess.get_file_as_string("res://art/pixel_pipeline_manifest.json"))
func frames_of(key: String) -> SpriteFrames:
    return load(man.assets[man.by_name[key]].res.sprite_frames)
# $Knight.sprite_frames = frames_of("knight"); $Knight.play("walk_S")
```

(For exported builds add `*.json` to the export preset's non-resource filter.)

## The game runtime (use it: it is what makes the sprites PLAY well)

Every export ships `res://<subdir>/pp_runtime/` (Godot: `pp_motion.gd`, `pp_manifest.gd`,
`pp_sprite3d.gdshader`, `motion.json`; Unity: `motion.json`, same keys). The HD-2D board
(`hd2d.py` board.gd), its autoplay and the playtest games all run on it. Manifest fields it reads:

- `animations[clip].stride = {px, units, cycle_ms, source}` for walk/run(/sneak): how far ONE
  cycle of the clip carries the character (`units` = tiles at `world_ppu`; `px` = sheet pixels;
  source `rig` = measured from the skeleton's planted feet, `frames` = from the pixels, `default`
  = a guess from the body height). Play the clip at **speed / stride.units cycles per second**
  (frame = floor(phase x frames), phase += rate x delta): the planted foot then stays still on
  the ground. Pick walk below ~walk_stride x 1.9 cycles/s x 0.92, run above (keep the phase when
  switching); don't move faster than run_stride x 2.9/s (`pp_motion.max_speed`) or the feet skate.
- `px_scale` (1.0 = the library's 64 px-person scale; Neo16's 144x96 knight = 1.5), `world_ppu`
  (= ppu x px_scale: sprite px per tile), `body_px` (standing height). Size sprites by
  world_ppu (3D) or 1/px_scale at a WHOLE screen scale (2D) so every character shares one scale.
- `anchor{x,y}` = the feet. `motion` = the tunables below; `runtime` = the shipped files.

`pp_motion.gd` (static; `const M := preload("res://<subdir>/pp_runtime/pp_motion.gd")`):

| need | call | algorithm (port this for Unity) |
|---|---|---|
| ease in/out | `vel = M.approach(vel, wish, accel, decel, dt)` | move_toward at accel (speeding up/turning) or decel (slowing): 14 / 18 tiles/s^2 |
| path speed | `M.path_speed(v, want, remaining, accel, decel, dt)` | min(want, sqrt(2 decel remaining)): stops exactly at the end |
| turning | `yaw = M.turn(yaw, target, 900, dt)` | short-way rotation at <= 900 deg/s; the sprite picks one of 16 facings from yaw every frame, so a turn shows the ones between |
| feet | `M.locomotion(speed, strides, has_run, body_tiles, current)` -> {clip, rate} | rate = speed / stride; run above 0.92 x walk_top (x0.85 hysteresis) |
| stop | `M.settle_target(phase)` | on stop keep playing the walk to the next contact pose (phase 0 / 0.5, <= 0.16 s), then idle |
| steps | `M.hop_y(u, ya, yb, hop)` | smoothstep between the heights + arc clearing the higher edge by 0.14 over +-0.32 tiles of the border |
| free move over steps | `M.ground_follow(y, vy, ground, 0.07, 22, dt)` | ground = highest cell under a 0.22-tile foot circle; climb in ~0.07 s, drop with gravity |
| camera | `M.smooth_damp(pos, target, vel, 0.28, dt)` | critically damped spring (no overshoot); clamp the target inside the map |
| hits | `M.hit_time(clip, length)` | the blow lands at the strike frame (attack 0.5, cast 0.58 of the clip); then hit-stop 75 ms (freeze both), knockback 0.22 tiles, shake trauma 0.06-0.2 (offset = trauma^2 x 0.32, decay 2.6/s), anticipation 0.12 s lean back, follow-through 0.18 s |

`pp_sprite3d.gdshader` (3D billboard sprites, HD-2D): every pixel gets the depth of the FEET
(node origin) moved `depth_bias` 0.35 toward the camera -> never cut by the ground, walls,
stairs, props or other sprites behind its feet, still hidden by things in front; `pixel_snap`
moves the quad in whole screen pixels (no shimmer). Put the node origin at the feet (quad
`center_offset` = size.y/2 - (frame_h - anchor.y)/ppu). Draw HP bars / damage numbers in 2D
(`Camera3D.unproject_position`), not as 3D sprites: depth of field blurs those into blobs.
Unity port: same vertex trick (clip.z = clip-space z of the foot point x clip.w).

Library: new characters get 12-frame walk/run (`Style.locomotion_frames`, 12 or 16; other
clips 8) and planted-feet IK walk/run clips; `library.rerender_locomotion(ch, 12)` upgrades an
old character on the CPU (bends its existing drawings: no GPU; `drawnrig.no_new_drawings()`).

## Gotchas

- One GPU job at a time; don't queue hours of GPU work unattended (Kevin's laptop crashed after a
  multi-hour full-tilt run). Batch a handful, check, then continue.
- Drafts (characters waiting for an OK) are not exported, restyled or animated: approve first.
- `switch_project` / `create_project(switch=true)` change the app's current project too. Jobs
  remember their own project, so a later switch doesn't misdirect them.
- `delete_asset` moves to `<project>/.trash/` (move it back to undo).
- Tilesets keep the best texture; `pick_variation(n, job=<that job>)` rebuilds from another (also
  after a restart; the old set goes to .trash).
- Re-export overwrites what the exporter wrote before: edit art in the studio (or re-skin), not in
  the game folder. Godot `.import` / `.uid` and Unity `.meta` files are kept.
- No ComfyUI: GPU makers fail with "ComfyUI isn't running ..." (the server starts the local one
  at the default address itself); drawn characters fall back to the 3D render with a notice.
- The existing photo/sprite tools (`convert_photos`, `make_sprite_sheet`, `make_tileset` = cut an
  image into tiles, `turnaround`, `draw_sprite`, `preview_image`, ...) still work as before.

## CLI quick reference

```
pixel-pipeline project list | create NAME --type topdown-rpg --engine godot | switch NAME
pixel-pipeline project style [--set field=value ...] --json
pixel-pipeline make "rusty iron sword" [--kind item] [--param variations=4] [--dry-run] --json
pixel-pipeline character review|approve|redraw-front|words|pick|finish|redraw REF [VALUE] --json
pixel-pipeline library search [words] [--kind item] | show ID | tag ID --add approved | delete ID
pixel-pipeline library pick ID --index 2      (tilesets: --job JOBID)
pixel-pipeline jobs [list] | status ID --wait 60 | cancel ID | resume [IDs]
pixel-pipeline export --godot C:/Games/MyGame [--subdir art] [--assets ID ...] [--verify] --json
pixel-pipeline export --unity C:/Games/MyUnityGame
```

GPU commands wait for their job (progress on stderr; Ctrl+C cancels, keeping what's finished).
