# pixel-pipeline — context for Claude

Kevin "spark" Sparks's personal game-dev tool: photos and phone videos → pixel-art
sprites, animations, sheets, tilesets, `.aseprite`, Godot resources. Separate from
Pawgress. Kevin has a game-dev / pixel-art background, uses **Godot**, owns
**Aseprite**, works on Windows (RTX 4070 Laptop, 8 GB VRAM; corrected 2026-10-11, earlier notes said 4080/12 GB) and an Android phone. He
wants terse answers and fixes over polish.

## Decisions (2026-10-07)

- **Hybrid:** the free algorithmic converter is the core. AI is optional and pluggable,
  and its output always goes back through the converter (or the palette snap) so it lands on
  the exact grid and palette. PixelLab vs Retro Diffusion comparison is built (Prompt tab),
  prompt-only by Kevin's choice; he runs it with his own keys. Local models
  (Qwen-Image-Edit / Flux Kontext via ComfyUI) need fp8/GGUF on his 8 GB GPU.
- **Desktop app, Python + Qt (PySide6).** Windows first.
- **No PyTorch in the core.** PixelOE's algorithm is reimplemented in numpy/OpenCV
  (`pixelize.py`, credited in THIRD_PARTY_NOTICES).
- **Background removal = rembg's ONNX models run directly with onnxruntime**
  (`cutout.py`). No rembg dependency (it breaks PyInstaller). Masks match rembg
  bit-for-bit. Models are cached in `~/.rembg/models`.
- **`.aseprite` is written directly** from the official spec. Indexed when ≤255
  colours with binary alpha (index 0 = transparent), else RGBA. Verified against the
  `ase-parser` npm package and the round-trip reader in `tests/test_aseprite.py`. The
  web app has a JS port of the same writer.
- **Godot 4 resources** (`godot.py`): SpriteFrames and TileSet `.tres` with relative
  texture paths. Verified by loading them in Godot 4.7 headless (set `GODOT_BIN` to run
  that test).
- **Phone:** `web/index.html` is a self-contained JS port of the converter (no AI
  cut-out; colour-based Auto/Tap background removal). It is published as a claude.ai
  artifact (Goc5YHwKAowqd1bqDDCS9m) and is ALSO the UI of the Android app. `mobile/` is a
  Flutter shell that loads it and adds native pickImage / ML Kit segment / save through a
  JS bridge (`window.PixelHost` channel + `window.__pixelHostReply`). Keep that bridge
  contract stable. The file is a fragment (no doctype/html/head) because the artifact
  host wraps it; the app wraps it in `tool/sync_web.dart`. Re-publish the artifact after
  editing it.
- **Retro systems** (`systems.py`): `system` setting (nes, snes, gameboy, gbc, sms,
  genesis, gba, pico8). It applies original hardware palettes or colour depths, colours
  per sprite, sizes per subject and outline style over the subject preset; explicit
  settings still win. `look`: `readable` (default; 48x64 people, multi-palette colour
  budgets) or `authentic` (one era sprite, e.g. NES 16x24 in 3 colours). Kevin found the
  authentic sizes unrecognizable for photos, hence readable as the default. Game Boy
  (`mono`) maps photos by brightness (`tone_map`) onto 4 evenly spread DMG shades. The
  JS port in `web/index.html` must stay in sync.
- **Turnarounds** (`turnaround.py`, CLI `turnaround`, MCP `turnaround`): 4/8/16 facing
  directions. 3D models (trimesh, `3d` extra) are point-splat rendered with an orthographic
  camera (`elevation`), one scale/palette; or photos/video walking around a subject
  (`views_from_capture`: start in front, walk right; `mirror` for half captures).
  Directions follow Godot's Vector2.angle (E, ESE, SE ... clockwise); `--godot` writes an
  animation per direction plus `_facing.gd`.
- **Animated directions** (`anim3d.py`, `turnaround --animate`, MCP `turnaround(animate=)`,
  Model Studio "Animate"): own dependency-free glTF reader (skins = linear blend skinning,
  LINEAR/STEP/CUBICSPLINE TRS animations, baseColorTexture/Factor/COLOR_0 in linear space,
  uv origin top-left, alpha MASK/BLEND cut). Points are sampled once on the rest pose and
  re-posed per frame (no colour flicker); `render_points` frames all poses alike. Loop =
  start/end vertices match after removing mean offset; in-place removes the linear centroid
  drift. Later files add animations by node name (Mixamo per-animation downloads); generic
  names (mixamo.com, Take 001) take the file stem. `--colours-from` re-colours by shape
  (voxel grid) since Mixamo drops vertex colours; `model --obj` / Save all writes the .obj
  to upload. Verified on Khronos Fox/CesiumMan/RiggedSimple and in Godot 4.7 headless.
- **Auto-rig** (`autorig.py`, CLI `rig` / `turnaround --auto-rig`, MCP turnaround(auto_rig=),
  Directions tab checkbox, Prompt to Model "Animate"), 2026-10-09, toward "build PixelLab"
  locally: front silhouette grid (points -> mask, z range per cell) -> Zhang-Suen thinning ->
  crossing-number ends/junctions (8-neighbour staircases fooled plain degree counts) -> head
  / feet / hands from ends, pelvis from the crotch row (the skeleton forks too high), shoulders
  where arm paths leave the spine past the chest half-width; fallbacks = standard proportions
  (notes say which). Separate silhouette pieces are bridged. Skin = inverse distance^4 to bone
  segments, top 4. 17 joints, world-aligned rest, animations sampled 25 keys/clip; extras.loop
  marks one-shots (anim3d honours it). Verified on Khronos CesiumMan / RiggedFigure /
  BrainStem and box figures; Godot imports 17 bones + 6 animations (it doesn't loop "walk"
  for 3D use unless set in import settings; sprite exports carry loop flags).
- **Four-legged rig** (`quadruped.py`, `rig --animal`, `turnaround --auto-rig -s animal`, MCP
  turnaround(animal=), Prompt to Model Animate with Subject animal), 2026-10-11 for Kevin's dog:
  long axis = longer horizontal extent; head = end with more mass above the back line (override
  `head=`); side silhouette: belly row, legs below it split front/back at the gap, left/right by
  depth; tail = thin columns at the back. Turned head +z, left +x (same as humanoid). 19 joints
  (hips spine chest neck head tail1 tail2 + upper/lower/paw x front_l front_r back_l back_r);
  legs bind only own side/half below the body. Clips: idle walk(4-beat) run sit happy lie_down
  death, legs by 2-bone IK to planted feet. autorig's glTF writing is shared (`write_skinned`).
- **Your own photo in Prompt to Model** (2026-10-11): From a picture -> cut out -> clean-up
  window (`ui/picture_editor.py`: rub out / bring back, Ctrl+Z) -> build. `prompt_to_model(
  mirror_back=, colours=)` / CLI `model --mirror-back --colours "black, white"`: picture snapped
  to the colours (Lab, lightness stretched to the palette's first: shaded white fur stays white;
  morphological open removes glare/hairs), model vertices snapped + neighbour majority. Repaint
  = same shape, no GPU. Photos load with EXIF rotation (phone pictures are stored sideways).
  PySide teardown crashes in tests came from a lambda capturing the window in a signal and
  deleteLater without an event loop: connect methods, hand dialogs back with setParent(None).
- **Online AIs** (`online.py`, CLI `ai`, MCP `online_sprites`, desktop Prompt tab), 2026-10-09:
  Kevin wants NEW sprites from prompts and does NOT want his photos sent to any service, so
  the online providers are prompt-only. PixelLab REST v2 (Bearer; create-image-pixen /
  pixflux / bitforge sync, generate-image-v2 queued via /background-jobs/{id}; base64 may be
  raw RGBA) and Retro Diffusion v2 (X-RD-Token; POST /inferences -> poll
  /inferences/tasks/{id} -> result.base64_images; styles rd_fast/rd_plus 64-384, rd_pro
  16-256). Shapes from PixelLab's SDK, a recent v2 CLI (aviorstudio/pixellab-cli) and RD's
  api-examples repo; the cloud container's network blocks both hosts, so they were only tested
  against fakes (tests/test_online.py). Results at sprite size are snapped to the palette
  (`to_sprite`, no resampling). Keys: env or ~/.pixel-pipeline/keys.json.
- **App layout (2026-10-12, Kevin: "make it feel intuitive")**: tabs in the order you make a
  game: `1 Characters` (ModelStudio + Character Library list on the left; Kind person/animal/
  object, Size words not numbers, Name, Add to library = rig + all clips x16 + normals at the
  library scale; More settings hides the raw numbers), `2 World` (map tiles), `3 Game`
  (`ui/game_tab.py`: tick up to 6 per team from the library -> hd2d project in
  ~/Pixel Pipeline/Games/<name>, godot --import, Play launches Godot found via GODOT_BIN/PATH/
  winget), `Sprite` (From a photo | From words, basics up front, More settings folds the rest),
  `More tools` (Directions from photos, Team colours, Portrait, Simple 2D demo). Per-tab status
  hints; `MainWindow.show_tool(w)` / `current_tool()` handle the nested tabs. **Character
  Library** = `library.py`, ~/Pixel Pipeline/Characters/<slug>/ (character.json, model.glb,
  picture.png, rigged.glb, sheet.* + sheet_n.png, thumb.png); PIXEL_LIBRARY overrides (tests use
  a temp one via conftest). SIZES: person small/normal/large 80x52/96x64/112x76, animal
  64x28/72x34/88x44.
- **Desktop app** (`ui/app.py` MainWindow = QTabWidget): Sprite (the original live
  converter), Directions / Teams / Portrait / Map tiles / Godot demo (`ui/tools.py`, each a
  `ToolTab` from `ui/common.py`: slow work on a QThread via `run(job, on_done)`, `sync=True`
  in tests), Prompt to Model (`ModelStudio` embedded). Icon: the demo knight
  (`ui/icon.png`, `packaging/icon.ico`). Windows exe = PyInstaller `packaging/pixel-pipeline.spec`.
- **Godot demo** (`demo.py`, CLI `demo`, MCP `godot_demo`): writes a Godot 4 project
  (640x360 x3, integer scale, gl_compatibility): terrain map (cells set at runtime from
  map/map.json; water + half-visible edge rows blocked), units created in code (`DemoUnit`
  in unit.gd: AnimatedSprite2D + generated facing script + duplicated team material), A*
  click-to-move, spent/next turn. Default unit = `mannequin()`, a rigged knight built in
  code with `GlbWriter` (idle + walk). Verified in Godot 4.7 headless (logic) and under
  xvfb (screenshots). `make_demo` returns (files, start-screen preview).
- **Prompt to Model** (`model3d.py`, CLI `model`, MCP `prompt_to_model`, Qt
  `ui/model_studio.py` = `pixel-pipeline-model` / Ctrl+M / `--model`): prompt -> SDXL front
  picture (no pixel LoRA) -> ComfyUI native Hunyuan3D 2.1 shape (workflow mirrors ComfyUI's
  own blueprint; history key "3d") -> `paint_from_picture` (finds orientation by silhouette
  IoU over 24 rotations, projects the picture on front-visible vertices, blurred
  through-projection behind, optional back picture) -> turnaround. Untested on a real GPU;
  the fake ComfyUI in tests/test_generate.py knows the 3D nodes. ComfyUI source for node
  names: github.com/Comfy-Org/ComfyUI comfy_extras/nodes_hunyuan3d.py, nodes_save_3d.py.
- **Teams + portraits** (`teams.py`, CLI `teams` / `portrait`, MCP `team_colours` /
  `portrait`): team family = dominant saturated hue (yellows 45-110 weighted down: trim),
  recoloured in LCh keeping lightness; Godot palette-swap shader + material per team (verified
  by rendering in Godot under xvfb-run + Mesa: `xvfb-run -a godot --rendering-driver opengl3`).
  Portraits: YuNet face box (downloaded on first use), else the subject's top.
- **Map tiles** (`maptiles.py`, CLI `maptiles`, MCP `map_tiles`): built-in procedural
  seamless terrains (periodic noise) or photo/prompt textures; corner masks are smoothstep
  bilinear + shared periodic jitter so tiles agree on edges; base variants only add interior
  details (same edges). TileSet .tres format copied from Godot's own ResourceSaver output;
  alternatives 0:0/1-15 = grass next to empty cells (Godot treats empty as a terrain).
- **Side-scroller + isometric tiles** (`tilesets.py`, `maptiles --style`, MCP map_tiles(style=),
  Map tiles tab Style): side = ground texture + top texture capped on open tops (periodic cap
  depth so neighbours line up), 16 tiles by open sides (bits top 1, right 2, bottom 4, left
  8; mask m at atlas (m%4, m//4)) + 4 one-way platform pieces in row 4; Godot terrain mode 2
  (Match Sides) with side peering bits, physics_layer_0 boxes from the grass top (lip rows of
  air). Iso = W x W/2 diamonds sampled from the seamless texture (u=(a+b)/2, v=(b-a)/2),
  blocks W x W (sides x0.72 / x0.55, grass/snow sides show dirt); TileSet tile_shape 1,
  tile_layout 5 (diamond down: map (1,0) -> (W, H)), blocks in source 1 with
  texture_origin (0, H/2). Godot verified: set_cells_terrain_connect picks the right side
  tiles; xvfb render shows blocks on their cells.
- **Sprite scale research** (`docs/SPRITE_SCALE.md`, interactive page artifact
  Jv1VrzgNjgmiH4LgAQg1SR): character height as a share of screen height for 41 classic
  games by genre, and the matching sprite heights. Kevin decided all his games render at
  1080p (4K = exactly 2x), so valid pixel-art bases are 320x180 (x6), 384x216 (x5),
  480x270 (x4) and 640x360 (x3).
- **Prompt → sprite** has three routes. (1) Claude draws it: `drawing.py`
  (`sprite_from_rows`), MCP `draw_sprite`, and the web app's "Draw from a prompt" via the
  artifact `sample` capability, which is hidden in the APK. (2) Local GPU: `generate.py`
  (ComfyUI HTTP client, SDXL + Pixel Art XL LoRA; setup in `docs/LOCAL_AI.md`), CLI
  `generate`, MCP `generate_sprite`, desktop File → Generate. (3) The mix: Claude
  sketches with `draw_sprite`, then `generate_sprite(init_image=...)`. Verified on Kevin's
  laptop 2026-10-11: ComfyUI 0.39.0 portable at `C:\AI\ComfyUI_windows_portable` (start
  `run_nvidia_gpu.bat`) with SDXL base + Pixel Art XL + fp16-fix VAE (used automatically when
  installed); 2 sprites in 53 s including the first model load. No LCM LoRA yet. Kevin asked
  (2026-10-11) for no ComfyUI window: `ComfyClient.ensure_running` starts it hidden
  (CREATE_NO_WINDOW, log ~/.pixel-pipeline/comfyui.log) only at the default address, and
  atexit stops only processes we started (`start_comfyui` / `stop_started_comfyui`).
- **HD-2D tactics demo** (`hd2d.py`, `diorama.py`, CLI `hd2d`), 2026-10-11, toward Kevin's goal
  (Triangle Strategy-style games): one Godot 4.6 project, menu + three looks of one map.
  `diorama.Diorama` = height levels (1 level = half a tile), terrain per cell, props, units;
  `harbour()` = stone tiers + stairs + cobble quay + pier over water. Real 3D (`board.gd`, mode
  real): terrain meshed at runtime with SurfaceTool from map.json, units = quads with a
  Y-billboard shader drawing a sheet frame, NORMAL_MAP from `<sheet>_n.png`, direction picked
  from camera-relative facing; fov 22 telephoto, CameraAttributesPractical near+far DOF
  (tilt-shift), glow, ACES, SSAO, sunset sun + torch OmniLights, water shader with glints; Q/E
  turn 45°. Fake 3D = same board in a SubViewportContainer (stretch_shrink 3 -> 640x360), ortho
  camera snapped to whole pixels, sprites fully camera-facing. Images load with
  Image.load_from_file (no import needed); the iso look needs `godot --import` once (the editor
  does it). Units: `add_unit` writes `<name>.unit.json` (frame size, feet row, ppu 42 = a
  64 px person stands ~1.5 tiles).
- **Tactics rules** (all three looks, 2026-10-11): `diorama.CLASSES` gives hp/attack/defence/
  move/jump/reach per unit name (Unit fields override); `diorama.damage` = attack - defence//2,
  x5/4 from a higher level, min 1, no dice. Select -> purple move range (own move/jump) -> move
  or stay -> red cells = enemies within Manhattan reach -> attack (attack/hurt/death clips, else
  lunge/flash/fade), HP bars, damage numbers; done units grey, turn passes when the team is done,
  banner when one team is left. `place_units(d, blue, red)` = roster. ONE SCALE for all sheets:
  frames wide enough that height binds (people 96x64; dog 64x28, wolf 72x34, bear 88x44), else a
  spear or bow shrinks its owner. 10 GPU characters in C:/Users/spark/Pictures/PixelPipeline/chars
  (make_chars.sh / rescale.sh).
- **Sprite normal maps** (`normals.py`, `turnaround/model --normals`): camera-space normal of the
  front-most splat per pixel, OpenGL (y up), cropped/resampled with the colour sprite's own
  geometry (`_fit_box`), averaged+renormalised, never palette-snapped; `<sheet>_n.png` packed
  like the sheet. Tests share ONE QApplication (`conftest.qt_app`): a second QApplication after
  the first died with a test crashed the Windows CI run at random.
- **Isometric tactics look** (`iso_scene.py`, CLI `iso-demo`), 2026-10-11, one of the three looks
  of the tactics demo over `diorama.py`: `write_iso_scene(d, project, {name: sheet.png})` writes
  `iso/` (+ map/map.json if absent, project.godot only if absent). One pre-rendered column per
  cell (top + sides from its height down to level -2; tile W, level = W/4 px; side texture 1:1,
  line per level), painter's order by z_index = (col+row)*8 + layer. GDScript range = path_cost
  (test compares them in Godot). Grid +col faces SE, +row SW. Verified on Godot 4.6.1 with real
  GPU screenshots (`Pictures/PixelPipeline/hd2d_iso_test/shot_*.png`, `shot.gd` makes them).
- The container that built this couldn't reach dl.google.com, so APKs build only in
  GitHub Actions.

## File map

- `settings.py`: `Settings` + per-subject `PRESETS`; `settings_for()` (None overrides =
  keep preset; height 0 = auto aspect)
- `pipeline.py`: load → isolate → prepare (crop, fit, fill, pixelize) → finish (palette,
  dither, cleanup, outline); `convert`, `convert_batch`, `sprites_from_rgba` (shared
  palette)
- `pixelize.py` (outline expansion, contrast downscale, tileable), `quantize.py`,
  `palettes.py`, `finishing.py` (orphan cleanup, sel-out), `color.py`
- `video.py` (frames → aligned sprites, track/fixed), `animate.py` (breathe/bob/sway)
- `export.py` (sheets + JSON, tilesets + Tiled .tmj, `write_animation`, `load_sprite`
  undoes upscaled exports), `aseprite.py`, `godot.py`
- `cli.py` (convert, video, idle, sheet, tileset, palettes), `ui/app.py` (desktop),
  `mcp_server.py` (MCP tools for Claude; needs mcp>=2)
- `systems.py` (retro looks), `drawing.py` (text-row sprites), `generate.py` (ComfyUI)
- `depth.py` + `parallax.py`: depth-based parallax layers + Godot Parallax2D scene
- `packaging/` (PyInstaller spec), `.github/workflows/ci.yml` (tests on win+linux, exe on
  main), `.github/workflows/android.yml` (APK + `android-latest` release)

## Conventions

- Images are numpy uint8 RGBA (H, W, 4); sprite alpha is binary (0/255).
- Perceptual maths in Lab via `color.py`.
- Run `pytest` before pushing; the UI test runs headless (`QT_QPA_PLATFORM=offscreen`).
- Keep workflows lean (Actions minutes are free while the repo is public).

## Repo and builds

GitHub: `sparkskevin96-cell/pixel-pipeline` (created 2026-10-10 private, made **public**
by Kevin 2026-10-11 so the app updates itself with no token). Pushes to `main`
run CI (tests on Linux + Windows) and publish the Windows app as the `windows-latest`
pre-release; changes under `mobile/` or `web/` publish the APK as `android-latest`. The
Claude GitHub app can't dispatch workflows (403), so trigger by pushing.

**Install + self-update** (`ui/updater.py`, windows in `ui/update_ui.py`), 2026-10-10, after
Kevin opened an old copy: CI writes `build.json` (run number, sha, date) into the app and
uploads it as `version.json` on the release. A downloaded copy offers to install itself to
`%LOCALAPPDATA%\Programs\PixelPipeline` (+ desktop/Start menu shortcuts) or opens the
installed copy if that's as new. Updating = download zip → unpack to
`%LOCALAPPDATA%\PixelPipeline\update` → PowerShell script waits for the pid, robocopies
(`_internal` mirrored) and restarts with `--updated`. Public repo => no token. If it is ever made
private again, the check needs a fine-grained token (Contents: read) saved as "github" in
keys.json (Help → Updates token). CI's "Update test" step updates a build-0 copy from the real zip; the publish step
checks the release through the API.

**Version number** = the CI run number of ci.yml (`build.json`), shown bottom-right in the
app, in its title and in the release title. Kevin asked (2026-10-10): after every push that
rebuilds the Windows app, wait for CI and tell him the new version number. Docs-only pushes
don't rebuild (paths-ignore), so no new version then.

Before the repo existed (2026-10-07..09), builds ran from branch `ccr-15172514-6ugcr0` of
`sparkskevin96-cell/pawgress` (local branch `pawgress-build`), with pre-releases
`pixel-pipeline-android-temp` and `pixel-pipeline-windows-temp` there. Cleaned up
2026-10-10 with Kevin's OK: both pre-releases and tags deleted, and that branch reset to
Pawgress `main` (this session can't delete branches).

## Next up

AI style pass (PixelLab / Retro Diffusion / ComfyUI), palette ramps and shading, batch
tools in the desktop app, video import on phone.
