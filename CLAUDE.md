# pixel-pipeline — context for Claude

Kevin "spark" Sparks's personal game-dev tool: photos and phone videos → pixel-art
sprites, animations, sheets, tilesets, `.aseprite`, Godot resources. Separate from
Pawgress. Kevin has a game-dev / pixel-art background, uses **Godot**, owns
**Aseprite**, works on Windows (RTX 4080 Laptop, 12 GB VRAM) and an Android phone. He
wants terse answers and fixes over polish.

## Decisions (2026-10-07)

- **Hybrid:** the free algorithmic converter is the core. AI is a later, optional,
  pluggable "style" step, and its output always goes back through the converter so it
  lands on the exact grid and palette. Next AI step: test PixelLab (40 free gens)
  against Retro Diffusion API (~$0.01/img) on Kevin's photos. Local models
  (Qwen-Image-Edit / Flux Kontext via ComfyUI) fit his 12 GB GPU.
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
  sketches with `draw_sprite`, then `generate_sprite(init_image=...)`. Never verified on a
  real GPU from the cloud container, so Kevin's first run is the real test.
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
- Private repo with limited Actions minutes: keep workflows lean.

## Temporary: built in the Pawgress repo (clean up!)

Kevin couldn't log in to GitHub on 2026-10-07, so (with his OK) the Android APK was built
by pushing local branch `pawgress-build` (main + a workflow tweak) to
`sparkskevin96-cell/pawgress` branch `ccr-15172514-6ugcr0`, which published the
pre-release `pixel-pipeline-android-temp` there. Once `sparkskevin96-cell/pixel-pipeline`
exists: push `main` there, then delete that Pawgress branch, release and tag.

## Next up

AI style pass (PixelLab / Retro Diffusion / ComfyUI), palette ramps and shading, batch
tools in the desktop app, video import on phone.
