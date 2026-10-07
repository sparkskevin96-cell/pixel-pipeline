# pixel-pipeline — context for Claude

Kevin "spark" Sparks's personal game-dev tool: photos → pixel-art sprites, sheets,
tilesets, `.aseprite`. Not a product (yet), and separate from Pawgress. Kevin has a
game-dev / pixel-art background, wants terse answers and fixes over polish.

## Decisions (2026-10-07)

- **Hybrid:** free algorithmic converter is the core. AI is a later, optional,
  pluggable "style" step, and its output always goes back through the converter
  so it lands on the exact grid and palette.
- **Desktop app, Python + Qt (PySide6).** Windows first.
- **Converter first, AI later.** Then test PixelLab (40 free gens) against Retro
  Diffusion API (~$0.01/img) on Kevin's own photos before wiring one in. Local
  models are viable: Kevin has an RTX 4080 Laptop (12 GB VRAM), enough for
  Qwen-Image-Edit / Flux Kontext quantized via ComfyUI.
- Kevin **owns Aseprite** (Aseprite MCP servers such as pixel-mcp are an option
  for Claude-driven cleanup).
- **No PyTorch in the core.** PixelOE's algorithm is reimplemented in numpy/OpenCV
  (`pixelize.py`, credited in THIRD_PARTY_NOTICES). rembg is an optional extra.
- **`.aseprite` is written directly** from the official spec
  (`docs/ase-file-specs.md` in aseprite/aseprite). Indexed when ≤255 colours with
  binary alpha (index 0 = transparent), else RGBA. Verified against the
  independent `ase-parser` npm package, plus the round-trip reader in
  `tests/test_aseprite.py`.

## File map

- `settings.py`: `Settings` dataclass + per-subject `PRESETS`; `settings_for()`
  (None overrides = keep preset; height 0 = auto aspect)
- `pipeline.py`: load → isolate → prepare (crop, fit, fill, pixelize) → finish
  (palette, dither, outline); `convert`, `convert_batch` (shared palette)
- `pixelize.py`: outline expansion, contrast downscale, tileable cross-fade
- `quantize.py`: nearest-in-Lab, Bayer, Floyd–Steinberg
- `palettes.py`: built-ins, .gpl/.hex/image files, `lospec:<slug>` (cached in
  ~/.cache/pixel-pipeline), auto k-means
- `cutout.py`: rembg wrapper (`CutoutUnavailable` if the extra isn't installed)
- `aseprite.py`, `export.py` (sheet JSON, Tiled .tmj), `cli.py`, `ui/app.py`

## Conventions

- Images are numpy uint8 RGBA (H, W, 4); sprite alpha is binary (0/255).
- Perceptual maths in Lab via `color.py`.
- Run `pytest` before pushing; UI test runs headless (`QT_QPA_PLATFORM=offscreen`,
  needs libEGL on Linux).

## Next up (roadmap)

AI style pass (PixelLab / Retro Diffusion / ComfyUI), an MCP server for Claude,
video → frames, sel-out outlines, batch and sheet tools in the UI.
