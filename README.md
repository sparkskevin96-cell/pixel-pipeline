# pixel-pipeline

Turn photos into game-ready pixel art: sprites, sprite sheets, tilesets and editable
`.aseprite` files. People, animals, objects and backgrounds, with control over size,
pixel density, palette, dithering and outlines.

```
photo ──► cut out subject ──► crop + fit ──► edge-preserving pixelize ──► palette + dither ──► outline ──► PNG / sheet / tileset / .aseprite
          (rembg, local)                      (PixelOE-style, no AI)        (auto / classic /
                                                                             Lospec / your file)
```

No AI service is required. Background removal runs locally, and everything else is
classical image processing, so results are predictable and free.

## Install (Windows)

Python 3.11–3.13 from python.org, then in the repo folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[ui,cutout]"
```

- `ui`: the desktop app (PySide6)
- `cutout`: background removal (rembg on CPU, ~1–3 s per photo). The first run
  downloads a ~170 MB model to `~/.u2net`.
- `cutout-gpu`: the same on your NVIDIA GPU. Needs CUDA 12 + cuDNN 9 installed;
  CPU is usually fast enough.

## Desktop app

```powershell
pixel-pipeline-ui            # or: pixel-pipeline-ui path\to\photo.jpg
```

Drop a photo on the window. Pick a subject type (it loads that preset), then
tweak anything on the left; the sprite re-renders live. File → Export PNG
(Ctrl+E) or Export .aseprite (Ctrl+Shift+E).

## Command line

```powershell
# One photo -> 32x32 dog sprite, PICO-8 colours, 8x preview PNG + .aseprite
pixel-pipeline convert rex.jpg -s animal --palette pico-8 --scale 8 --aseprite -o rex.png

# A burst of photos -> frames that share ONE palette, then an animated sheet
pixel-pipeline convert walk_*.jpg -s animal -W 48 -H 32 -o frames/
pixel-pipeline sheet frames/*.png -o rex_walk.png --tag walk --duration 120 --aseprite

# Background: 16:9, dithered, seamless, then cut into a Tiled tileset + map
pixel-pipeline convert forest.jpg -s background -W 256 -H 144 --tileable -o forest.png
pixel-pipeline tileset forest.png --tile 16 -o forest_tiles.png

pixel-pipeline palettes                      # list built-ins
pixel-pipeline convert me.jpg -s person --palette lospec:resurrect-64
pixel-pipeline convert --help                # every option
```

## Settings

| Setting | CLI | What it does |
|---|---|---|
| Subject preset | `-s person\|animal\|object\|background` | Loads sensible defaults for that kind of photo |
| Size | `-W 32 -H 48` | Output size in sprite pixels. `-H 0` keeps the photo's aspect |
| Fit / anchor | `--fit contain\|cover\|stretch`, `--anchor bottom\|center` | Pad or crop to the size; `bottom` keeps feet on the floor |
| Background removal | `--bg-removal / --no-bg-removal`, `--model` | Cut the subject out (`u2net_human_seg` for people, `birefnet-general` for best edges) |
| Crop | `--crop / --no-crop` | Crop to the subject before fitting |
| Pixel density | `--patch 8` | Photo pixels sampled per sprite pixel |
| Detail | `--detail 0-4` | Edge thickening so linework survives the shrink |
| Downscale | `--downscale contrast\|median\|center\|area` | How each block becomes one pixel |
| Palette | `--palette auto\|pico-8\|sweetie-16\|endesga-32\|gameboy\|1bit\|lospec:<slug>\|file.gpl/.hex/.png` | Colour scheme |
| Colours | `--colors 16` | Palette size when `auto` |
| Dither | `--dither none\|floyd\|bayer2\|bayer4\|bayer8`, `--dither-strength` | Gradients with few colours |
| Outline | `--outline / --no-outline`, `--outline-color '#1a1c2c'` | 1px silhouette outline (auto = darkest palette colour) |
| Tone | `--brightness --contrast --saturation` | Pre-adjust the photo |
| Seamless | `--tileable` | Make backgrounds/textures wrap |
| Export | `--scale 8`, `--aseprite` | Upscaled PNG preview, editable Aseprite file |

## Outputs

- **PNG**: 1x (true pixels), plus optional nearest-neighbour upscale.
- **.aseprite**: written directly from the file spec, so Aseprite isn't needed to
  make it. Sprites that fit in ≤255 colours are saved **Indexed**, with index 0
  transparent, so palette swaps work in Aseprite. Sheets become animated files
  with a tag.
- **Sprite sheet + .json**: Aseprite/TexturePacker "json-array" format, which
  Godot, Unity, GameMaker and Phaser importers read.
- **Tileset + .tmj**: unique tiles plus a Tiled map that rebuilds the image; Godot,
  LDtk and most engines import Tiled maps.

## Python API

```python
from pixel_pipeline import convert, convert_batch, settings_for
from pixel_pipeline.aseprite import write_aseprite

s = settings_for("animal", width=48, height=32, palette="endesga-32", dither="bayer4")
sprite = convert("rex.jpg", s)
sprite.save("rex.png", scale=8)
write_aseprite("rex.aseprite", [sprite.rgba], sprite.palette)
```

## Tests

```powershell
pip install -e ".[dev,ui]"
pytest
```

## Roadmap

1. **Done:** converter core, CLI, desktop app, sheets, tilesets, `.aseprite`.
2. AI style pass (pluggable, optional): PixelLab (animation, 4/8-direction
   rotations), Retro Diffusion API, local models via ComfyUI on the GPU. Whatever
   the AI returns goes back through the converter, so it always lands on your exact
   grid and palette.
3. MCP server so Claude can drive the pipeline ("turn these photos into 32x32 idle
   sprites with PICO-8").
4. Video → animation frames, selective outlines (sel-out), palette ramps,
   batch/sheet tools in the desktop app.

## Credits

The edge-preserving pixelization is adapted from
[PixelOE](https://github.com/KohakuBlueleaf/PixelOE) by Shih-Ying Yeh (Apache-2.0).
See `THIRD_PARTY_NOTICES.md`.
