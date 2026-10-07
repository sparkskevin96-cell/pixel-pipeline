# pixel-pipeline

Turn photos and phone videos into game-ready pixel art: sprites, animations, sprite
sheets, tilesets and editable `.aseprite` files, with Godot 4 resources ready to drop
in. People, animals, objects and backgrounds, with control over size, pixel density,
palette, dithering and outlines.

```
photo / video ──► cut out subject ──► crop + fit ──► edge-preserving pixelize ──► palette + dither ──► cleanup + outline
                  (local ONNX model)                  (PixelOE-style, no AI)        (auto / classic /       (solid or sel-out)
                                                                                     Lospec / your file)
      ──► PNG · sprite sheet + JSON · .aseprite · Godot SpriteFrames / TileSet · Tiled map
```

No AI service is required. Background removal runs locally, and everything else is
classical image processing, so results are predictable and free.

## Install (Windows)

**Easiest:** download the ready-made app from GitHub. Repo → Actions → latest "CI" run on
`main` → Artifacts → `PixelPipeline-windows`. Unzip the whole folder and run
`PixelPipeline.exe`. The exe is unsigned, so Windows SmartScreen will warn: More info →
Run anyway. Keep the exe inside its folder. `pixel-pipeline.exe` in the same folder is
the command line.

**From source** (needed for GPU background removal and the MCP server). Install
Python 3.11–3.13 from python.org, then in the repo folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[ui,cutout,mcp]"
```

- `ui`: the desktop app (PySide6)
- `cutout`: background removal (onnxruntime on CPU, ~1–3 s per photo). The first use
  downloads a ~170 MB model to `~/.rembg/models` (shared with rembg, if you have it).
- `cutout-gpu`: the same on your NVIDIA GPU (needs CUDA 12 + cuDNN 9; CPU is
  usually fast enough)
- `mcp`: the MCP server, so Claude can drive everything (see below)

iPhone `.heic` photos work out of the box.

## On your phone

- **Web app**, works now: `web/index.html`, published privately at
  https://claude.ai/artifact/Goc5YHwKAowqd1bqDDCS9m. Take or choose a photo, pick a
  subject, palette, size, dither and outline, then save a PNG, a pack (PNGs + `.aseprite`
  + `.gpl` palette) or an idle animation (sheet + `.json` + `.aseprite` + Godot `.tres`).
  Background removal is colour-based (Auto, or tap the background).
- **Android app** (`mobile/`): a Flutter shell around the same web app. It adds
  on-device AI background removal (Google ML Kit), native camera and gallery, and saving
  to your gallery. GitHub Actions builds the APK. Phone → repo → Releases →
  `android-latest` → download the APK → allow installs from your browser → open.

## Desktop app

```powershell
pixel-pipeline-ui            # or: pixel-pipeline-ui path\to\photo.jpg
```

Drop a photo on the window. Pick a subject type (it loads that preset), then
tweak anything on the left; the sprite re-renders live.

- File → Export PNG (Ctrl+E) / Export .aseprite (Ctrl+Shift+E)
- File → Export idle animation (Ctrl+I): breathe / bob / sway loop as sheet +
  `.aseprite` + Godot `.tres`
- File → Open video (Ctrl+Shift+O): pick frames and a time range. The animation
  plays in the preview; File → Export video animation (Ctrl+Shift+A) writes sheet +
  `.aseprite` + Godot `.tres`.

## Command line

```powershell
# One photo -> 32x32 dog sprite, PICO-8 colours, 8x preview PNG + .aseprite
pixel-pipeline convert rex.jpg -s animal --palette pico-8 --scale 8 --aseprite -o rex.png

# Phone video of the dog walking -> 8-frame walk cycle in place, one shared palette
pixel-pipeline video rex_walk.mp4 -o rex_walk.png -W 48 -H 32 --start 1.2 --end 2.0 --tag walk --aseprite --godot

# Idle loop from any sprite (upscaled exports are detected and shrunk back to 1x)
pixel-pipeline idle rex.png -o rex_idle.png --style breathe --aseprite --godot

# Several photos -> frames sharing ONE palette -> animated sheet
pixel-pipeline convert pose_*.jpg -s person -o frames/
pixel-pipeline sheet frames/*.png -o hero_wave.png --tag wave --duration 120 --aseprite --godot

# Background: 16:9, dithered, seamless, then a tileset + Tiled map + Godot TileSet
pixel-pipeline convert forest.jpg -s background -W 256 -H 144 --tileable -o forest.png
pixel-pipeline tileset forest.png --tile 16 -o forest_tiles.png --godot

pixel-pipeline palettes                      # list built-ins
pixel-pipeline convert me.jpg -s person --palette lospec:resurrect-64 --outline-style selout
pixel-pipeline <command> --help              # every option
```

### Video tips

- Film the subject side-on and fill the frame. A plain background helps the cut-out.
- For a seamless loop, pick `--start`/`--end` around exactly one stride (paw back
  to the same position). Frames are spread evenly with the end excluded.
- `--align track` (default) keeps the subject centred on the floor, so it walks in
  place, which is what a game sprite wants; a hand-held camera is fine.
  `--align fixed` keeps real movement through the shot (jumps); use a still camera.

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
| Cleanup | `--cleanup / --no-cleanup` | Remove stray low-contrast single pixels (keeps highlights like eye glints) |
| Outline | `--outline / --no-outline`, `--outline-style solid\|selout`, `--outline-color '#1a1c2c'` | 1px outline; `selout` uses a darker shade of whatever it touches |
| Tone | `--brightness --contrast --saturation` | Pre-adjust the photo |
| Seamless | `--tileable` | Make backgrounds/textures wrap |
| Export | `--scale 8`, `--aseprite`, `--godot` | Upscaled PNG preview, editable Aseprite file, Godot resource |

## Outputs

- **PNG**: 1x (true pixels), plus an optional nearest-neighbour upscale.
- **.aseprite**: written directly from the file spec, so Aseprite isn't needed to
  make it. Sprites that fit in ≤255 colours are saved **Indexed**, with index 0
  transparent, so palette swaps work in Aseprite. Animations get frame timings and a
  tag.
- **Sprite sheet + .json**: Aseprite/TexturePacker "json-array" format, which
  Godot, Unity, GameMaker and Phaser importers read.
- **Godot 4 (`--godot`)**: `SpriteFrames` `.tres` for AnimatedSprite2D, and `TileSet`
  `.tres` for TileMapLayer. Texture paths are relative, so keep the PNG next to the
  `.tres`. For crisp pixels set Project Settings → Rendering → Textures → Default
  Texture Filter = **Nearest**.
- **Tileset + .tmj**: unique tiles plus a Tiled map that rebuilds the image.

## Claude (MCP server)

Claude Code / Claude Desktop can run the whole pipeline and see the results.

```powershell
claude mcp add pixel-pipeline --scope user -- C:\path\to\pixel-pipeline\.venv\Scripts\pixel-pipeline-mcp.exe
```

Claude Desktop (`%APPDATA%\Claude\claude_desktop_config.json`), then restart it:

```json
{ "mcpServers": { "pixel-pipeline": {
    "command": "C:\\path\\to\\pixel-pipeline\\.venv\\Scripts\\pixel-pipeline-mcp.exe" } } }
```

Tools: `convert_photos`, `video_to_animation`, `idle_animation`, `make_sprite_sheet`,
`make_tileset`, `preview_image`, `list_palettes`. Every result includes a preview
image so Claude can check its work and adjust. Use absolute paths. Example:

> Turn the photos in C:\Users\spark\Pictures\rex into 32x32 animal sprites with the
> PICO-8 palette and sel-out outlines, show me, then make a breathing idle with Godot
> output.

The first background removal downloads a ~170 MB model. If that first call times
out, retry once it's cached.

## Python API

```python
from pixel_pipeline import convert, convert_batch, settings_for
from pixel_pipeline.aseprite import write_aseprite
from pixel_pipeline.video import video_to_sprites

s = settings_for("animal", width=48, height=32, palette="endesga-32", outline_style="selout")
sprite = convert("rex.jpg", s)
sprite.save("rex.png", scale=8)
write_aseprite("rex.aseprite", [sprite.rgba], sprite.palette)
frames, ms = video_to_sprites("rex_walk.mp4", s, count=8, start=1.2, end=2.0)
```

## Tests

```powershell
pip install -e ".[dev,ui,cutout,mcp]"
pytest                        # set GODOT_BIN=path\to\godot.exe to also load the Godot files in Godot
```

## Roadmap

1. **Done:** converter, CLI, desktop app, video → animation, idle loops, sel-out
   outlines, sheets, tilesets, `.aseprite`, Godot resources, MCP server, Windows exe,
   phone web app, Android app.
2. AI style pass (pluggable, optional): PixelLab (animation, 4/8-direction
   rotations), Retro Diffusion API, local models via ComfyUI on the GPU. Whatever
   the AI returns goes back through the converter, so it always lands on your exact
   grid and palette.
3. Palette ramps and shading, batch/sheet tools in the app.

## Credits

The edge-preserving pixelization is adapted from
[PixelOE](https://github.com/KohakuBlueleaf/PixelOE) by Shih-Ying Yeh (Apache-2.0).
See `THIRD_PARTY_NOTICES.md`.
