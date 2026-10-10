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

**Easiest:** download the ready-made app: repo → **Releases** → **windows-latest** →
`PixelPipeline-windows.zip` (every push to `main` rebuilds it). Extract it and run
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

- **Draw from a prompt** (web version): Claude draws the sprite pixel by pixel at your
  size, colours and retro system, using your Claude account.
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

One window, a tab per tool:

| Tab | What it does |
|---|---|
| **Sprite** | Photo or picture → sprite, live. Video → walk cycle. Idle loops. |
| **Prompt** | Words → sprite from PixelLab, Retro Diffusion and your GPU, side by side. |
| **Directions** | 3D model, photos or a video → 4/8/16 facing directions. A rigged `.glb` plays its animations from every side. Save writes sheet, `.json`, `.aseprite`, Godot SpriteFrames + facing script, GIF. |
| **Teams** | Sprite or sheet → player / enemy / ally / other / spent colours + the Godot palette shader. |
| **Portrait** | Photo, picture or prompt (GPU) → head-and-shoulders dialogue portrait. |
| **Map tiles** | Built-in, photo or prompt terrains → tiles with shores and edges + Godot TileSet. |
| **Godot demo** | A ready-to-open Godot project: map + units that walk where you click. |
| **Prompt to Model** | Prompt → 3D model on your GPU → direction sprites. |

Sprite tab: drop a photo on the window. Pick a subject type (it loads that preset), then
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

# Retro console look: original palette / colour depth, colours per sprite, sizes
pixel-pipeline convert rex.jpg -s animal --system nes -o rex_nes.png      # also snes, gameboy, gbc, sms, genesis, gba, pico8
pixel-pipeline convert rex.jpg -s animal --system nes --look authentic -o rex_tiny.png   # one 16x16, 3-colour sprite

# Text prompt -> sprite on your own GPU (ComfyUI; setup: docs/LOCAL_AI.md)
pixel-pipeline generate --check
pixel-pipeline generate "a knight with a red cape" -s person --system snes --count 4 -o knight.png
pixel-pipeline generate "a blue wizard" -s person --init sketch.png --denoise 0.6 -o wizard.png

# Scenery photo -> far/mid/near parallax layers + Godot 4 Parallax2D scene
pixel-pipeline parallax lake.jpg -o bg/ -W 320 -H 180 --scale 3

# Map tiles: grass base + water/sand/road with corner transitions + Godot TileSet (paint with Terrains)
pixel-pipeline maptiles -o maps/terrain.png -t grass -t water -t sand -t road
pixel-pipeline maptiles -o maps/cave.png -t stone -t water=murky_water.jpg -t "lava=prompt:glowing lava cracks"

# Team colours (Fire Emblem style): blue player, red enemy, green ally, purple other, grey spent
pixel-pipeline teams knight_dirs.png --godot          # + palette strip, shader, a material per team
# Dialogue portrait (head and shoulders) from a photo, a picture, or a prompt on your GPU
pixel-pipeline portrait me.jpg -o me_portrait.png -W 96 -H 96 --scale 4
pixel-pipeline portrait --prompt "a young knight with red hair" -o knight_face.png

# 16 facing directions from a 3D model (e.g. an image-to-3D AI's .glb), Godot-ready
pixel-pipeline turnaround knight.glb -o knight_dirs.png --directions 16 -W 48 -H 48 --godot --aseprite --preview 6
# ...or from a video walking all the way around a still subject (start in front, walk to your right)
pixel-pipeline turnaround walkaround.mp4 -o rex_dirs.png -s animal --directions 8
# Animated: every animation of a rigged .glb in 8 directions (walk_E, walk_SE ... in Godot)
pixel-pipeline turnaround knight_rigged.glb -o knight_anim.png --animate --frames 8 --directions 8 --godot --preview 3
# ...animations from separate Mixamo downloads, colours from the original Prompt to Model .glb
pixel-pipeline turnaround knight_idle.glb knight_walk.glb --animate --colours-from knight.glb -o knight_anim.png --godot

# A ready-to-open Godot 4 project using it all: map + units that walk where you click
pixel-pipeline demo -o MyDemo --preview                       # built-in knight
pixel-pipeline demo -o MyDemo --unit knight_idle.glb knight_walk.glb --colours-from knight.glb --directions 16

pixel-pipeline palettes                      # list built-ins
pixel-pipeline convert me.jpg -s person --palette lospec:resurrect-64 --outline-style selout
pixel-pipeline <command> --help              # every option
```

### Prompt to Model (app)

`pixel-pipeline-model` (or File → Prompt to 3D model in the desktop app): type what you want,
your GPU makes a picture and a 3D model (Hunyuan3D in ComfyUI), and you get it spinning next
to its 4/8/16-direction sprites, ready for Godot. Setup and details:
[docs/LOCAL_AI.md](docs/LOCAL_AI.md#prompt-to-3d-model-the-prompt-to-model-app).

### Any sprite in 16 directions

A photo or a generated picture only shows one side, so the other directions have to be
made up. The tool does it the way pre-rendered games did (Diablo, StarCraft): through a 3D
model.

1. **Front view**: `generate` on your GPU, Draw mode, or any picture of the thing.
2. **Image → 3D model (.glb)**: an image-to-3D AI invents the back and sides.
   `pixel-pipeline model` / the Prompt to Model app does it on your GPU (Hunyuan3D shape +
   painted from the picture). Online alternatives: Tripo, Meshy or the Hunyuan3D demo. See
   [docs/LOCAL_AI.md](docs/LOCAL_AI.md#image-to-3d-for-8--and-16-direction-sprites).
3. **`turnaround model.glb`**: renders every direction with one camera angle
   (`--elevation 30` top-down/isometric, `0` side view), one scale and one palette.

Directions run clockwise from facing right, the way Godot measures angles: E, ESE, SE,
SSE, S (facing the viewer) ... N (away). With `--godot` you get SpriteFrames with one
animation per direction and `<name>_facing.gd`: attach it to an AnimatedSprite2D and call
`face(velocity)`. If S doesn't face the viewer, turn the model with `--yaw 90` (or 180, -90);
a model lying on its back needs `--up z`.

### Prompt → sprite: PixelLab vs Retro Diffusion vs your GPU

The **Prompt** tab (or `pixel-pipeline ai "a knight with a red cape" -o knight.png`) makes the
same words with each service, side by side: what it drew, and the sprite on your settings.
Only the text is sent; none of your pictures. The services draw real pixel art at your
sprite's size, so their pixels are kept and only snapped to your palette (Retro Diffusion's
fast/plus styles draw at 64 px or more, so small sprites are made at 2x and shrunk back).

- **PixelLab** (pixellab.ai): `pixen` fast, `pixflux` / `bitforge` classic, `pro` best;
  camera (side / low top-down / high top-down) and facing.
- **Retro Diffusion** (retrodiffusion.ai): `rd_fast__default` $0.03, `rd_plus__default`
  $0.06, `rd_pro__default` $0.18 per image, plus retro / isometric / fantasy styles.
- **Your GPU**: ComfyUI + SDXL + Pixel Art XL (free, see docs/LOCAL_AI.md).

Keys: Prompt tab → Keys… (saved in `%USERPROFILE%\.pixel-pipeline\keys.json`, only sent to
their own service), or `pixel-pipeline ai --save-key pixellab=... --save-key
retrodiffusion=...`, or the `PIXELLAB_API_KEY` / `RETRO_DIFFUSION_API_KEY` variables.
`pixel-pipeline ai --balance` shows what's left. "Open in the Sprite tab" sends a result
on, to retro looks, team colours and so on.

### Characters that walk, run and attack (no Mixamo)

Any standing humanoid model gets a skeleton and six animations: **idle, walk, run, attack,
hurt, death**. The joints are found from the model's own shape (its front silhouette), the
mesh is skinned to them, and the animations are made to fit. Arms come down from a T/A
pose and the feet stay on the ground.

- App: **Prompt to Model** tab → make a character → tick **Animate** → **Update sprites**.
  Or **Directions** tab → add a model → tick **Add a skeleton + animations**. "Show the
  skeleton" shows the joints it found.
- Command line: `pixel-pipeline turnaround knight.glb --auto-rig --directions 8 --godot`, or
  `pixel-pipeline rig knight.glb --preview rig.png` for just the rigged `.glb` (Godot and
  Blender open it: 17 bones, 6 animations).

Works best on a character standing in an A- or T-pose facing the front, arms clear of the
body (Prompt to Model now asks for that pose). Long robes get standard leg proportions; the
attack swings the right hand (`--weapon-hand left`).

### Animated directions (walk, attack ... in 16 directions)

`turnaround --animate` plays a rigged, animated model (.glb/.gltf with a skeleton) and
renders every frame from every direction, like Diablo and StarCraft did: one scale, one
ground line and one palette across all of them. The sheet has a row per animation and
direction and a column per frame. With `--godot` the SpriteFrames hold `walk_E`,
`walk_ESE`, ... and `<name>_facing.gd` has `face(velocity)` plus `act(&"walk")`; turning
mid-walk keeps the step. Walks stay on the spot (root motion is removed; `--no-in-place`
keeps it). Loops are detected (a walk's last frame isn't a repeat of its first); one-shot
animations end on their last pose and don't loop in Godot.

Getting a rigged model, free:

1. Make the model: Prompt to Model or `pixel-pipeline model ... --obj` (Save all in the app
   writes the `.obj` too). Any humanoid in a T- or A-pose works.
2. [Mixamo](https://www.mixamo.com) (free Adobe account): Upload Character → the `.obj` →
   place the markers → it adds a skeleton. Pick animations; tick **In Place** for walks and
   runs. Download each as FBX (first one *With Skin*, the rest can be *Without Skin*).
3. FBX → glTF: in Blender, File → Import → FBX, then File → Export → glTF 2.0 (.glb). Or
   in Godot 4 drop the FBX in a project, open it, and Scene → Export As → glTF 2.0.
4. `pixel-pipeline turnaround idle.glb walk.glb attack.glb --animate --colours-from knight.glb ...`
   The first file gives the model, every file adds its animations (named after the file
   when the name inside is Mixamo's `mixamo.com`). Mixamo drops vertex colours, so
   `--colours-from` copies them back from the original by shape. `--animations idle,walk`
   picks and orders them; `run=walk_fast` renames.

Tripo and Meshy can also auto-rig and animate their models (paid). Rendering is about 0.1 s
per frame, so 3 animations x 16 directions x 8 frames takes under a minute.

### Godot demo project

`pixel-pipeline demo -o MyDemo` writes a Godot 4 project that shows the outputs working
together: 640x360 scaled x3 to 1080p (integer scaling, nearest filtering, pixel snap), a map
from the map tiles, and four units in 8 (or 16) directions with team colours, two blue
players and two red enemies sharing one sprite sheet through the palette shader. Open
`project.godot` and press F5: click your unit, then a tile. It walks there along an A* path
around the water, turning and playing its walk, then greys out (spent); Space starts the
next turn. With no `--unit` it uses a built-in rigged knight (idle + walk), so it works
with no files; `--unit` takes the same sources as `turnaround` (a rigged .glb plays the
animations whose names contain idle and walk). The scripts (`main.gd`, `unit.gd`) are short
and meant to be copied into your game.

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
| Retro system | `--system nes\|snes\|gameboy\|gbc\|sms\|genesis\|gba\|pico8` | Original console colours (hardware palette or colour depth) and that era's outline style. Game Boy maps brightness to its 4 shades |
| Look | `--look readable\|authentic` | With a system. `readable` (default): recognizable size (people 48×64, animals 48×48, objects 32×32) with a multi-palette character's colours (NES 12, Game Boy 4, SNES/Genesis/GBA 32). `authentic`: one sprite of that era (NES 16×24 in 3 colours), faithful but photos turn into blobs |
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
- **Map tiles** (`maptiles`): an atlas (row 0 = base variants, then a row per terrain with its
  15 corner tiles, the last one full), `<name>.tres` TileSet with a Match Corners terrain set,
  and an example map. In Godot add a TileMapLayer, set the TileSet, and paint in the
  **Terrains** tab. Terrains blend only into the base, so keep base between two others.
- **Team colours** (`teams`): a PNG per team, and with `--godot` a palette strip
  (`<name>_teams.png`, row 0 = colours as drawn, then player, enemy, ally, other, spent), the
  `team_colours.gdshader` palette-swap shader and a ShaderMaterial per team: put one on the
  AnimatedSprite2D and set `team` / `spent` in code, so one sheet serves every side.
- **Turnaround** (`turnaround`): one sheet with a frame per direction (8 per row), JSON with
  a tag per direction, `.aseprite` with the same tags, Godot SpriteFrames with an animation
  per direction plus a `_facing.gd` script, and `--preview` a spinning GIF. With
  `--animate`: a row per animation and direction, tags/animations `walk_SE` etc.,
  `_facing.gd` with `face()` and `act()`, and `--preview` a GIF playing every direction.
- **Parallax** (`parallax`): one PNG per depth layer (far one opaque, nearer ones
  transparent, with hidden areas filled so scrolling never shows holes), sharing one
  palette, and `<name>_parallax.tscn` with Parallax2D nodes (scroll 0.2 / 0.5 / 0.85,
  horizontal repeat). Drop the folder into your Godot project and instance the scene. The
  depth model (Depth Anything V2 Small, ~95 MB) downloads on first use. It works best on
  scenery. `--depth my_depth.png` uses a hand-painted depth map instead (white = near).

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

Tools: `convert_photos`, `generate_sprite` (local GPU), `draw_sprite` (Claude draws it
pixel by pixel), `video_to_animation`, `idle_animation`, `parallax_layers`,
`make_sprite_sheet`, `make_tileset`, `turnaround` (also animated), `prompt_to_model` (local
GPU), `online_sprites` (PixelLab / Retro Diffusion / GPU side by side), `team_colours`,
`portrait`, `map_tiles`, `godot_demo`, `preview_image`, `list_palettes`.

**Claude + your PC:** Claude can sketch a sprite with `draw_sprite`, pass it to
`generate_sprite` as `init_image` so your GPU renders the detail, and then check the
preview and adjust. Or Claude can write a detailed prompt and try a few seeds. Every result includes a preview
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

1. **Done:** converter, CLI, desktop app, video → animation, idle loops (breathe, bob,
   sway, engine rumble), sel-out outlines, retro systems, sheets, tilesets, parallax,
   `.aseprite`, Godot resources, MCP server, Windows exe, phone web app, Android app,
   Claude-drawn sprites, local GPU generation (ComfyUI).
2. Hosted AI (optional): PixelLab (animation, 4/8-direction rotations) or Retro
   Diffusion. Whatever the AI returns goes back through the converter, so it always
   lands on your exact grid and palette.
3. Palette ramps and shading, batch/sheet tools in the app.

## Credits

The edge-preserving pixelization is adapted from
[PixelOE](https://github.com/KohakuBlueleaf/PixelOE) by Shih-Ying Yeh (Apache-2.0).
See `THIRD_PARTY_NOTICES.md`.
