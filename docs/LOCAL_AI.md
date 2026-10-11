# Local AI: type a prompt, get a sprite (your GPU, free)

```
"a knight with a red cape" ──► ComfyUI on your RTX 4070 ──► 1024 px picture ──► converter ──► 24x32 SNES sprite
                               (SDXL + Pixel Art XL LoRA)                     (cut-out, grid, palette, outline)
```

The AI picture always goes back through the normal converter, so the result is on
your exact sprite size, palette and outline, like a converted photo. Nothing leaves your
PC, and there's no per-image cost. ComfyUI does the AI part. pixel-pipeline talks to it
over its local web API at `http://127.0.0.1:8188`.

## 1. Install ComfyUI (once)

1. Download **`ComfyUI_windows_portable_nvidia.7z`** (about 1.9 GB) from
   <https://github.com/comfyanonymous/ComfyUI/releases/latest>. Direct link:
   <https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z>.
   This is the build for RTX 20-series and newer. Ignore the `_cu126`, AMD and Intel files.
2. Extract it with [7-Zip](https://www.7-zip.org/) (or the Windows 11 extractor) to a
   short path such as `C:\AI\`, which gives you `C:\AI\ComfyUI_windows_portable\`. If extraction
   fails, right-click the .7z → Properties → **Unblock**, then try again.
3. Update your NVIDIA driver (GeForce Experience / NVIDIA App) if it's older than a few months.

## 2. Download the models (once, about 7 GB)

| File | Put it in `ComfyUI_windows_portable\ComfyUI\…` | Size | Download |
|---|---|---|---|
| `sd_xl_base_1.0.safetensors` (**required**) | `models\checkpoints\` | 6.9 GB | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0.safetensors) |
| `pixel-art-xl.safetensors` (**required**) | `models\loras\` | 171 MB | [nerijs/pixel-art-xl](https://huggingface.co/nerijs/pixel-art-xl/blob/main/pixel-art-xl.safetensors) |
| `pytorch_lora_weights.safetensors` → rename to **`lcm-lora-sdxl.safetensors`** (optional, fast mode) | `models\loras\` | 394 MB | [latent-consistency/lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl/blob/main/pytorch_lora_weights.safetensors) |
| `sdxl_vae.safetensors` (optional, fixes colour speckles) | `models\vae\` | 335 MB | [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/sdxl_vae.safetensors) |

On each page, click the **download** button (↓) next to the file name. Similar file names
also work: `sd_xl_base_1.0_0.9vae.safetensors`, or Civitai's `Pixel_Art_XL_v1.1.safetensors`
(subfolders are fine too). pixel-pipeline finds them by name. Any other SDXL checkpoint
works if you name it, e.g. `--checkpoint juggernautXL.safetensors`.

## 3. Start ComfyUI

Double-click **`run_nvidia_gpu.bat`** in `ComfyUI_windows_portable`. Wait until the console
says `To see the GUI go to: http://127.0.0.1:8188`, and **leave that window open** while
you generate. If Windows Firewall asks, "Private networks" is enough (it's local only).

Optional sanity check: open that address in a browser, pick `sd_xl_base_1.0` in the
"Load Checkpoint" box and press **Run**. You should get a picture after a few seconds.

## 4. Make sprites

```powershell
pixel-pipeline generate --check                       # is ComfyUI up, are the models found?
pixel-pipeline generate "a knight with a red cape" -s person --system snes -o knight.png
pixel-pipeline generate "a red fox" -s animal --palette pico-8 --count 4 -o fox.png   # 4 variations
pixel-pipeline generate "a red fox" -s animal --seed 1234 -o fox.png                 # re-make one you liked
pixel-pipeline generate "a misty pine forest at dusk" -s background -W 320 -H 180 -o forest.png
pixel-pipeline generate "a treasure chest" -s object --fast -o chest.png              # LCM: ~3x faster
```

All the usual `convert` options (`-W/-H`, `--palette`, `--colors`, `--outline-style`,
`--scale`, `--aseprite`, `--godot` …) apply to the result. Each output also keeps the raw
AI picture (`*.raw.png`) and prints its **seed**: same prompt + same seed = same picture.

- `-s` / subject shapes the prompt. person/animal/object ask for "full body, side view,
  centered, plain white background, game asset", which cuts out cleanly. background asks for a
  wide side-scroller scene. Just describe the thing, e.g. "a knight with a red cape".
- `--system snes` adds era words ("16-bit SNES game style") and applies that console's
  sprite size, colours and outline.
- `--raw-prompt` sends your text unchanged. `--negative "..."` replaces the built-in negative prompt.

**Claude (MCP):** the `generate_sprite` tool does the same. Example:

> Generate a 32x32 pixel-art fox sprite with the PICO-8 palette, 4 variations, and show me.
> Then make a breathing idle with Godot output from the best one.

**Python:**

```python
from pixel_pipeline import settings_for
from pixel_pipeline.generate import generate_sprite

for sprite, raw, seed in generate_sprite("a red fox", settings_for("animal"), subject="animal", count=4):
    sprite.save(f"fox_{seed}.png", scale=8)
```

## 5. Sketch → GPU → sprite (img2img)

Give it a rough sprite (yours, or one Claude drew) and the GPU re-renders it in pixel-art
style, keeping the layout:

```powershell
pixel-pipeline generate "a blue wizard with a staff" -s person --init wizard_sketch.png --denoise 0.6 -o wizard.png
```

The sketch is enlarged with nearest-neighbour to ~1024 px (transparent areas become white),
rendered, then snapped back to your grid. `--denoise`: 0.3 = stay close to the sketch,
0.6 = default, 0.8 = only the rough layout survives.

## Settings that matter

| Option | Default | Notes |
|---|---|---|
| steps | 25 (fast: 8) | 20–30 is plenty for SDXL |
| cfg | 6.0 (fast: 1.5) | higher = follows the prompt harder, harsher colours |
| sampler / scheduler | euler_ancestral / normal (fast: lcm / sgm_uniform) | |
| LoRA strength | 1.0 (fast: 1.2, the LoRA author's LCM setting) | 0.8–1.2. Lower = less "pixel" |
| generation size | SDXL size nearest the sprite's shape: 1024x1024, 896x1152, 832x1216, 1344x768 … | stay near 1 megapixel |
| `--fast` | off | needs `lcm-lora-sdxl.safetensors`. A bit rougher, ~3x faster |
| `--vae sdxl_vae.safetensors` | the checkpoint's VAE | use it if you see speckles or washed-out colours |

Pixel Art XL's own tips: no trigger word or style prompt is needed, don't use the SDXL
refiner, and it works for isometric and side-on art. The LoRA draws "pixels" about 8
screen-pixels wide (128x128 logical at 1024²). The converter then picks your real size and
palette.

## Speed and VRAM (RTX 4070 Laptop, 8 GB)

- SDXL in fp16 uses about 7 GB of VRAM plus 1–2 GB working memory at 1024x1024, which
  is tight in 8 GB: ComfyUI offloads part of the model to RAM by itself (slower, still works). Use the fp16-fix VAE.
- The first image after starting ComfyUI takes about 20–60 s (loading 7 GB from disk). After
  that, expect roughly **6–12 s per image** at 25 steps and **2–4 s with `--fast`**. These are
  estimates; the first runs will tell. Plug the laptop in: on battery the GPU is throttled hard.
- Close GPU-heavy apps (games, a running Godot project, video editors) while generating.

## Troubleshooting

| Message / symptom | Fix |
|---|---|
| `ComfyUI isn't running at http://127.0.0.1:8188` | Start `run_nvidia_gpu.bat` and wait for the "To see the GUI" line. Using **ComfyUI Desktop** instead? It uses port 8000: `set COMFYUI_URL=http://127.0.0.1:8000` (PowerShell: `$env:COMFYUI_URL="http://127.0.0.1:8000"`). |
| `'pixel-art-xl.safetensors' isn't installed … Installed loras: …` | File is in the wrong folder or has a different name. The message lists what ComfyUI sees. Move/rename it, then restart ComfyUI or press **R** in its browser tab. |
| `torch.OutOfMemoryError: CUDA out of memory` | Close other GPU apps, keep sizes near 1024x1024, or edit `run_nvidia_gpu.bat` and add `--lowvram` to the end of the `python … main.py` line. |
| `didn't finish within 900 s` | ComfyUI is stuck or very slow: look at its console window. Restart it. |
| Smooth/painterly, not pixel art | The LoRA isn't loading (check `--check`), or raise LoRA strength to 1.2. |
| Speckles, odd colours | Add the fp16-fix VAE (table above) and pass `--vae sdxl_vae.safetensors`. |
| Several characters / a sprite sheet instead of one | Re-roll (new seed), simplify the prompt, or use `--init` with a sketch. |
| Background not cut out cleanly | Try `--model birefnet-general`, or a plainer prompt ("on a plain white background"). |

## Prompt to 3D model (the Prompt to Model app)

Type what you want; your GPU makes a front picture, turns it into a 3D shape with
Hunyuan3D 2.1 (built into ComfyUI), the tool paints the shape from the picture, then shows it
spinning next to its 4/8/16-direction sprites. **Save all** writes the `.glb`, the picture, the
sprite sheet + `.json`, `.aseprite`, Godot SpriteFrames + a facing script, and a spinning GIF.

Setup (once, on top of steps 1–3 above):
1. **Update ComfyUI** (portable: run `update\update_comfyui.bat`). The Hunyuan3D nodes and
   "Voxel to Mesh" need a 2025 or newer ComfyUI.
2. **Download** [hunyuan_3d_v2.1.safetensors](https://huggingface.co/Comfy-Org/hunyuan3D_2.1_repackaged/resolve/main/hunyuan_3d_v2.1.safetensors)
   (several GB) into `ComfyUI\models\checkpoints\`. Restart ComfyUI. Older Hunyuan3D 2.0
   checkpoints (`hunyuan3d-dit-v2-0`, `-mini`, `-mv`) work too; the first one installed is used.
3. Open the app: **`pixel-pipeline-model`**, or **PixelPipeline.exe → File → Prompt to 3D model**
   (Ctrl+M), or `PixelPipeline.exe --model`. Command line:

   ```
   pixel-pipeline model "a knight with a red cape" -o knight.glb --directions 16 --godot --aseprite
   pixel-pipeline model --image my_drawing.png -o hero.glb          # your own front picture
   pixel-pipeline model --image front.png --back back.png -o hero.glb  # better colours behind
   ```

How it works and what to expect (not yet timed on your laptop):
- **Picture** (~10 s): SDXL **without** the pixel-art LoRA, prompted for one whole subject
  facing the viewer on plain white with even light. Clean pictures rebuild best.
- **Shape** (likely 1–3 min): Hunyuan3D 2.1, wired exactly like ComfyUI's own "Image to Model
  (Hunyuan3d 2.1)" template (30 steps, cfg 5, octree 256). About 10 GB of VRAM, so close games
  and other GPU apps first. `--octree 192` is lighter, `384` more detailed.
- **Colour** (CPU, ~10–40 s): Hunyuan's own texture step needs ~21 GB, so the tool paints the
  shape itself: the picture goes onto everything the front view shows (it finds the model's
  orientation by matching outlines), and the back gets a blurred version of the front
  straight through, so a red cape stays red behind but the back of a head may come out skin
  coloured. Give a back-view picture (`--back`, or draw one) to fix that.
- **Sprites**: rendered with one camera tilt (`--elevation 30` top-down/isometric, `0` side
  view), one scale and one palette. Pick the size with the [sprite scale table](SPRITE_SCALE.md).

Swap in a different 3D model maker: export any ComfyUI workflow in API format with
`{{image}}` in its LoadImage and a "Save 3D Model" (SaveGLB) node at the end, and pass it with
`--template`. A model that comes back with its own colours (e.g. ComfyUI's TRELLIS.2 nodes on a
16 GB+ GPU) is kept as it is. Or skip the GPU: **Open a 3D model…** in the app takes any
`.glb` (Tripo, Meshy, Blender, ...).

| Problem | Fix |
|---|---|
| `No Hunyuan3D checkpoint in ComfyUI/models/checkpoints` | Download it (step 2) into that folder and restart ComfyUI. |
| `Node 'VoxelToMesh' has unknown class_type` (or `EmptyLatentHunyuan3Dv2`) | Update ComfyUI (step 1). |
| Out of memory during the shape | Close other GPU apps, use `--octree 192`, or add `--lowvram` to the ComfyUI start line. |
| Model's colours look shifted or "the outline only matches the picture N%" | The shape doesn't match the picture well (often extra props or a cropped picture). Re-roll with another seed, or simplify the prompt. |
| Wrong way round in the sprites | It found the back instead of the front: turn it with `turnaround model.glb --yaw 180`. |

## Image to 3D (for 8- and 16-direction sprites)

`pixel-pipeline turnaround model.glb` turns any 3D model into 4, 8 or 16 facing sprites. To
get a model from a picture (a `generate` result, a drawing, a photo), an image-to-3D AI makes
up the back and sides. None of these have been tested on your laptop yet; numbers are from
their own docs and community reports.

| Option | Runs on | VRAM | Colours | Notes |
|---|---|---|---|---|
| **Stable Fast 3D** (Stability AI) | your 4070 | ~6 GB | textured | Under a second per model. Free licence under $1M yearly revenue. Gated download on Hugging Face (accept the terms first). |
| **TripoSR** | your 4070 | ~6 GB | vertex colours | MIT licence. Lowest detail, fine at sprite sizes. |
| **Hunyuan3D 2.1** | needs ~10 GB: over the 4070's 8 GB | ~10 GB shape, ~21 GB texture | shape: none | Best shapes. Native ComfyUI nodes do the shape; texturing needs a bigger GPU or a low-VRAM fork. An untextured model still renders (grey, shaded), useful for a silhouette you recolour. |
| **TRELLIS.2** (Microsoft) | 16–24 GB GPU or cloud | 16 GB+ | PBR | Top quality, too big for 8 GB without heavy offloading. |
| **Tripo, Meshy, Hunyuan3D web demo** | online | none | textured | Free tiers; upload the picture, download the `.glb`. Quickest way to try the pipeline today. |

Tips:
- Give the AI a clean picture: the whole subject, side or three-quarter view, plain
  background. `generate` with "full body, plain white background" works well.
- Sprite sizes hide a lot: a rough model is fine at 48–80 px. Pick the size with the
  [sprite scale table](SPRITE_SCALE.md).
- The tool lights the model from the upper left (`--light 0.35`). Models whose texture
  already has shading baked in look better with `--light 0.15`.

## Your own ComfyUI workflows (Flux, Qwen-Image, …)

pixel-pipeline can run any ComfyUI workflow you build, so you can swap models without code
changes:

1. Build and test it in the ComfyUI browser UI.
2. **Workflow → Export (API)**. Older versions: Settings → enable Dev mode → **Save (API Format)**.
   The normal "Save" format won't work.
3. In the exported JSON, replace the values with placeholders: `"text": "{{prompt}}"`,
   `"seed": "{{seed}}"`, and optionally `{{negative}}`, `{{width}}`, `{{height}}`, `{{steps}}`, `{{cfg}}`,
   `{{sampler}}`, `{{scheduler}}`, `{{checkpoint}}`, `{{lora}}`, `{{lora_strength}}`, `{{prefix}}`. For
   img2img, use `{{image}}` (LoadImage) and `{{denoise}}`. A quoted placeholder that is the whole value
   becomes a number where needed.
4. `pixel-pipeline generate "a fox" -s animal --template my_flux_api.json -o fox.png`

Starting points: `tools/comfyui/sdxl_pixel_art_txt2img.json` and `sdxl_pixel_art_img2img.json`
(the built-in workflow as templates). The workflow must end in a **Save Image** (or Preview Image)
node. Flux / Qwen-Image need their own model files (in `models\diffusion_models` etc.). In 8 GB,
use the fp8 or GGUF versions.

## Licences (short version, not legal advice)

- **SDXL 1.0 base**: CreativeML Open RAIL++-M. Free, including commercial use of what you
  make. You may not use it for the harmful uses the licence lists (illegal content, harassment,
  etc.). You're responsible for your outputs.
- **Pixel Art XL** (nerijs): CreativeML OpenRAIL-M. Same idea.
- **LCM-LoRA SDXL**: OpenRAIL++. **sdxl-vae-fp16-fix**: MIT.
- **ComfyUI**: GPL-3.0. pixel-pipeline only talks to it over HTTP and doesn't bundle it, so this
  doesn't affect pixel-pipeline or your games.
- Selling a game with these sprites is fine. Two caveats: in the US, purely AI-generated images
  generally can't be copyrighted (your edits in Aseprite can be), and **Steam asks you to disclose
  AI-generated content** when you submit a game.
