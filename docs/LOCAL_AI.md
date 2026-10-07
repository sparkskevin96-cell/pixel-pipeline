# Local AI: type a prompt, get a sprite (your GPU, free)

```
"a knight with a red cape" ──► ComfyUI on your RTX 4080 ──► 1024 px picture ──► converter ──► 24x32 SNES sprite
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

## Speed and VRAM (RTX 4080 Laptop, 12 GB)

- SDXL in fp16 uses about 7 GB of VRAM plus 1–2 GB working memory at 1024x1024, which
  fits in 12 GB. ComfyUI loads and unloads models by itself.
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
node. Flux / Qwen-Image need their own model files (in `models\diffusion_models` etc.). In 12 GB,
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
