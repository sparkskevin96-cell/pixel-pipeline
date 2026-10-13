# Train style (project style LoRA)

Teach SDXL one project's look from its finished assets (20+ characters' front sprites, item
icons, props, pixel portraits). New assets generated with that LoRA plus the project's style
word should then match. Code: `styletrain.py`, dialog `ui/style_train.py`, tests
`tests/test_styletrain.py` (dry run, no GPU).

## Trainer: ComfyUI's built-in training nodes

ComfyUI 0.39.0 (the portable install at `C:\AI\ComfyUI_windows_portable`, torch 2.14.0+cu130)
ships `comfy_extras/nodes_train.py` and `nodes_dataset.py`. Nothing to download or install.

| Option | Install | SDXL on 8 GB | Notes |
|---|---|---|---|
| **ComfyUI TrainLoraNode (chosen)** | none (already installed) | bf16 + gradient checkpointing, cached latents; optional offloading | AdamW/Adam/SGD/RMSprop only (no 8-bit AdamW or Adafactor); one prompt runs every step, so pacing is done by chunking |
| kohya-ss sd-scripts `sdxl_train_network.py` | git clone plus its own venv (torch, xformers, bitsandbytes...), about 3-5 GB | documented: 8 GB works, 10 GB recommended | the most proven option; AdamW8bit, Adafactor, `--cache_text_encoder_outputs`, real resume (`--save_state`) |
| OneTrainer | its own venv, about 5 GB | yes | GUI-first; harder to drive from our code |

Why ComfyUI: no install, the same models and folders generation already uses, driven through
the same HTTP API as `generate.py`. kohya is the fallback if ComfyUI's node turns out too slow
or runs out of memory on the 4070 (see "If it doesn't fit").

### Workflows (exact node inputs from the 0.39 source)

1. Encode once (cached latents): `CheckpointLoaderSimple` -> `LoraLoader` (Pixel Art XL) ->
   `LoadImageTextDataSetFromFolder(folder="pp_style/<project>")` (ComfyUI/input) ->
   `MakeTrainingDataset(images, texts, vae, clip)` -> `SaveTrainingDataset(folder_name=
   "pp_style/<project>")` (ComfyUI/datasets). The VAE is `sdxl_vae.safetensors` when installed.
2. Each chunk: `CheckpointLoaderSimple` -> `LoraLoader` (Pixel Art XL) ->
   `LoadTrainingDataset` -> `TrainLoraNode(steps=chunk, rank 8, lr 5e-4, AdamW, MSE, bf16/bf16,
   algorithm LoRA, gradient_checkpointing, checkpoint_depth 1, existing_lora=<last checkpoint>)`
   -> `SaveLoRA(prefix="loras/pp_style/<project>_c<N>", steps=<TrainLoraNode steps output>)`.

The style LoRA is trained **on top of Pixel Art XL**, because every generation already uses
Pixel Art XL. It learns only the step from "Pixel Art XL look" to "this project's look", and at
generation time it is stacked after Pixel Art XL. `base_lora="none"` trains on bare SDXL.

## Dataset

- Assets tagged `approved` (all assets of those kinds if none carry the tag), kinds character,
  item, prop and portrait. The dialog lets you untick any.
- Source = the asset's own pixels: character sheet frame `idle_S 0` (then SE/SW), item
  `icon.png`, portrait `sheet_pixel.png`, prop sheet front frame, else `thumb.png`.
- Cropped, enlarged by a **whole number with nearest-neighbour** to about 75% of 1024, centred
  on plain white, binary alpha. Pixels stay square and sharp.
- Caption: `<word> style, pixel art sprite of <prompt or name>, <kind words>, plain white
  background, flat colors, clean outline`. Close to generate.py's own prompt wording.
- Style word per project = consonants of its name + `px` ("Default" -> `dfltpx`, "Moon Keep"
  -> `mnkppx`). It can be edited in the dialog.
- Steps: about 40 per picture, 600 to 1600, in whole chunks. 30 pictures = 1200 steps
  (the same as kohya's 10 repeats x 4 epochs). TrainLoraNode samples pictures at random, so
  no repeat folders are needed.

## Pacing (Kevin's laptop crashed once after hours at full load)

- Training runs as chunks of 150 steps. Each chunk is one ComfyUI prompt that ends in a
  checkpoint (`ComfyUI/models/loras/pp_style/<project>_ckpt_<N>_steps_00001_.safetensors`;
  only the newest is kept).
- Between chunks the GPU rests 45 s. If `nvidia-smi` reads 80 °C or more, the next chunk waits
  until 70 °C (gives up after 15 min, and Start resumes).
- At 87 °C during a chunk, it calls `/interrupt`, loses that chunk only, cools, and redoes it.
  After 3 hot chunks in a row it stops.
- The power limit is NOT changed: `nvidia-smi -pl` needs admin, and this GPU reports the
  power limit as N/A.
- Time budget: 90 min per session. The real seconds per step are measured from ComfyUI's
  progress line and used for planning. If the next chunk would pass the budget, it finishes
  with what it has, and Start again ("train more") continues from there.
- Stop (or closing the dialog) interrupts the current chunk. Start resumes from the last
  checkpoint when the pictures and settings are unchanged (`style_train/state.json`). Changed
  pictures or settings start a fresh run.

Two compromises, because the node has no resume of its own. Continuing with `existing_lora`
reloads the LoRA up/down weights, but **AdamW's state restarts** each chunk. The node's
**bias/norm "diff" weights restart from zero** each chunk (it only reloads 2-D LoRA weights),
so the final file carries the last chunk's bias diffs. LoRA quality should be close to an
unchunked run, but this is not verified.

## Result and hooks

- `ComfyUI/models/loras/<project-slug>_style.safetensors` (keys `diffusion_model.*.lora_up/
  lora_down/alpha/diff/diff_b`, which plain `LoraLoader` loads).
- `<project>/style_lora.json` = `{"lora", "trigger", "strength": 0.8, "trained", "steps",
  "base_lora"}`. When `assets.Style` has the fields `style_lora`, `style_trigger` and
  `style_strength`, they are set as well. `styletrain.project_style_lora(project)` reads them
  back: Style fields first, an empty `style_lora` means off, then the json.
- Generation hook: `styletrain.apply_style_lora(wf, lora, strength)` chains a `LoraLoader`
  (node "15") after Pixel Art XL in a `generate.build_workflow` workflow (under the LCM
  sampling patch when fast) and rewires KSampler and both prompts. Add the style word with
  `styled_prompt(prompt, trigger)`.
- Before/after: `styletrain.before_after(project, prompt, seed=...)` runs the same prompt
  and seed without and with the LoRA. It is the dialog's "Before / after" button.

## Time for 30 pictures on the RTX 4070 Laptop (8 GB): estimate, not measured

1200 steps at an assumed 3.5 s/step (1024 px, bf16, checkpointing) = 70 min, plus about
4 min of chunk loading, 6 min of rests and 1 min of encoding = **about 80 min**. At 768 px
(assumed 2 s/step) it is about 50 min. If the first chunk shows more than about 6 s/step, the
GPU is spilling into shared memory: use 768.

## If it doesn't fit

- 16 GB of system RAM is tight. TrainLoraNode deep-copies the SDXL UNet (about 5 GB) next to
  the cached checkpoint. Close other big apps while training.
- Out of memory: resolution 768, then `offloading=True` (slow).
- Still bad: kohya sd-scripts with AdamW8bit, `--cache_text_encoder_outputs`, dim 8. This
  needs Kevin's approval for the download (see the plan in the report).
