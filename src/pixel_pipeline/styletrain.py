"""Train style: teach SDXL a project's look from its approved assets (a style LoRA), so new
assets match.

    approved assets ──► dataset (sprites enlarged crisp on white, 1024 px, captions with the
    project's trigger word) ──► ComfyUI's built-in training nodes (TrainLoraNode, no install)
    in short chunks with GPU-temperature pauses ──► ComfyUI/models/loras/<project>_style.safetensors
    ──► recorded for the project (style_lora.json, and Style.style_lora when assets.py has it)

Trainer: ComfyUI 0.39's own comfy_extras/nodes_train.py + nodes_dataset.py (TrainLoraNode,
SaveLoRA, MakeTrainingDataset, SaveTrainingDataset, LoadTrainingDataset). Nothing to download:
it trains SDXL base (+ Pixel Art XL underneath, so the style LoRA only learns the difference),
bf16, gradient checkpointing, cached latents, rank 8, AdamW, batch 1. See docs/STYLE_TRAINING.md.

Pacing (Kevin's laptop crashed once after hours at full load): training runs in chunks of
`chunk_steps`; between chunks it rests, and waits while the GPU is above `pause_above` °C
(nvidia-smi). Above `hard_limit` during a chunk the chunk is interrupted and redone after
cooling. Every chunk ends in a checkpoint, so Stop / a crash loses at most one chunk and
Start again resumes. The whole run keeps to `max_minutes` (steps are trimmed if the GPU is
slower than estimated; "train more" continues later).

Tests use FakeBackend (dry run, no GPU, no ComfyUI).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import struct
import subprocess
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

from . import assets as assets_mod
from .assets import Asset, Project

log = logging.getLogger(__name__)

APPROVED_TAG = "approved"
TRAIN_KINDS = ("character", "item", "prop", "portrait")
SUBFOLDER = "pp_style"                       # under ComfyUI input/, datasets/, models/loras/
MIN_IMAGES = 20                              # fewer still trains, with a warning

# Rough speeds for the RTX 4070 Laptop 8 GB (seconds per step, batch 1, bf16, gradient
# checkpointing). NOT measured yet: the runner measures the real speed on the first chunk and
# re-plans; the UI shows the estimate as "about".
SEC_PER_STEP = {1024: 3.5, 768: 2.0, 512: 1.2}
CHUNK_OVERHEAD = 30.0                        # s per chunk: copy SDXL in RAM + load it to the GPU
ENCODE_SECONDS = 60.0                        # VAE + CLIP encoding the dataset once


class StyleTrainError(RuntimeError):
    """Something the user has to fix (message written for them)."""


class Interrupted(Exception):
    """A chunk was interrupted (Stop pressed, or the GPU got too hot)."""

    def __init__(self, why: str):
        super().__init__(why)
        self.why = why


# --------------------------------------------------------------------------- config

@dataclass
class TrainConfig:
    resolution: int = 1024           # SDXL native; 768 if 1024 runs out of memory / is slow
    rank: int = 8
    learning_rate: float = 5e-4      # TrainLoraNode default (alpha 1, so the scale is 1/rank)
    steps: int = 0                   # 0 = auto from the image count (auto_steps)
    chunk_steps: int = 150           # steps per ComfyUI prompt = per checkpoint
    batch_size: int = 1
    optimizer: str = "AdamW"
    checkpoint: str = "sd_xl_base_1.0.safetensors"
    base_lora: str = "pixel-art-xl.safetensors"   # trained on top of it; "none" = SDXL alone
    base_lora_strength: float = 1.0
    seed: int = 0
    trigger: str = ""                # "" = from the project name (trigger_word)
    background: tuple = (255, 255, 255)
    fill: float = 0.75               # sprite's longer side as a share of the picture
    mirror: bool = False             # also train on mirrored copies
    gradient_checkpointing: bool = True
    offloading: bool = False         # weights on the CPU: less VRAM, much slower
    # pacing
    max_minutes: float = 90.0
    rest_seconds: float = 45.0       # pause between chunks
    pause_above: float = 80.0        # °C: wait before the next chunk until...
    resume_below: float = 70.0       # ...the GPU is this cool
    hard_limit: float = 87.0         # °C during a chunk: interrupt it, cool, redo it
    max_cool_minutes: float = 15.0   # give up (stop, resumable) if it doesn't cool by then
    poll_seconds: float = 3.0
    strength: float = 0.8            # default strength when generating with the result


def auto_steps(images: int, chunk: int = 150) -> int:
    """~40 steps per image, 600..1600, whole chunks (30 images -> 1200)."""
    steps = int(np.clip(round(images * 40), 600, 1600))
    return int(np.ceil(steps / chunk) * chunk)


def estimate_seconds(images: int, cfg: TrainConfig | None = None,
                     sec_per_step: float | None = None, done_steps: int = 0) -> float:
    """Whole run (encode + chunks + rests), capped by nothing: compare with max_minutes."""
    cfg = cfg or TrainConfig()
    total = cfg.steps or auto_steps(images, cfg.chunk_steps)
    left = max(0, total - done_steps)
    sps = sec_per_step or SEC_PER_STEP.get(cfg.resolution, SEC_PER_STEP[1024] * (cfg.resolution / 1024) ** 2)
    if cfg.offloading:
        sps *= 2.5
    chunks = int(np.ceil(left / cfg.chunk_steps))
    return (ENCODE_SECONDS if done_steps == 0 else 0) + left * sps + chunks * CHUNK_OVERHEAD \
        + max(0, chunks - 1) * cfg.rest_seconds


def trigger_word(project_name: str) -> str:
    """A rare made-up word for the project's look: consonants of its name + 'px'
    ("Default" -> "dfltpx"). Rare tokens don't drag in meanings the model already has."""
    s = assets_mod.slug(project_name).replace("-", "")
    cons = re.sub(r"[aeiou0-9]", "", s)[:6]
    if len(cons) < 3:
        cons = (cons + "pps")[:3]
    return cons + "px"


def lora_file_name(project: Project) -> str:
    return f"{assets_mod.slug(project.name)}_style.safetensors"


# --------------------------------------------------------------------------- dataset

@dataclass
class Sample:
    asset: Asset
    rgba: np.ndarray                 # the asset's front sprite at its own pixel size
    caption: str
    source: str                      # file it came from (for the dialog / debugging)

    @property
    def key(self) -> str:
        return f"{self.asset.kind}/{self.asset.folder.name}"


_PREFERRED_FRAMES = ("idle_S 0", "idle_SE 0", "idle_SW 0", "S", "SE", "neutral")


def _frame(sheet: Path) -> np.ndarray | None:
    meta_p = sheet.with_suffix(".json")
    try:
        img = np.asarray(Image.open(sheet).convert("RGBA"))
    except (OSError, ValueError):
        return None
    try:
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        frames = meta["frames"]
        if isinstance(frames, dict):
            frames = [{"filename": k, **v} for k, v in frames.items()]
        by_name = {str(f.get("filename", "")): f["frame"] for f in frames}
    except (OSError, ValueError, KeyError, TypeError):
        return img
    pick = next((by_name[k] for k in _PREFERRED_FRAMES if k in by_name), None)
    if pick is None:   # a frame named like "turn_S 0" / "S 0"
        for name, r in by_name.items():
            if "S" in re.split(r"[^A-Za-z0-9]+", name):
                pick = r
                break
    r = pick or next(iter(by_name.values()))
    return img[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]


def source_sprite(asset: Asset) -> tuple[np.ndarray, str] | None:
    """The asset's front-facing pixel sprite at its own size (not the enlarged thumbnail
    when anything better exists). None = nothing usable."""
    f = asset.folder
    candidates: list[Path] = []
    if asset.kind == "item":
        candidates += [f / "icon.png"]
    if asset.kind == "portrait":
        candidates += [f / "sheet_pixel.png"]
    candidates += [f / "sheet.png", f / "icon.png", f / "thumb.png"]
    for p in candidates:
        if not p.exists():
            continue
        if p.name.startswith("sheet") and p.with_suffix(".json").exists():
            rgba = _frame(p)
        else:
            try:
                rgba = np.asarray(Image.open(p).convert("RGBA"))
            except (OSError, ValueError):
                rgba = None
        if rgba is not None and rgba.size and (rgba[..., 3] > 0).any():
            return np.ascontiguousarray(rgba), str(p)
    return None


_KIND_WORDS = {
    "person": "single character, full body, standing, front view",
    "animal": "single animal, full body",
    "item": "single item, game item icon",
    "prop": "single prop, game asset",
    "portrait": "character portrait, head and shoulders",
}


def caption(asset: Asset, trigger: str) -> str:
    """'<trigger> style, pixel art sprite of <prompt or name>, <kind words>, plain white
    background' — close to generate.py's own prompts, so generation prompts plus the trigger
    word land on what was learned."""
    desc = str(asset.extra.get("prompt") or "").strip() or asset.name.replace("-", " ")
    role = str(asset.extra.get("role") or "").strip()
    if role and role.lower() not in desc.lower():
        desc = f"{desc}, {role}"
    key = asset.kind
    if asset.kind == "character":
        key = "animal" if asset.extra.get("body") == "animal" else "person"
    what = "pixel art portrait of" if asset.kind == "portrait" else "pixel art sprite of"
    return f"{trigger} style, {what} {desc}, {_KIND_WORDS.get(key, 'game asset')}, " \
           "plain white background, flat colors, clean outline"


def collect(project: Project | None = None, *, tag: str | None = APPROVED_TAG,
            kinds=TRAIN_KINDS, trigger: str | None = None,
            only: set[str] | None = None) -> list[Sample]:
    """The project's training pictures. tag "approved" by default; if no asset carries it,
    every asset of the kinds. `only` = Sample.key values to keep (the dialog's ticks)."""
    proj = project or assets_mod.current_project()
    trig = trigger or trigger_word(proj.name)
    pool = [a for k in kinds for a in assets_mod.list_assets(k, project=proj)]
    if tag:
        tagged = [a for a in pool if tag in a.tags]
        pool = tagged or pool
    out = []
    for a in pool:
        if only is not None and f"{a.kind}/{a.folder.name}" not in only:
            continue
        src = source_sprite(a)
        if src is not None:
            out.append(Sample(a, src[0], caption(a, trig), src[1]))
    return out


def prepare_image(rgba: np.ndarray, resolution: int = 1024, background=(255, 255, 255),
                  fill: float = 0.75) -> Image.Image:
    """Sprite -> square RGB training picture: cropped to its pixels, enlarged by a whole
    number with nearest-neighbour (pixel art LoRA datasets keep the pixels square and sharp),
    centred on a plain background. Alpha is binary-cut so no fringe colours appear."""
    a = rgba[..., 3] > 127
    ys, xs = np.nonzero(a)
    if len(xs):
        rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape
    rgb = np.where(a[..., None], rgba[..., :3], np.array(background, np.uint8)).astype(np.uint8)
    img = Image.fromarray(rgb, "RGB")
    room = max(1, int(resolution * fill))
    k = room / max(w, h)
    if k >= 1:
        k = int(k)
        nw, nh = w * k, h * k
    else:
        nw, nh = max(1, round(w * k)), max(1, round(h * k))
    img = img.resize((nw, nh), Image.NEAREST)
    canvas = Image.new("RGB", (resolution, resolution), tuple(int(c) for c in background))
    canvas.paste(img, ((resolution - nw) // 2, (resolution - nh) // 2))
    return canvas


@dataclass
class Dataset:
    folder: Path                     # images + .txt captions
    count: int                       # pictures written (with mirrors)
    originals: int                   # assets used
    trigger: str
    signature: str                   # changes when pictures, captions or training setup change


def write_dataset(samples: list[Sample], folder: Path, cfg: TrainConfig, trigger: str) -> Dataset:
    """Pictures + same-name .txt captions (the layout ComfyUI's LoadImageTextDataSetFromFolder
    and kohya both read). The folder is emptied first."""
    if not samples:
        raise StyleTrainError("No pictures to train on: approve some assets first (tag them "
                              f"'{APPROVED_TAG}' in the Library) or tick some in the list.")
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    h = hashlib.sha1()
    n = 0
    for i, s in enumerate(samples):
        img = prepare_image(s.rgba, cfg.resolution, cfg.background, cfg.fill)
        pics = [img] + ([img.transpose(Image.FLIP_LEFT_RIGHT)] if cfg.mirror else [])
        for j, pic in enumerate(pics):
            stem = f"{i:03d}_{assets_mod.slug(s.asset.name)[:40]}" + ("_m" if j else "")
            pic.save(folder / f"{stem}.png")
            (folder / f"{stem}.txt").write_text(s.caption, encoding="utf-8")
            h.update(np.asarray(pic).tobytes())
            h.update(s.caption.encode())
            n += 1
    for k in ("resolution", "checkpoint", "base_lora", "base_lora_strength", "rank",
              "learning_rate", "batch_size", "optimizer"):
        h.update(f"{k}={getattr(cfg, k)}".encode())
    (folder / "dataset.json").write_text(json.dumps(
        {"trigger": trigger, "images": n, "assets": [s.key for s in samples],
         "captions": [s.caption for s in samples]}, indent=1), encoding="utf-8")
    return Dataset(folder, n, len(samples), trigger, h.hexdigest()[:16])


# --------------------------------------------------------------------------- ComfyUI workflows

def _node(wf: dict, nid: str, cls: str, title: str, **inputs) -> list:
    wf[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}
    return [nid, 0]


def _use_base_lora(cfg: TrainConfig) -> bool:
    return bool(cfg.base_lora) and cfg.base_lora.strip().lower() not in ("none", "off", "")


def encode_workflow(image_folder: str, dataset_name: str, cfg: TrainConfig,
                    vae: str | None = None) -> dict:
    """Pictures + captions -> latents + conditioning saved once (cached latents):
    LoadImageTextDataSetFromFolder(input/<image_folder>) -> MakeTrainingDataset(vae, clip)
    -> SaveTrainingDataset(datasets/<dataset_name>)."""
    wf: dict = {}
    _node(wf, "1", "CheckpointLoaderSimple", "Load checkpoint", ckpt_name=cfg.checkpoint)
    clip, vae_out = ["1", 1], ["1", 2]
    if _use_base_lora(cfg):
        _node(wf, "2", "LoraLoader", "Pixel art LoRA", model=["1", 0], clip=clip,
              lora_name=cfg.base_lora, strength_model=cfg.base_lora_strength,
              strength_clip=cfg.base_lora_strength)
        clip = ["2", 1]
    if vae:
        vae_out = _node(wf, "3", "VAELoader", "Load VAE", vae_name=vae)
    _node(wf, "4", "LoadImageTextDataSetFromFolder", "Training pictures", folder=image_folder)
    _node(wf, "5", "MakeTrainingDataset", "Encode pictures + captions", images=["4", 0],
          texts=["4", 1], vae=vae_out, clip=clip)
    _node(wf, "6", "SaveTrainingDataset", "Save encoded dataset", latents=["5", 0],
          conditioning=["5", 1], folder_name=dataset_name, shard_size=1000)
    return wf


def train_workflow(dataset_name: str, cfg: TrainConfig, *, steps: int, seed: int,
                   existing_lora: str = "[None]", prefix: str) -> dict:
    """One chunk: CheckpointLoaderSimple -> [LoraLoader Pixel Art XL] -> TrainLoraNode(latents
    + conditioning from LoadTrainingDataset, continuing `existing_lora`) -> SaveLoRA(prefix,
    total steps). Input names are exactly comfy_extras/nodes_train.py's (ComfyUI 0.39)."""
    wf: dict = {}
    model = _node(wf, "1", "CheckpointLoaderSimple", "Load checkpoint", ckpt_name=cfg.checkpoint)
    if _use_base_lora(cfg):
        _node(wf, "2", "LoraLoader", "Pixel art LoRA", model=["1", 0], clip=["1", 1],
              lora_name=cfg.base_lora, strength_model=cfg.base_lora_strength,
              strength_clip=cfg.base_lora_strength)
        model = ["2", 0]
    _node(wf, "3", "LoadTrainingDataset", "Encoded dataset", folder_name=dataset_name)
    _node(wf, "4", "TrainLoraNode", "Train style LoRA", model=model, latents=["3", 0],
          positive=["3", 1], batch_size=cfg.batch_size, grad_accumulation_steps=1,
          steps=int(steps), learning_rate=cfg.learning_rate, rank=cfg.rank,
          optimizer=cfg.optimizer, loss_function="MSE", seed=int(seed) % 2**63,
          training_dtype="bf16", lora_dtype="bf16", quantized_backward=False, algorithm="LoRA",
          gradient_checkpointing=cfg.gradient_checkpointing, checkpoint_depth=1,
          offloading=cfg.offloading, existing_lora=existing_lora, bucket_mode=False,
          bypass_mode=False)
    _node(wf, "5", "SaveLoRA", "Save LoRA", lora=["4", 0], prefix=prefix, steps=["4", 2])
    return wf


def apply_style_lora(wf: dict, lora: str, strength: float = 0.8, nid: str = "15") -> dict:
    """THE generation hook: chain a style LoRA after the model/clip a generate.build_workflow
    workflow samples with (after Pixel Art XL, before the LCM LoRA's sampling patch is fine
    too). Rewires KSampler.model and both CLIPTextEncode.clip. Returns wf (changed in place)."""
    sampler = next((n for n in wf.values() if n.get("class_type") == "KSampler"), None)
    encoders = [n for n in wf.values() if n.get("class_type") == "CLIPTextEncode"]
    if sampler is None or not encoders:
        raise ValueError("Not a txt2img/img2img workflow (needs KSampler + CLIPTextEncode).")
    model_in = sampler["inputs"]["model"]
    # Put the style LoRA below ModelSamplingDiscrete (LCM) if present: patch the model it wraps.
    msd = wf.get(model_in[0]) if isinstance(model_in, list) else None
    target = msd if msd and msd.get("class_type") == "ModelSamplingDiscrete" else sampler
    _node(wf, nid, "LoraLoader", "Project style LoRA", model=target["inputs"]["model"],
          clip=encoders[0]["inputs"]["clip"], lora_name=lora, strength_model=float(strength),
          strength_clip=float(strength))
    target["inputs"]["model"] = [nid, 0]
    for e in encoders:
        e["inputs"]["clip"] = [nid, 1]
    return wf


# --------------------------------------------------------------------------- GPU temperature

@dataclass
class GpuStatus:
    temp: float
    util: float = 0.0
    mem_used: float = 0.0            # MiB
    mem_total: float = 0.0


def _num(text: str) -> float:
    try:
        return float(text.strip())
    except ValueError:
        return 0.0


def read_gpu() -> GpuStatus | None:
    """nvidia-smi query (read only; we never change power limits: that needs admin and is a
    system setting). None when nvidia-smi isn't there."""
    exe = shutil.which("nvidia-smi") or r"C:\Windows\System32\nvidia-smi.exe"
    kw: dict[str, Any] = {"capture_output": True, "text": True, "timeout": 8}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        out = subprocess.run([exe, "--query-gpu=temperature.gpu,utilization.gpu,memory.used,"
                              "memory.total", "--format=csv,noheader,nounits"], **kw)
    except (OSError, subprocess.SubprocessError):
        return None
    line = (out.stdout or "").strip().splitlines()
    if out.returncode != 0 or not line:
        return None
    parts = line[0].split(",")
    if not parts or not parts[0].strip().replace(".", "").isdigit():
        return None
    vals = [_num(p) for p in parts] + [0.0] * 4
    return GpuStatus(vals[0], vals[1], vals[2], vals[3])


# --------------------------------------------------------------------------- backends

StepFn = Callable[[int, int, float | None, float | None], None]   # step, of, loss, sec/step
AbortFn = Callable[[], str | None]                                # reason to interrupt, or None


class Backend(Protocol):
    def lora_dir(self) -> Path: ...
    def prepare(self, dataset: Dataset, name: str, cfg: TrainConfig) -> None: ...
    def train_chunk(self, name: str, cfg: TrainConfig, *, steps: int, seed: int,
                    existing: Path | None, prefix: str, on_step: StepFn,
                    should_abort: AbortFn) -> Path: ...


_TQDM = re.compile(r"Training LoRA:\s*\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]")


def parse_progress(text: str) -> tuple[int, int, float | None, float | None] | None:
    """Last tqdm line of TrainLoraNode -> (step, of, loss, seconds per step)."""
    found = list(_TQDM.finditer(text))
    if not found:
        return None
    m = found[-1]
    inside = m.group(3)
    loss = re.search(r"loss=([-\d.eE+]+)", inside)
    rate = re.search(r"([\d.]+)\s*(s/it|it/s)", inside)
    sps = None
    if rate:
        v = float(rate.group(1))
        sps = v if rate.group(2) == "s/it" else (1 / v if v else None)
    return int(m.group(1)), int(m.group(2)), float(loss.group(1)) if loss else None, sps


class ComfyBackend:
    """The real trainer: the local ComfyUI portable (needs its folder on this computer:
    pictures go into ComfyUI/input, checkpoints into ComfyUI/models/loras)."""

    def __init__(self, client=None, root: Path | str | None = None):
        from .generate import ComfyClient, find_comfyui
        self.client = client or ComfyClient(timeout=60)
        if root is None:
            portable = find_comfyui()
            if portable is None:
                raise StyleTrainError("Can't find ComfyUI on this computer (set COMFYUI_DIR to "
                                      "the ComfyUI_windows_portable folder). Training needs its "
                                      "folders, so a ComfyUI on another computer won't do.")
            root = portable / "ComfyUI"
        self.root = Path(root)
        self._pid: str | None = None

    def lora_dir(self) -> Path:
        return self.root / "models" / "loras"

    def _vae(self) -> str | None:
        try:
            names = self.client.list_models("vae")
        except Exception:  # noqa: BLE001
            return None
        return next((n for n in names if "sdxl_vae" in n.replace("\\", "/").lower()
                     or "fp16-fix" in n.lower()), None)

    def _check_nodes(self) -> None:
        try:
            info = self.client._get_json("/object_info/TrainLoraNode")
        except Exception:  # noqa: BLE001
            info = None
        if not info:
            raise StyleTrainError("This ComfyUI has no TrainLoraNode (it's built in since "
                                  "ComfyUI 0.3.4x; 0.39 has it). Update ComfyUI.")

    def _run(self, wf: dict, on_step: StepFn | None = None, should_abort: AbortFn | None = None,
             poll: float = 2.0) -> dict:
        from .generate import _execution_error
        pid = self.client.queue(wf)
        self._pid = pid
        try:
            while True:
                entry = (self.client._get_json(f"/history/{pid}") or {}).get(pid)
                if entry:
                    status = entry.get("status") or {}
                    if status.get("status_str") == "error":
                        msgs = [m[0] for m in status.get("messages") or [] if m]
                        if "execution_interrupted" in msgs:
                            raise Interrupted("interrupted")
                        err = _execution_error(status)
                        if "out of memory" in str(err).lower() or "OutOfMemory" in str(err):
                            raise StyleTrainError(
                                f"{err}\nThe GPU ran out of memory: set the resolution to 768 "
                                "(or turn on offloading) and Start again; it resumes.")
                        raise err
                    return entry
                if on_step is not None:
                    try:
                        raw = self.client._get_json("/internal/logs/raw")
                        lines = "".join(e.get("m", "") for e in (raw or {}).get("entries", [])[-60:])
                        p = parse_progress(lines)
                        if p:
                            on_step(*p)
                    except Exception:  # noqa: BLE001 - progress is best effort
                        pass
                if should_abort is not None:
                    why = should_abort()
                    if why:
                        self.interrupt()
                        self._wait_gone(pid)
                        raise Interrupted(why)
                time.sleep(poll)
        finally:
            self._pid = None

    def _wait_gone(self, pid: str, timeout: float = 120.0) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            try:
                if (self.client._get_json(f"/history/{pid}") or {}).get(pid):
                    return
            except Exception:  # noqa: BLE001
                return
            time.sleep(1.0)

    def interrupt(self) -> None:
        try:
            body = json.dumps({"prompt_id": self._pid} if self._pid else {}).encode()
            self.client._request("POST", "/interrupt", body, {"Content-Type": "application/json"})
        except Exception:  # noqa: BLE001
            pass

    def prepare(self, dataset: Dataset, name: str, cfg: TrainConfig) -> None:
        self.client.ensure_running()
        self._check_nodes()
        inp = self.root / "input" / Path(name)
        if inp.exists():
            shutil.rmtree(inp)
        shutil.copytree(dataset.folder, inp, ignore=shutil.ignore_patterns("*.json"))
        out = self.root / "datasets" / Path(name)
        if out.exists():
            shutil.rmtree(out)
        self._run(encode_workflow(name, name, cfg, self._vae()))

    def train_chunk(self, name: str, cfg: TrainConfig, *, steps: int, seed: int,
                    existing: Path | None, prefix: str, on_step: StepFn,
                    should_abort: AbortFn) -> Path:
        self.client.ensure_running()
        existing_name = "[None]"
        if existing is not None:
            names = self.client.list_models("loras")
            existing_name = next((n for n in names if Path(n.replace("\\", "/")).name == existing.name),
                                 None)
            if existing_name is None:
                raise StyleTrainError(f"ComfyUI doesn't list the checkpoint {existing.name} yet; "
                                      "press R in ComfyUI or restart it, then Start again.")
        started = time.time()
        self._run(train_workflow(name, cfg, steps=steps, seed=seed, existing_lora=existing_name,
                                 prefix=f"loras/{name}/{Path(prefix).name}"),
                  on_step, should_abort, poll=cfg.poll_seconds)
        outdir = self.root / "output" / "loras" / Path(name)
        found = [p for p in outdir.glob(f"{Path(prefix).name}_*_steps_*.safetensors")
                 if p.stat().st_mtime >= started - 5]
        if not found:
            raise StyleTrainError(f"ComfyUI finished but no LoRA was saved in {outdir}.")
        return max(found, key=lambda p: p.stat().st_mtime)


def _fake_safetensors(path: Path, meta: dict) -> Path:
    header = json.dumps({"__metadata__": {k: str(v) for k, v in meta.items()}}).encode()
    header += b" " * (-len(header) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    return path


class FakeBackend:
    """Dry run: no GPU, no ComfyUI. Steps tick instantly; LoRAs are tiny valid safetensors."""

    def __init__(self, root: Path, sec_per_step: float = 0.0, fail_at: int | None = None):
        self.root = Path(root)
        self.sec_per_step = sec_per_step
        self.fail_at = fail_at           # interrupt-like failure at this step of a chunk (tests)
        self.prepared: list[str] = []
        self.chunks: list[dict] = []

    def lora_dir(self) -> Path:
        return self.root / "loras"

    def prepare(self, dataset: Dataset, name: str, cfg: TrainConfig) -> None:
        if not any(dataset.folder.glob("*.png")):
            raise StyleTrainError("empty dataset")
        self.prepared.append(name)

    def train_chunk(self, name: str, cfg: TrainConfig, *, steps: int, seed: int,
                    existing: Path | None, prefix: str, on_step: StepFn,
                    should_abort: AbortFn) -> Path:
        before = 0
        if existing is not None:
            before = int(existing.name.split("_steps_")[0].split("_")[-1])
        self.chunks.append({"steps": steps, "seed": seed, "existing": existing and existing.name})
        for i in range(1, steps + 1):
            why = should_abort()
            if why:
                raise Interrupted(why)
            if self.fail_at is not None and i == self.fail_at:
                self.fail_at = None
                raise Interrupted("hot")
            if self.sec_per_step:
                time.sleep(self.sec_per_step)
            on_step(i, steps, 0.12 - 0.05 * (before + i) / (before + steps + 1), self.sec_per_step or None)
        out = self.root / "output" / f"{Path(prefix).name}_{before + steps}_steps_00001_.safetensors"
        return _fake_safetensors(out, {"steps": before + steps, "rank": cfg.rank})


# --------------------------------------------------------------------------- runner

@dataclass
class Progress:
    phase: str                       # dataset, encode, train, rest, cool, done, stopped, error
    step: int = 0                    # total steps done (all chunks)
    total: int = 0
    loss: float | None = None
    temp: float | None = None
    eta: float | None = None         # seconds left
    message: str = ""


@dataclass
class TrainState:
    project: str
    trigger: str
    signature: str
    total_steps: int
    done_steps: int = 0
    checkpoint: str = ""             # file in <loras>/pp_style/
    encoded: bool = False
    status: str = "new"              # new, training, stopped, done
    sec_per_step: float = 0.0        # measured
    elapsed: float = 0.0             # seconds of training so far (all sessions)
    losses: list = field(default_factory=list)   # [step, loss]
    notes: list = field(default_factory=list)
    result: str = ""


def state_path(project: Project) -> Path:
    return project.folder / "style_train" / "state.json"


def load_state(project: Project) -> TrainState | None:
    p = state_path(project)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return TrainState(**{k: v for k, v in d.items() if k in TrainState.__dataclass_fields__})
    except (OSError, ValueError, TypeError):
        return None


def save_state(project: Project, st: TrainState) -> None:
    p = state_path(project)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(st), indent=1), encoding="utf-8")
    os.replace(tmp, p)


class StyleTrainer:
    """Start / Stop / resume one project's style training. run() blocks (call it on a worker
    thread); stop() from any thread interrupts the current chunk and returns soon."""

    def __init__(self, project: Project, cfg: TrainConfig | None = None, *,
                 backend: Backend | None = None, gpu: Callable[[], GpuStatus | None] = read_gpu,
                 sleep: Callable[[float], None] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 progress: Callable[[Progress], None] | None = None):
        self.project = project
        self.cfg = cfg or TrainConfig()
        self._backend = backend
        self.gpu = gpu
        self.clock = clock
        self._stop = threading.Event()
        self._sleep = sleep or (lambda s: self._stop.wait(s))
        self.progress = progress or (lambda p: None)
        self.name = f"{SUBFOLDER}/{assets_mod.slug(project.name)}"
        self.state: TrainState | None = None
        self._last_temp: float | None = None

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            self._backend = ComfyBackend()
        return self._backend

    def stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    # -- helpers
    def _temp(self) -> float | None:
        g = self.gpu()
        self._last_temp = g.temp if g else None
        return self._last_temp

    def _emit(self, phase: str, message: str = "", loss: float | None = None,
              step: int | None = None) -> None:
        st = self.state
        eta = None
        done = step if step is not None else (st.done_steps if st else 0)
        if st and st.total_steps:
            sps = st.sec_per_step or SEC_PER_STEP.get(self.cfg.resolution, 3.5)
            left = max(0, st.total_steps - done)
            chunks = int(np.ceil(left / self.cfg.chunk_steps))
            eta = left * sps + chunks * CHUNK_OVERHEAD + max(0, chunks - 1) * self.cfg.rest_seconds
        self.progress(Progress(phase, done, st.total_steps if st else 0,
                               loss, self._last_temp, eta, message))

    def _ckpt_dir(self) -> Path:
        return self.backend.lora_dir() / SUBFOLDER

    def _rest(self, seconds: float, phase: str = "rest") -> None:
        end = self.clock() + seconds
        while not self.stopping and self.clock() < end:
            self._temp()
            self._emit(phase, f"Resting the GPU ({max(0, end - self.clock()):.0f} s)")
            self._sleep(min(5.0, max(0.0, end - self.clock())))

    def _cool_down(self) -> bool:
        """Wait while the GPU is hot. False = didn't cool in time (stopped, resumable)."""
        t = self._temp()
        if t is None or t < self.cfg.pause_above:
            return True
        start = self.clock()
        while not self.stopping:
            t = self._temp()
            if t is None or t <= self.cfg.resume_below:
                return True
            if self.clock() - start > self.cfg.max_cool_minutes * 60:
                return False
            self._emit("cool", f"GPU at {t:.0f} °C: waiting for {self.cfg.resume_below:.0f} °C")
            self._sleep(10.0)
        return True

    # -- the run
    def plan(self, samples: list[Sample]) -> dict:
        """What Start would do: images, steps, minutes (estimate), resume point."""
        total = self.cfg.steps or auto_steps(len(samples), self.cfg.chunk_steps)
        old = load_state(self.project)
        done = old.done_steps if old and old.status != "done" and old.total_steps == total else 0
        secs = estimate_seconds(len(samples), self.cfg, old.sec_per_step if old else None, done)
        return {"images": len(samples), "steps": total, "done": done, "seconds": secs,
                "minutes": secs / 60, "over_budget": secs / 60 > self.cfg.max_minutes,
                "few": len(samples) < MIN_IMAGES}

    def run(self, samples: list[Sample] | None = None) -> TrainState:
        cfg = self.cfg
        proj = self.project
        trigger = cfg.trigger or trigger_word(proj.name)
        if samples is None:
            samples = collect(proj, trigger=trigger)
        self._emit("dataset", f"Preparing {len(samples)} pictures…")
        ds = write_dataset(samples, proj.folder / "style_train" / "dataset", cfg, trigger)
        total = cfg.steps or auto_steps(ds.originals, cfg.chunk_steps)
        old = load_state(proj)
        ckdir = self._ckpt_dir()
        if (old and old.signature == ds.signature and old.status != "done" and old.checkpoint
                and (ckdir / old.checkpoint).exists()):
            st = old
            st.total_steps = max(total, st.done_steps)
            st.notes.append(f"resumed at step {st.done_steps}")
        elif old and old.signature == ds.signature and old.status == "done" and old.checkpoint \
                and (ckdir / old.checkpoint).exists():
            st = old                                    # "train more": continue the finished one
            st.total_steps = st.done_steps + total
            st.status = "training"
            st.notes.append(f"training more from step {st.done_steps}")
        else:
            st = TrainState(proj.name, trigger, ds.signature, total)
        self.state = st
        if len(samples) < MIN_IMAGES:
            st.notes.append(f"only {len(samples)} pictures (20+ recommended)")
        st.status = "training"
        save_state(proj, st)

        try:
            if not st.encoded:
                self._emit("encode", "Encoding the pictures once (VAE + text)…")
                if not self._cool_down():
                    return self._halt(st, "GPU didn't cool down; Start again later")
                self.backend.prepare(ds, self.name, cfg)
                st.encoded = True
                save_state(proj, st)
            session_start = self.clock()
            first = True
            hot_retries = 0
            while st.done_steps < st.total_steps:
                if self.stopping:
                    return self._halt(st, "stopped")
                if not first:
                    self._rest(cfg.rest_seconds)
                    if self.stopping:
                        return self._halt(st, "stopped")
                first = False
                if not self._cool_down():
                    return self._halt(st, "GPU didn't cool down; Start again later")
                if self.stopping:
                    return self._halt(st, "stopped")
                chunk = min(cfg.chunk_steps, st.total_steps - st.done_steps)
                # time budget for this session
                used = self.clock() - session_start
                sps = st.sec_per_step or SEC_PER_STEP.get(cfg.resolution, 3.5)
                need = chunk * sps + CHUNK_OVERHEAD
                if st.done_steps and used + need > cfg.max_minutes * 60:
                    st.notes.append(f"time budget ({cfg.max_minutes:.0f} min) reached at step "
                                    f"{st.done_steps}; Train more continues")
                    st.total_steps = st.done_steps
                    break
                path = self._chunk(st, chunk)
                if path is None:
                    if self.stopping:
                        return self._halt(st, "stopped")
                    hot_retries += 1                  # too hot: cool down and redo the chunk
                    if hot_retries >= 3:
                        return self._halt(st, f"GPU reached {cfg.hard_limit:.0f} °C in 3 chunks "
                                              "in a row; let the laptop cool / raise it for "
                                              "airflow, then Start again")
                    continue
                hot_retries = 0
            return self._finish(st)
        except Exception as e:
            st.status = "stopped"
            st.notes.append(f"error: {str(e).splitlines()[0] if str(e) else type(e).__name__}")
            save_state(proj, st)
            self._emit("error", str(e))
            raise

    def _chunk(self, st: TrainState, chunk: int) -> Path | None:
        cfg = self.cfg
        existing = self._ckpt_dir() / st.checkpoint if st.checkpoint else None
        base_done = st.done_steps
        t0 = self.clock()
        last_poll = [0.0]

        def on_step(step, of, loss, sps):
            st_done = base_done + step
            if loss is not None:
                st.losses.append([st_done, round(float(loss), 5)])
            if sps:
                st.sec_per_step = float(sps)
            # st.done_steps moves only with a saved checkpoint
            self._emit("train", f"Step {st_done} of {st.total_steps}", loss, step=st_done)

        def should_abort():
            if self.stopping:
                return "stopped"
            now = self.clock()
            if now - last_poll[0] >= cfg.poll_seconds:
                last_poll[0] = now
                t = self._temp()
                if t is not None and t >= cfg.hard_limit:
                    return f"GPU reached {t:.0f} °C"
            return None

        self._emit("train", f"Training steps {base_done + 1}-{base_done + chunk}")
        try:
            saved = self.backend.train_chunk(self.name, cfg, steps=chunk,
                                             seed=cfg.seed + base_done, existing=existing,
                                             prefix=f"{assets_mod.slug(self.project.name)}_c{base_done}",
                                             on_step=on_step, should_abort=should_abort)
        except Interrupted as e:
            st.notes.append(f"chunk at step {base_done} interrupted ({e.why})")
            save_state(self.project, st)
            self._emit("cool" if not self.stopping else "stopped", f"Chunk interrupted: {e.why}")
            return None
        took = self.clock() - t0
        st.elapsed += took
        if not st.sec_per_step and chunk:
            st.sec_per_step = max(0.0, (took - CHUNK_OVERHEAD) / chunk) or took / chunk
        st.done_steps = base_done + chunk
        ckdir = self._ckpt_dir()
        ckdir.mkdir(parents=True, exist_ok=True)
        name = f"{assets_mod.slug(self.project.name)}_ckpt_{st.done_steps}_steps_00001_.safetensors"
        dest = ckdir / name
        shutil.move(str(saved), dest)
        if st.checkpoint and st.checkpoint != name:
            try:
                (ckdir / st.checkpoint).unlink()
            except OSError:
                pass
        st.checkpoint = name
        save_state(self.project, st)
        self._emit("train", f"Checkpoint at step {st.done_steps}")
        return dest

    def _halt(self, st: TrainState, why: str) -> TrainState:
        st.status = "stopped"
        st.notes.append(why)
        save_state(self.project, st)
        self._emit("stopped", f"Stopped at step {st.done_steps} ({why}). Start resumes.")
        return st

    def _finish(self, st: TrainState) -> TrainState:
        if not st.checkpoint:
            return self._halt(st, "no steps trained")
        src = self._ckpt_dir() / st.checkpoint
        final = self.backend.lora_dir() / lora_file_name(self.project)
        shutil.copy2(src, final)
        st.status = "done"
        st.result = final.name
        save_state(self.project, st)
        record_style_lora(self.project, final.name, st.trigger, self.cfg.strength,
                          steps=st.done_steps, base_lora=self.cfg.base_lora)
        self._emit("done", f"Done: {final.name} ({st.done_steps} steps)")
        return st


# --------------------------------------------------------------------------- project record

RECORD = "style_lora.json"


def record_style_lora(project: Project, lora: str, trigger: str, strength: float = 0.8,
                      **extra) -> Path:
    """Remember the project's style LoRA: <project>/style_lora.json always; also
    Style.style_lora / style_trigger / style_strength when assets.Style has those fields."""
    p = project.folder / RECORD
    data = {"lora": lora, "trigger": trigger, "strength": strength, "trained": time.time(), **extra}
    p.write_text(json.dumps(data, indent=1), encoding="utf-8")
    style = project.style
    fields = getattr(type(style), "__dataclass_fields__", {})
    changed = False
    for k, v in (("style_lora", lora), ("style_trigger", trigger), ("style_strength", strength)):
        if k in fields:
            setattr(style, k, v)
            changed = True
    if changed:
        project.save_style(style)
    return p


def project_style_lora(project: Project | None = None) -> dict | None:
    """{"lora", "trigger", "strength"} for generation, or None. Style fields win over the
    json (so turning it off in the Style turns it off)."""
    proj = project or assets_mod.current_project()
    style = proj.style
    lora = getattr(style, "style_lora", None)
    data: dict = {}
    try:
        data = json.loads((proj.folder / RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    if lora is not None:                     # assets.Style knows about style LoRAs
        if not lora:
            return None
        return {"lora": lora, "trigger": getattr(style, "style_trigger", "") or data.get("trigger", ""),
                "strength": float(getattr(style, "style_strength", 0) or data.get("strength", 0.8))}
    if not data.get("lora"):
        return None
    return {"lora": data["lora"], "trigger": data.get("trigger", ""),
            "strength": float(data.get("strength", 0.8))}


def styled_prompt(prompt: str, trigger: str) -> str:
    """Prefix the trigger word ('<trigger> style, ...') unless it's already there."""
    if not trigger or trigger.lower() in prompt.lower():
        return prompt
    return f"{trigger} style, {prompt}"


# --------------------------------------------------------------------------- before / after

def before_after(project: Project, prompt: str, *, subject: str = "person", seed: int = 1234,
                 strength: float | None = None, client=None) -> tuple[Image.Image, Image.Image]:
    """Same prompt and seed, without and with the project's style LoRA (GPU)."""
    from . import generate as gen
    rec = project_style_lora(project)
    if not rec:
        raise StyleTrainError("This project has no trained style yet.")
    client = client or gen.ComfyClient()
    client.ensure_running()
    positive, negative = gen.sprite_prompt(prompt, subject)
    positive = styled_prompt(positive, rec["trigger"])
    vaes = client.list_models("vae")
    vae = next((n for n in vaes if "sdxl_vae" in n.replace("\\", "/").lower()
                or "fp16-fix" in n.lower()), None)
    loras = client.list_models("loras")
    style = next((n for n in loras if Path(n.replace("\\", "/")).name == rec["lora"]), rec["lora"])
    pixel = next((n for n in loras if Path(n.replace("\\", "/")).name == gen.DEFAULT_LORA), None)
    out = []
    for with_style in (False, True):
        wf = gen.build_workflow(positive, negative, seed=seed, lora=pixel, vae=vae,
                                filename_prefix="pixel-pipeline-style-test")
        if with_style:
            apply_style_lora(wf, style, rec["strength"] if strength is None else strength)
        pid = client.queue(wf)
        entry = client.wait(pid, 900)
        out.append(client.fetch_images(pid, entry)[0])
    return out[0], out[1]
