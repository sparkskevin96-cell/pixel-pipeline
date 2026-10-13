"""Text -> pixel-art sprite on your own NVIDIA GPU, via ComfyUI.

    prompt ──► ComfyUI (SDXL + Pixel Art XL LoRA) ──► raw ~1024 px image ──► convert() ──► sprite

The AI image always goes back through the normal converter, so the result lands on
the exact sprite size, palette and outline of your settings. img2img: pass a rough
sketch (e.g. a 32x48 sprite Claude drew) as `init_image`; it is upscaled with
nearest-neighbour, re-rendered by the GPU, then snapped back by the converter.

ComfyUI must be running (run_nvidia_gpu.bat, http://127.0.0.1:8188; set COMFYUI_URL
for another address). Setup guide: docs/LOCAL_AI.md. Dependency-free: urllib only.

    from pixel_pipeline import settings_for
    from pixel_pipeline.generate import generate_sprite
    for sprite, raw, seed in generate_sprite("a knight with a red cape", settings_for("person"),
                                             subject="person", count=2):
        sprite.save(f"knight_{seed}.png", scale=8)

Custom workflows (Flux, Qwen-Image, ...): export one from ComfyUI as API format, put
{{prompt}}, {{seed}}, ... where the values go and pass template="that.json"
(see PLACEHOLDERS). Examples: tools/comfyui/*.json.
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import io
import json
import logging
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from PIL import Image

from .pipeline import ImageSource, Sprite, convert, load_image
from .settings import SUBJECTS, Settings

log = logging.getLogger("pixel_pipeline.generate")

DEFAULT_URL = "http://127.0.0.1:8188"
URL_ENV = "COMFYUI_URL"
DEFAULT_CHECKPOINT = "sd_xl_base_1.0.safetensors"
DEFAULT_LORA = "pixel-art-xl.safetensors"
DEFAULT_LCM_LORA = "lcm-lora-sdxl.safetensors"
COMFY_DIR = r"ComfyUI_windows_portable\ComfyUI"

# SDXL's native ~1 megapixel sizes; we pick the one closest to the sprite's aspect.
SDXL_SIZES = ((1024, 1024), (1152, 896), (896, 1152), (1216, 832), (832, 1216),
              (1344, 768), (768, 1344), (1536, 640), (640, 1536))

MODEL_FOLDERS = {
    "checkpoints": r"models\checkpoints", "loras": r"models\loras", "vae": r"models\vae",
    "diffusion_models": r"models\diffusion_models", "text_encoders": r"models\text_encoders",
    "controlnet": r"models\controlnet", "upscale_models": r"models\upscale_models",
}
# Loader input name -> model folder (to explain "value not in list" errors).
LOADER_INPUTS = {"ckpt_name": "checkpoints", "lora_name": "loras", "vae_name": "vae",
                 "unet_name": "diffusion_models", "clip_name": "text_encoders",
                 "clip_name1": "text_encoders", "clip_name2": "text_encoders",
                 "control_net_name": "controlnet", "model_name": "upscale_models"}
# Fallback when GET /models/<folder> is missing (older ComfyUI): node + input listing the files.
_OBJECT_INFO = {"checkpoints": ("CheckpointLoaderSimple", "ckpt_name"),
                "loras": ("LoraLoader", "lora_name"), "vae": ("VAELoader", "vae_name"),
                "diffusion_models": ("UNETLoader", "unet_name"),
                "text_encoders": ("CLIPLoader", "clip_name"),
                "controlnet": ("ControlNetLoader", "control_net_name"),
                "upscale_models": ("UpscaleModelLoader", "model_name")}
DOWNLOADS = {
    DEFAULT_CHECKPOINT: "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 (6.9 GB)",
    DEFAULT_LORA: "https://huggingface.co/nerijs/pixel-art-xl (171 MB)",
    DEFAULT_LCM_LORA: "https://huggingface.co/latent-consistency/lcm-lora-sdxl "
                      "(pytorch_lora_weights.safetensors, 394 MB; rename it to lcm-lora-sdxl.safetensors)",
    "sdxl_vae.safetensors": "https://huggingface.co/madebyollin/sdxl-vae-fp16-fix (335 MB)",
}
# Other file names accepted for the defaults (compared as lower-case letters/digits only),
# e.g. sd_xl_base_1.0_0.9vae.safetensors or Civitai's Pixel_Art_XL_v1.1.safetensors.
_ALIASES = {DEFAULT_CHECKPOINT: ("sdxlbase",), DEFAULT_LORA: ("pixelartxl",),
            DEFAULT_LCM_LORA: ("lcmlorasdxl", "lcmsdxl")}


# --------------------------------------------------------------------------- errors

class ComfyUIError(RuntimeError):
    """Anything that went wrong talking to ComfyUI. The message is written for the user."""


class ComfyUIUnavailable(ComfyUIError):
    """ComfyUI isn't running (or isn't reachable) at the configured address."""


class ComfyUIWorkflowError(ComfyUIError):
    """ComfyUI rejected the workflow, or it failed while running (out of memory, ...)."""


class ComfyUIModelMissing(ComfyUIWorkflowError):
    """A model file the workflow needs isn't installed in ComfyUI."""

    def __init__(self, message: str, kind: str = "", name: str = "", installed=()):
        super().__init__(message)
        self.kind, self.name, self.installed = kind, name, list(installed)


class _HTTPStatus(Exception):
    def __init__(self, code: int, body: bytes):
        super().__init__(code)
        self.code, self.body = code, body


def _json(data: bytes, what: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ComfyUIError(f"ComfyUI sent an unreadable reply to {what}: {data[:200]!r}") from e


def _short(text: Any, limit: int = 300) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _base(name: str) -> str:
    return re.split(r"[\\/]", name)[-1]


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _base(name).lower())


def _combo_options(spec: Any) -> list | None:
    """Allowed values of a COMBO input: old [[...], {...}] or new ["COMBO", {"options": [...]}]."""
    if isinstance(spec, (list, tuple)) and spec:
        if isinstance(spec[0], (list, tuple)):
            return list(spec[0])
        if spec[0] == "COMBO" and len(spec) > 1 and isinstance(spec[1], dict):
            opts = spec[1].get("options")
            return list(opts) if isinstance(opts, (list, tuple)) else None
    return None


def _list_text(names: list[str], limit: int = 25) -> str:
    if not names:
        return "(none)"
    more = f" ... (+{len(names) - limit} more)" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


def _missing_hint(kind: str, name: str, installed: list[str]) -> str:
    folder = MODEL_FOLDERS.get(kind, f"models\\{kind}")
    where = DOWNLOADS.get(_base(str(name)))
    return (f"'{name}' isn't installed. Put it in {COMFY_DIR}\\{folder}"
            + (f" (download: {where})" if where else "")
            + ", then restart ComfyUI or press R in its browser tab to refresh the model lists. "
            f"Installed {kind}: {_list_text(installed)}.")


# --------------------------------------------------------------------------- start-up

DIR_ENV = "COMFYUI_DIR"   # folder holding run_nvidia_gpu.bat (ComfyUI_windows_portable)


def find_comfyui() -> Path | None:
    """The ComfyUI portable folder: $COMFYUI_DIR, else the usual places. None = not found."""
    home = Path.home()
    places = [os.environ.get(DIR_ENV)] + [str(p / "ComfyUI_windows_portable") for p in (
        Path(r"C:\AI"), Path("C:\\"), home, home / "Downloads", home / "Desktop",
        home / "OneDrive" / "Desktop", home / "Documents")]
    for place in places:
        if place and (Path(place) / "run_nvidia_gpu.bat").is_file():
            return Path(place)
    return None


def comfyui_command(folder: Path) -> list[str]:
    """The python line of run_nvidia_gpu.bat (keeps flags such as --lowvram), without
    the browser tab it opens."""
    args = ["-s", r"ComfyUI\main.py", "--windows-standalone-build"]
    try:
        for line in (folder / "run_nvidia_gpu.bat").read_text(errors="replace").splitlines():
            if "main.py" in line and "python" in line.lower():
                args = line.split()[1:]
                break
    except OSError:
        pass
    if "--disable-auto-launch" not in args:
        args.append("--disable-auto-launch")
    return [str(folder / "python_embeded" / "python.exe"), *args]


COMFYUI_LOG = Path.home() / ".pixel-pipeline" / "comfyui.log"
_started: list = []        # ComfyUI processes we started; stopped when we exit


def start_comfyui(folder: Path, extra_args: tuple = ()):
    """Start ComfyUI hidden (no window), output to COMFYUI_LOG. It is stopped when this
    program exits; a ComfyUI you started yourself is never touched. extra_args: more flags
    (training needs --disable-dynamic-vram)."""
    import atexit
    import subprocess
    COMFYUI_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(COMFYUI_LOG, "wb")  # noqa: SIM115 - the child writes to it while running
    kw: dict[str, Any] = {"cwd": str(folder), "stdout": log_file, "stderr": subprocess.STDOUT,
                          "stdin": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    cmd = comfyui_command(folder)
    cmd += [a for a in extra_args if a not in cmd]
    proc = subprocess.Popen(cmd, **kw)
    proc.pp_flags = tuple(extra_args)
    log_file.close()
    if not _started:
        atexit.register(stop_started_comfyui)
    _started.append(proc)
    return proc


def stop_started_comfyui() -> None:
    """Stop the ComfyUI processes start_comfyui started (frees the GPU memory)."""
    while _started:
        proc = _started.pop()
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except Exception:  # noqa: BLE001
                proc.kill()


def _log_tail(lines: int = 6) -> str:
    try:
        text = COMFYUI_LOG.read_text(errors="replace").strip().splitlines()
    except OSError:
        return ""
    return "\n".join(text[-lines:])


# --------------------------------------------------------------------------- client

class ComfyClient:
    """Minimal ComfyUI HTTP API client (urllib only).

    base_url defaults to $COMFYUI_URL or http://127.0.0.1:8188. `timeout` is per HTTP
    request; waiting for a generation uses wait()'s own timeout.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 30.0,
                 poll_interval: float = 0.5):
        url = (base_url or os.environ.get(URL_ENV) or DEFAULT_URL).strip().rstrip("/")
        self.base_url = url if "://" in url else f"http://{url}"
        self.timeout = timeout
        self.poll_interval = poll_interval
        self.client_id = uuid.uuid4().hex
        host = urllib.parse.urlsplit(self.base_url).hostname or ""
        # Never send local traffic through a system HTTP proxy.
        local = host in ("127.0.0.1", "localhost", "::1") or host.startswith("127.")
        self._opener = urllib.request.build_opener(
            *([urllib.request.ProxyHandler({})] if local else []))

    def __repr__(self) -> str:
        return f"ComfyClient({self.base_url!r})"

    # -- plumbing
    def _down(self, reason: Any) -> str:
        if isinstance(reason, TimeoutError) or "timed out" in str(reason):
            return (f"ComfyUI at {self.base_url} didn't answer within {self.timeout:.0f} s. "
                    "It may be frozen or still starting: check its console window.")
        return (f"ComfyUI isn't running at {self.base_url} ({reason}). Start it with "
                "run_nvidia_gpu.bat in your ComfyUI_windows_portable folder, wait until its "
                f"window says 'To see the GUI go to: {DEFAULT_URL}', then try again. "
                f"Different address or port? Set {URL_ENV} (ComfyUI Desktop uses "
                "http://127.0.0.1:8000).")

    def _request(self, method: str, path: str, body: bytes | None = None,
                 headers: dict | None = None, timeout: float | None = None) -> bytes:
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers or {},
                                     method=method)
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            try:
                payload = e.read()
            except OSError:
                payload = b""
            raise _HTTPStatus(e.code, payload) from None
        except urllib.error.URLError as e:
            raise ComfyUIUnavailable(self._down(e.reason)) from e
        except (OSError, http.client.HTTPException) as e:
            raise ComfyUIUnavailable(self._down(e)) from e

    def _get_json(self, path: str) -> Any:
        return _json(self._request("GET", path), path)

    # -- API
    def ping(self) -> dict:
        """GET /system_stats. Raises ComfyUIUnavailable with start-up instructions."""
        try:
            stats = self._get_json("/system_stats")
        except _HTTPStatus as e:
            raise ComfyUIUnavailable(f"Something answered at {self.base_url} (HTTP {e.code}) but it "
                                     f"isn't ComfyUI. Check the port, or set {URL_ENV}.") from None
        except ComfyUIUnavailable:
            raise
        except ComfyUIError as e:
            raise ComfyUIUnavailable(f"Something answered at {self.base_url} but it isn't "
                                     f"ComfyUI ({e}).") from None
        if not isinstance(stats, dict) or "system" not in stats:
            raise ComfyUIUnavailable(f"Something answered at {self.base_url} but it isn't ComfyUI.")
        return stats

    def ensure_running(self, wait: float = 240.0, say=None, extra_args: tuple = ()) -> dict:
        """ping(), but if nothing answers at the default address, start the installed
        ComfyUI portable (find_comfyui) hidden in the background and wait for it.
        extra_args: flags it must run with; a ComfyUI we started without them is restarted
        (one you started yourself is used as it is)."""
        try:
            info = self.ping()
            ours = [p for p in _started if p.poll() is None]
            if not extra_args or not ours or all(set(extra_args) <= set(getattr(p, "pp_flags", ()))
                                                  for p in ours):
                return info
            (say or log.info)(f"Restarting ComfyUI with {' '.join(extra_args)}…")
            stop_started_comfyui()
            time.sleep(3)
            folder = find_comfyui()
            if folder is None:
                raise ComfyUIUnavailable("ComfyUI portable not found")
        except ComfyUIUnavailable:
            if self.base_url.replace("localhost", "127.0.0.1") != DEFAULT_URL:
                raise
            folder = find_comfyui()
            if folder is None:
                raise
        (say or log.info)(f"Starting ComfyUI from {folder}…")
        proc = start_comfyui(folder, extra_args) if extra_args else start_comfyui(folder)
        deadline = time.monotonic() + wait
        while True:
            time.sleep(2)
            try:
                return self.ping()
            except ComfyUIUnavailable:
                ended = proc is not None and proc.poll() is not None
                if ended or time.monotonic() > deadline:
                    why = "stopped" if ended else f"didn't answer within {wait:.0f} s"
                    tail = _log_tail()
                    raise ComfyUIUnavailable(
                        f"Started ComfyUI from {folder} but it {why}. Its output is in "
                        f"{COMFYUI_LOG}" + (f"; last lines:\n{tail}" if tail else ".")) from None

    def list_models(self, kind: str) -> list[str]:
        """Installed model files in a folder: checkpoints, loras, vae, diffusion_models, ..."""
        names: Any = None
        try:
            names = self._get_json(f"/models/{urllib.parse.quote(kind)}")
        except _HTTPStatus as e:
            if e.code != 404:
                raise ComfyUIError(f"ComfyUI couldn't list {kind} (HTTP {e.code}).") from None
        if not isinstance(names, list):
            if kind not in _OBJECT_INFO:
                raise ComfyUIError(f"ComfyUI has no model folder called '{kind}'.")
            node, inp = _OBJECT_INFO[kind]
            try:
                info = self._get_json(f"/object_info/{node}")
            except _HTTPStatus as e:
                raise ComfyUIError(f"ComfyUI couldn't list {kind} (HTTP {e.code}).") from None
            spec = ((((info or {}).get(node) or {}).get("input") or {}).get("required") or {}).get(inp)
            names = _combo_options(spec)
            if names is None:
                raise ComfyUIError(f"ComfyUI couldn't list {kind}.")
        return [str(n) for n in names]

    def queue(self, workflow: dict) -> str:
        """POST /prompt. Returns the prompt id; a rejected workflow raises a readable error."""
        body = json.dumps({"prompt": workflow, "client_id": self.client_id}).encode("utf-8")
        try:
            data = _json(self._request("POST", "/prompt", body,
                                       {"Content-Type": "application/json"}), "/prompt")
        except _HTTPStatus as e:
            try:
                detail = json.loads(e.body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise ComfyUIWorkflowError(f"ComfyUI rejected the workflow (HTTP {e.code}): "
                                           f"{_short(e.body.decode('utf-8', 'replace'))}") from None
            raise self._rejection(detail) from None
        pid = data.get("prompt_id") if isinstance(data, dict) else None
        if not pid:
            raise ComfyUIWorkflowError(f"ComfyUI didn't queue the workflow: {_short(data)}")
        return str(pid)

    def _rejection(self, data: Any) -> ComfyUIWorkflowError:
        """ComfyUI's 400 {"error": ..., "node_errors": ...} -> one readable exception."""
        data = data if isinstance(data, dict) else {"error": data}
        problems, missing = [], []
        for nid, ne in (data.get("node_errors") or {}).items():
            cls = (ne or {}).get("class_type") or "?"
            for err in (ne or {}).get("errors") or []:
                info = err.get("extra_info") or {}
                inp = info.get("input_name")
                if err.get("type") == "value_not_in_list" and inp in LOADER_INPUTS:
                    value = info.get("received_value")
                    missing.append((LOADER_INPUTS[inp], str(value),
                                    _combo_options(info.get("input_config")) or []))
                    problems.append(f"{cls} (node {nid}): model file '{value}' is not installed")
                else:
                    det = _short(err.get("details"))
                    problems.append(f"{cls} (node {nid}): {err.get('message') or err.get('type')}"
                                    + (f" ({det})" if det else ""))
        err = data.get("error")
        if not problems and err:
            if isinstance(err, dict):
                det = _short(err.get("details"))
                problems.append(f"{err.get('message') or err.get('type')}" + (f" ({det})" if det else ""))
            else:
                problems.append(_short(err))
        text = "ComfyUI rejected the workflow:\n" + "\n".join(
            f"  - {p}" for p in problems or ["(no details)"])
        if not missing:
            return ComfyUIWorkflowError(text)
        installed: list[str] = []
        for kind, name, allowed in missing:
            try:
                installed = self.list_models(kind)
            except ComfyUIError:
                installed = [str(a) for a in allowed]
            text += "\n" + _missing_hint(kind, name, installed)
        kind, name, _ = missing[0]
        return ComfyUIModelMissing(text, kind, name, installed)

    def wait(self, prompt_id: str, timeout: float = 900.0) -> dict:
        """Poll GET /history/<id> until the prompt finished. Returns its history entry."""
        deadline = time.monotonic() + timeout
        path = f"/history/{urllib.parse.quote(prompt_id)}"
        while True:
            try:
                hist = self._get_json(path)
            except _HTTPStatus as e:
                raise ComfyUIError(f"ComfyUI's history failed (HTTP {e.code}).") from None
            except ComfyUIUnavailable as e:
                raise ComfyUIUnavailable(f"ComfyUI stopped answering while generating (crashed, or "
                                         f"out of memory?). {e}") from None
            entry = hist.get(prompt_id) if isinstance(hist, dict) else None
            if entry:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise _execution_error(status)
                return entry
            if time.monotonic() >= deadline:
                raise ComfyUIError(f"ComfyUI didn't finish within {timeout:.0f} s (prompt {prompt_id}). "
                                   "It may still be busy (the first run loads the models from disk): "
                                   "check its console window, then try again.")
            time.sleep(self.poll_interval)

    def fetch_images(self, prompt_id: str, entry: dict | None = None) -> list[Image.Image]:
        """The images a finished prompt produced (SaveImage outputs; PreviewImage if none)."""
        if entry is None:
            entry = (self._get_json(f"/history/{urllib.parse.quote(prompt_id)}") or {}).get(prompt_id) or {}
        outputs = entry.get("outputs") or {}
        refs = []
        for nid in sorted(outputs, key=lambda k: (0, int(k), "") if str(k).isdigit() else (1, 0, str(k))):
            refs += [r for r in (outputs[nid] or {}).get("images") or [] if isinstance(r, dict)]
        saved = [r for r in refs if r.get("type", "output") == "output"]
        refs = saved or refs
        if not refs:
            raise ComfyUIWorkflowError("ComfyUI finished but returned no images. The workflow needs a "
                                       "SaveImage (or PreviewImage) node.")
        images = []
        for r in refs:
            query = urllib.parse.urlencode({"filename": r.get("filename", ""),
                                            "subfolder": r.get("subfolder", ""),
                                            "type": r.get("type", "output")})
            try:
                data = self._request("GET", f"/view?{query}")
            except _HTTPStatus as e:
                raise ComfyUIError(f"ComfyUI couldn't send image {r.get('filename')} "
                                   f"(HTTP {e.code}).") from None
            img = Image.open(io.BytesIO(data))
            img.load()
            images.append(img)
        return images

    def fetch_files(self, prompt_id: str, entry: dict | None = None,
                    suffixes: tuple[str, ...] = (".glb", ".gltf", ".obj", ".ply", ".stl"),
                    ) -> list[tuple[str, bytes]]:
        """Files a finished prompt saved whose names end in `suffixes` (e.g. SaveGLB's
        "3d" outputs): [(filename, bytes)]."""
        if entry is None:
            entry = (self._get_json(f"/history/{urllib.parse.quote(prompt_id)}") or {}).get(prompt_id) or {}
        refs = []
        for nid in sorted(entry.get("outputs") or {}, key=str):
            for items in (entry["outputs"][nid] or {}).values():
                if isinstance(items, list):
                    refs += [r for r in items if isinstance(r, dict)
                             and str(r.get("filename", "")).lower().endswith(suffixes)]
        files = []
        for r in refs:
            query = urllib.parse.urlencode({"filename": r.get("filename", ""),
                                            "subfolder": r.get("subfolder", ""),
                                            "type": r.get("type", "output")})
            try:
                files.append((str(r["filename"]), self._request("GET", f"/view?{query}", timeout=120)))
            except _HTTPStatus as e:
                raise ComfyUIError(f"ComfyUI couldn't send {r.get('filename')} (HTTP {e.code}).") from None
        return files

    def upload_image(self, image: ImageSource, name: str | None = None) -> str:
        """POST /upload/image into ComfyUI's input folder. Returns the name LoadImage wants."""
        img = image if isinstance(image, Image.Image) else Image.fromarray(load_image(image))
        buf = io.BytesIO()
        img.save(buf, "PNG")
        png = buf.getvalue()
        name = name or f"pixel-pipeline-init-{hashlib.sha1(png).hexdigest()[:12]}.png"
        boundary = f"----pixelpipeline{uuid.uuid4().hex}"
        parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
                  f'filename="{name}"\r\nContent-Type: image/png\r\n\r\n').encode() + png + b"\r\n"]
        for key, value in (("overwrite", "true"), ("type", "input")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n'
                         f"{value}\r\n".encode())
        parts.append(f"--{boundary}--\r\n".encode())
        try:
            data = _json(self._request("POST", "/upload/image", b"".join(parts),
                                       {"Content-Type": f"multipart/form-data; boundary={boundary}"}),
                         "/upload/image")
        except _HTTPStatus as e:
            raise ComfyUIError(f"ComfyUI refused the image upload (HTTP {e.code}): "
                               f"{_short(e.body.decode('utf-8', 'replace'))}") from None
        got, sub = data.get("name") or name, data.get("subfolder") or ""
        return f"{sub}/{got}" if sub else got


def _execution_error(status: dict) -> ComfyUIWorkflowError:
    for item in status.get("messages") or []:
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            continue
        event, data = item
        data = data if isinstance(data, dict) else {}
        if event == "execution_interrupted":
            return ComfyUIWorkflowError("Generation was cancelled in ComfyUI (interrupted).")
        if event == "execution_error":
            etype, emsg = data.get("exception_type") or "Error", str(data.get("exception_message") or "")
            text = (f"ComfyUI failed in {data.get('node_type') or '?'} (node {data.get('node_id')}): "
                    f"{etype}: {_short(emsg, 500)}")
            low = f"{etype} {emsg}".lower()
            if "out of memory" in low or "outofmemory" in low:
                text += ("\nThe GPU ran out of memory. Close other GPU-heavy apps (games, Godot, video), "
                         "keep the size near 1024x1024 (SDXL's native size), or add --lowvram to the "
                         "python line in run_nvidia_gpu.bat and restart ComfyUI.")
            elif "filenotfound" in low or "no such file" in low:
                text += f"\nA model file is missing or was moved: check {COMFY_DIR}\\models."
            return ComfyUIWorkflowError(text)
    return ComfyUIWorkflowError("ComfyUI reported an error without details; check its console window.")


def gpu_summary(stats: dict) -> str:
    """'ComfyUI 0.39.0, NVIDIA GeForce RTX 4080 Laptop GPU, 12.0 GB VRAM (10.9 GB free)'."""
    parts = []
    version = (stats.get("system") or {}).get("comfyui_version")
    if version:
        parts.append(f"ComfyUI {version}")
    for dev in stats.get("devices") or []:
        name = str(dev.get("name", "?")).split(" : ")[0]
        name = re.sub(r"^cuda:\d+\s+", "", name)
        total, free = dev.get("vram_total") or 0, dev.get("vram_free") or 0
        parts.append(f"{name}, {total / 2**30:.1f} GB VRAM ({free / 2**30:.1f} GB free)" if total else name)
    return ", ".join(parts) or "ComfyUI"


# --------------------------------------------------------------------------- models

def resolve_model(client: ComfyClient, kind: str, wanted: str, default: str | None = None, *,
                  auto: bool = True, installed: list[str] | None = None) -> str:
    """The installed name for `wanted` (subfolders allowed). If `wanted` is the default and
    `auto`, a similarly named file is accepted. Raises ComfyUIModelMissing otherwise."""
    if installed is None:
        installed = client.list_models(kind)
    want = _base(wanted).lower()
    for name in installed:
        if name == wanted or _base(name).lower() == want:
            return name
    if auto and default and wanted == default:
        for alias in _ALIASES.get(default, ()):
            for name in installed:
                if alias in _norm(name):
                    log.info("Using %s '%s' (instead of %s)", kind, name, default)
                    return name
    raise ComfyUIModelMissing(f"ComfyUI model missing: {_missing_hint(kind, wanted, installed)}",
                              kind, wanted, installed)


# --------------------------------------------------------------------------- options

@dataclass
class GenOptions:
    """Generation knobs (all optional). None = default; the defaults suit SDXL + Pixel Art XL."""
    checkpoint: str = DEFAULT_CHECKPOINT   # any SDXL checkpoint in models/checkpoints
    lora: str = DEFAULT_LORA               # pixel-art LoRA in models/loras; "none" = no LoRA
    style_lora: str | None = None          # a project's trained style (styletrain.py); None =
    style_lora_strength: float = 0.8       # the current project's, "" = off
    lora_strength: float | None = None     # 1.0 (1.2 with fast)
    vae: str | None = None                 # e.g. sdxl_vae.safetensors (fp16-fix); None = checkpoint's
    fast: bool = False                     # LCM-LoRA: 8 steps, cfg 1.5 (~3x faster, a bit rougher)
    lcm_lora: str = DEFAULT_LCM_LORA       # used when fast
    steps: int | None = None               # 25 (8 with fast)
    cfg: float | None = None               # 6.0 (1.5 with fast)
    sampler: str | None = None             # euler_ancestral (lcm with fast)
    scheduler: str | None = None           # normal (sgm_uniform with fast)
    width: int | None = None               # generation size; None = SDXL size nearest the
    height: int | None = None              # sprite's aspect (1024x1024 for square sprites)
    negative: str | None = None            # None = built-in negative prompt; "" = none
    shape_prompt: bool = True              # wrap the prompt for the subject; False = send as-is
    template: str | None = None            # API-format workflow JSON with {{placeholders}}
    filename_prefix: str = "pixel-pipeline"  # ComfyUI output/ file names
    timeout: float = 900.0                 # seconds to wait for each image
    auto_models: bool = True               # accept similarly named files for the default models

    def sampling(self) -> dict:
        fast = self.fast
        return {"steps": self.steps or (8 if fast else 25),
                "cfg": self.cfg if self.cfg is not None else (1.5 if fast else 6.0),
                "sampler": self.sampler or ("lcm" if fast else "euler_ancestral"),
                "scheduler": self.scheduler or ("sgm_uniform" if fast else "normal"),
                "lora_strength": self.lora_strength if self.lora_strength is not None
                else (1.2 if fast else 1.0)}

    @property
    def use_lora(self) -> bool:
        return bool(self.lora) and self.lora.strip().lower() not in ("none", "off", "false", "0")


GEN_OPTION_NAMES = tuple(f.name for f in fields(GenOptions))


def gen_options(**opts) -> GenOptions:
    """GenOptions from keyword args; None values keep the default (like settings_for)."""
    unknown = set(opts) - set(GEN_OPTION_NAMES)
    if unknown:
        raise TypeError(f"Unknown generation option(s): {', '.join(sorted(unknown))} "
                        f"(valid: {', '.join(GEN_OPTION_NAMES)})")
    out = GenOptions(**{k: v for k, v in opts.items() if v is not None})
    if out.template is not None:
        out.template = str(out.template)
    return out


# --------------------------------------------------------------------------- prompts

SUBJECT_PROMPTS = {
    "person": "pixel art sprite of {prompt}, single character, full body, standing, side view, "
              "centered, plain white background, simple background, game asset, flat colors, "
              "clean outline",
    "animal": "pixel art sprite of {prompt}, single animal, full body, side view, centered, "
              "plain white background, simple background, game asset, flat colors, clean outline",
    "object": "pixel art sprite of {prompt}, single item, centered, plain white background, "
              "simple background, game item icon, game asset, flat colors, clean outline",
    "background": "pixel art landscape background of {prompt}, side-scroller game background, "
                  "wide scenic view, detailed environment, no characters",
}
_NEG_COMMON = ("3d render, realistic, photo, blurry, soft focus, smooth gradient, noise, "
               "jpeg artifacts, text, watermark, signature, logo")
NEGATIVE_PROMPTS = {
    **{s: _NEG_COMMON + ", multiple characters, multiple views, sprite sheet, cropped, out of "
          "frame, cut off, busy background, scenery, cast shadow, frame, border"
       for s in ("person", "animal", "object")},
    "background": _NEG_COMMON + ", characters, people, user interface, frame, border",
}
SYSTEM_STYLES = {
    "nes": "8-bit NES game style, very limited palette",
    "snes": "16-bit SNES game style",
    "gameboy": "original Game Boy style, 4 shades of green",
    "gbc": "Game Boy Color style, 8-bit",
    "sms": "8-bit Sega Master System style",
    "genesis": "16-bit Sega Genesis style",
    "gba": "Game Boy Advance style",
    "pico8": "PICO-8 style, tiny chunky pixels",
}
SMALL_HINT = "simple, low detail, chunky pixels, bold readable silhouette"


def sprite_prompt(prompt: str, subject: str = "object", *, system: str | None = None,
                  size: tuple[int, int | None] | None = None) -> tuple[str, str]:
    """User prompt -> (positive, negative) prompts that give a clean, cut-out-able sprite."""
    if subject not in SUBJECT_PROMPTS:
        raise ValueError(f"Unknown subject '{subject}' (use {', '.join(SUBJECTS)})")
    text = re.sub(r"^\s*(an?\s+)?pixel[- ]?art\s+(sprite\s+)?(of\s+)?", "", prompt or "",
                  flags=re.IGNORECASE).strip().rstrip(".,; ")
    if not text:
        raise ValueError("Empty prompt: describe what to draw, e.g. 'a knight with a red cape'.")
    positive = SUBJECT_PROMPTS[subject].format(prompt=text)
    extras = []
    if system and system.lower() not in ("off", "none"):
        extras.append(SYSTEM_STYLES.get(system.lower(), f"{system} style"))
    if size and subject != "background":
        w, h = size[0], size[1] or size[0]
        if max(w, h) <= 24:
            extras.append(SMALL_HINT)
    if extras:
        positive += ", " + ", ".join(extras)
    return positive, NEGATIVE_PROMPTS[subject]


def guess_subject(settings: Settings) -> str:
    """Best guess when the caller didn't say: background-like settings, else object."""
    return "background" if not settings.remove_background and settings.fit == "cover" else "object"


# --------------------------------------------------------------------------- workflows

def build_workflow(prompt: str, negative: str = "", *, seed: int, width: int = 1024,
                   height: int = 1024, image: str | None = None, denoise: float = 1.0,
                   checkpoint: str = DEFAULT_CHECKPOINT, lora: str | None = DEFAULT_LORA,
                   lora_strength: float = 1.0, lcm_lora: str | None = None, vae: str | None = None,
                   steps: int = 25, cfg: float = 6.0, sampler: str = "euler_ancestral",
                   scheduler: str = "normal", filename_prefix: str = "pixel-pipeline",
                   mask: str | None = None, grow_mask_by: int | None = None) -> dict:
    """ComfyUI API-format workflow from core nodes only.

    txt2img: CheckpointLoaderSimple -> [LoraLoader] -> 2x CLIPTextEncode -> EmptyLatentImage
             -> KSampler -> VAEDecode -> SaveImage
    img2img (image = uploaded name): LoadImage -> VAEEncode -> KSampler(denoise) instead.
    inpaint (image + mask = uploaded mask picture, white = repaint): LoadImageMask(red) ->
             SetLatentNoiseMask on the VAEEncode latent (keeps the init under the noise, so a
             denoise below 1 still starts from it); grow_mask_by = VAEEncodeForInpaint instead
             (greys the masked pixels: for denoise 1.0 / inpainting checkpoints).
    lcm_lora adds a second LoraLoader + ModelSamplingDiscrete(lcm); vae adds a VAELoader.
    """
    wf: dict[str, dict] = {}

    def node(nid: str, cls: str, title: str, **inputs) -> None:
        wf[nid] = {"class_type": cls, "inputs": inputs, "_meta": {"title": title}}

    node("1", "CheckpointLoaderSimple", "Load checkpoint", ckpt_name=checkpoint)
    model, clip, vae_out = ["1", 0], ["1", 1], ["1", 2]
    if lora:
        node("2", "LoraLoader", "Pixel art LoRA", model=model, clip=clip, lora_name=lora,
             strength_model=lora_strength, strength_clip=lora_strength)
        model, clip = ["2", 0], ["2", 1]
    if lcm_lora:
        node("3", "LoraLoader", "LCM LoRA (fast)", model=model, clip=clip, lora_name=lcm_lora,
             strength_model=1.0, strength_clip=1.0)
        model, clip = ["3", 0], ["3", 1]
        node("4", "ModelSamplingDiscrete", "LCM sampling", model=model, sampling="lcm", zsnr=False)
        model = ["4", 0]
    if vae:
        node("5", "VAELoader", "Load VAE", vae_name=vae)
        vae_out = ["5", 0]
    node("6", "CLIPTextEncode", "Prompt", text=prompt, clip=clip)
    node("7", "CLIPTextEncode", "Negative prompt", text=negative, clip=clip)
    if image is None:
        node("8", "EmptyLatentImage", "Empty canvas", width=width, height=height, batch_size=1)
        latent = ["8", 0]
    else:
        node("8", "LoadImage", "Init image", image=image)
        if mask is not None:
            node("13", "LoadImageMask", "Inpaint mask", image=mask, channel="red")
        if mask is not None and grow_mask_by is not None:
            node("9", "VAEEncodeForInpaint", "Encode for inpaint", pixels=["8", 0], vae=vae_out,
                 mask=["13", 0], grow_mask_by=int(grow_mask_by))
            latent = ["9", 0]
        else:
            node("9", "VAEEncode", "Encode init image", pixels=["8", 0], vae=vae_out)
            latent = ["9", 0]
            if mask is not None:
                node("14", "SetLatentNoiseMask", "Repaint only the mask", samples=latent,
                     mask=["13", 0])
                latent = ["14", 0]
    node("10", "KSampler", "Sampler", model=model, seed=seed, steps=steps, cfg=cfg,
         sampler_name=sampler, scheduler=scheduler, positive=["6", 0], negative=["7", 0],
         latent_image=latent, denoise=denoise if image is not None else 1.0)
    node("11", "VAEDecode", "Decode", samples=["10", 0], vae=vae_out)
    node("12", "SaveImage", "Save", images=["11", 0], filename_prefix=filename_prefix)
    return wf


def txt2img_workflow(prompt: str, negative: str = "", **kw) -> dict:
    return build_workflow(prompt, negative, image=None, **kw)


def img2img_workflow(prompt: str, negative: str, image: str, *, denoise: float = 0.6, **kw) -> dict:
    return build_workflow(prompt, negative, image=image, denoise=denoise, **kw)


def inpaint_workflow(prompt: str, negative: str, image: str, mask: str, *, denoise: float = 0.85,
                     **kw) -> dict:
    """img2img that repaints only where the uploaded `mask` picture is white."""
    return build_workflow(prompt, negative, image=image, mask=mask, denoise=denoise, **kw)


def mask_array(mask) -> np.ndarray:
    """A mask (float/bool array 0..1, uint8 0..255 array, or picture: its grey value; RGBA
    pictures use alpha) -> float32 (H, W) in 0..1."""
    if isinstance(mask, np.ndarray) and mask.ndim == 2:
        m = mask.astype(np.float32)
        return np.clip(m / 255 if mask.dtype == np.uint8 else m, 0, 1)
    if isinstance(mask, Image.Image):
        img = mask
    elif isinstance(mask, np.ndarray):
        img = Image.fromarray(mask)
    else:
        img = Image.open(mask)
    if img.mode in ("RGBA", "LA"):
        return np.asarray(img.getchannel("A"), np.float32) / 255
    return np.asarray(img.convert("L"), np.float32) / 255


def composite_masked(base: Image.Image, painted: Image.Image, mask) -> Image.Image:
    """painted where mask is 1, base where it's 0 (blended between): pixels outside the mask
    come back exactly as in `base`. painted is resized to base's size if it differs."""
    b = np.asarray(base.convert("RGB"), np.float32)
    p = painted.convert("RGB")
    if p.size != base.size:
        p = p.resize(base.size, Image.LANCZOS)
    pa = np.asarray(p, np.float32)
    m = mask_array(mask)
    if m.shape != b.shape[:2]:
        m = np.asarray(Image.fromarray(m).resize(base.size, Image.BILINEAR), np.float32)
    m = np.clip(m, 0, 1)[..., None]
    out = pa * m + b * (1 - m)
    return Image.fromarray(np.rint(out).astype(np.uint8), "RGB")


PLACEHOLDERS = {
    "prompt": "positive prompt (already shaped for the subject unless shape_prompt=False)",
    "negative": "negative prompt",
    "seed": "integer seed",
    "width": "generation width", "height": "generation height",
    "image": "uploaded init image name (img2img; LoadImage's 'image')",
    "denoise": "img2img strength 0-1 (1.0 for txt2img)",
    "steps": "sampling steps", "cfg": "CFG scale",
    "sampler": "KSampler sampler_name", "scheduler": "KSampler scheduler",
    "checkpoint": "checkpoint file name", "lora": "pixel-art LoRA file name",
    "lora_strength": "LoRA strength",
    "prefix": "SaveImage filename_prefix",
}
_PH = re.compile(r"\{\{\s*(\w+)\s*\}\}")


def load_template(src: str | Path | dict) -> dict:
    """Read an API-format workflow ("Export (API)" in ComfyUI) used as a template."""
    if isinstance(src, dict):
        wf: Any = src
    else:
        path = Path(src)
        if not path.is_file():
            raise FileNotFoundError(f"Workflow template not found: {path}")
        try:
            wf = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Workflow template {path} isn't valid JSON: {e}") from e
    if isinstance(wf, dict) and isinstance(wf.get("prompt"), dict) and "nodes" not in wf:
        wf = wf["prompt"]  # a saved /prompt request body
    if isinstance(wf, dict) and "nodes" in wf and "links" in wf:
        raise ValueError("That's a ComfyUI UI workflow, not the API format. In ComfyUI use "
                         "Workflow > Export (API) (older versions: enable Dev mode in the settings, "
                         "then 'Save (API Format)').")
    if not (isinstance(wf, dict) and wf
            and all(isinstance(n, dict) and "class_type" in n for n in wf.values())):
        raise ValueError("Not a ComfyUI API-format workflow (expected {node_id: {class_type, inputs}}).")
    return copy.deepcopy(wf)


def template_placeholders(wf: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(wf, str):
        found.update(m.group(1) for m in _PH.finditer(wf))
    elif isinstance(wf, dict):
        for v in wf.values():
            found |= template_placeholders(v)
    elif isinstance(wf, list):
        for v in wf:
            found |= template_placeholders(v)
    return found


def _braces(names) -> str:
    return ", ".join("{{" + n + "}}" for n in names)


def fill_template(wf: dict, values: dict) -> dict:
    """Replace {{name}} placeholders. A value that is exactly "{{seed}}" becomes the typed
    value (int/float); inside longer text it's substituted as text."""
    used = template_placeholders(wf)
    unknown = used - set(PLACEHOLDERS)
    if unknown:
        raise ValueError(f"Unknown template placeholder(s): {_braces(sorted(unknown))} "
                         f"(supported: {_braces(PLACEHOLDERS)})")
    missing = sorted(n for n in used if values.get(n) is None)
    if missing:
        if "image" in missing:
            raise ValueError("This workflow template is img2img ({{image}}): give it an init image.")
        raise ValueError(f"No value for template placeholder(s): {_braces(missing)}")

    def sub(v: Any) -> Any:
        if isinstance(v, str):
            whole = _PH.fullmatch(v.strip())
            if whole:
                return values[whole.group(1)]
            return _PH.sub(lambda m: str(values[m.group(1)]), v)
        if isinstance(v, list):
            return [sub(x) for x in v]
        if isinstance(v, dict):
            return {k: sub(x) for k, x in v.items()}
        return v

    return sub(copy.deepcopy(wf))


# --------------------------------------------------------------------------- generation

class Generated(NamedTuple):
    sprite: Sprite        # converted: exact size + palette of the settings
    image: Image.Image    # raw image from ComfyUI
    seed: int


def _snap8(v: float) -> int:
    return max(64, round(v / 8) * 8)


def generation_size(aspect: float | None = None, width: int | None = None,
                    height: int | None = None) -> tuple[int, int]:
    """Explicit size (snapped to multiples of 8), else the SDXL size nearest `aspect` (w/h)."""
    if width and height:
        return _snap8(width), _snap8(height)
    if width or height:
        a = aspect or 1.0
        return (_snap8(width), _snap8(width / a)) if width else (_snap8(height * a), _snap8(height))
    if not aspect or aspect <= 0:
        return 1024, 1024
    return min(SDXL_SIZES, key=lambda s: abs(math.log(s[0] / s[1]) - math.log(aspect)))


def prepare_init_image(src: ImageSource, width: int, height: int,
                       background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """Sketch -> RGB generation-size canvas: transparency flattened onto `background`,
    small sketches upscaled by a whole-number factor with nearest-neighbour, centred."""
    rgba = load_image(src)
    h, w = rgba.shape[:2]
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = rgba[..., :3].astype(np.float32) * a + np.array(background, np.float32) * (1 - a)
    img = Image.fromarray(np.rint(rgb).astype(np.uint8))
    scale = min(width / w, height / h)
    if scale >= 1:
        k = int(scale)
        nw, nh = w * k, h * k
        img = img.resize((nw, nh), Image.NEAREST)
    else:
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        img = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (width, height), background)
    canvas.paste(img, ((width - nw) // 2, (height - nh) // 2))
    return canvas


def _models(client: ComfyClient, opts: GenOptions, needed: set[str]) -> dict:
    """Installed names for the models this run needs (auto-picks for the defaults)."""
    cache: dict[str, list[str]] = {}

    def find(kind: str, wanted: str, default: str | None) -> str:
        if kind not in cache:
            cache[kind] = client.list_models(kind)
        return resolve_model(client, kind, wanted, default, auto=opts.auto_models,
                             installed=cache[kind])

    out: dict[str, str | None] = {"checkpoint": opts.checkpoint, "lora": None, "lcm_lora": None,
                                  "vae": opts.vae}
    if "checkpoint" in needed:
        out["checkpoint"] = find("checkpoints", opts.checkpoint, DEFAULT_CHECKPOINT)
    if "lora" in needed and opts.use_lora:
        out["lora"] = find("loras", opts.lora, DEFAULT_LORA)
    if "lcm_lora" in needed and opts.fast:
        out["lcm_lora"] = find("loras", opts.lcm_lora, DEFAULT_LCM_LORA)
    if "vae" in needed and opts.vae:
        out["vae"] = find("vae", opts.vae, None)
    elif "vae" in needed and opts.auto_models:
        # The fp16-fix VAE avoids speckles when SDXL runs in fp16; use it whenever it's installed.
        cache["vae"] = client.list_models("vae")
        out["vae"] = next((n for n in cache["vae"] if "sdxl_vae" in n.replace("\\", "/").lower()
                           or "fp16-fix" in n.lower()), None)
    return out


def generate_images(prompt: str, negative: str = "", *, client: ComfyClient | None = None,
                    seed: int | None = None, count: int = 1, init_image: ImageSource | None = None,
                    denoise: float = 0.6, aspect: float | None = None, mask=None,
                    grow_mask_by: int | None = None,
                    **gen_opts) -> list[tuple[Image.Image, int]]:
    """Raw ComfyUI images, no conversion: [(image, seed)]. Prompts are sent as given
    (a "negative" in gen_opts is ignored here; pass `negative`).

    mask (with init_image): inpaint. Same size as init_image, white/1 = repaint (see
    mask_array for the forms it takes). Only the masked part is redrawn and the result is
    composited over the init in Python, so pixels outside the mask come back unchanged.

    count > 1 queues one prompt per image with seeds seed, seed+1, ... so each image can
    be re-made from its own seed.
    """
    opts = gen_options(**gen_opts)
    if count < 1:
        raise ValueError("count must be at least 1")
    if init_image is not None and not 0 < denoise <= 1:
        raise ValueError("denoise must be between 0 and 1 (0.3 = stay close to the sketch, "
                         "0.8 = redraw freely)")
    if mask is not None and init_image is None:
        raise ValueError("Inpainting needs an init image to paint into (init_image).")
    if mask is not None and opts.template:
        raise ValueError("Workflow templates can't inpaint (mask); use the built-in workflow.")
    template = load_template(opts.template) if opts.template else None
    used = template_placeholders(template) if template is not None else set()
    if template is not None:
        if init_image is not None and "image" not in used:
            raise ValueError("The workflow template has no {{image}} placeholder, so it can't use "
                             "an init image (img2img).")
        if init_image is None and "image" in used:
            raise ValueError("This workflow template is img2img ({{image}}): give it an init image.")

    client = client or ComfyClient()
    client.ensure_running()
    init = load_image(init_image) if init_image is not None else None
    if init is not None and aspect is None:
        aspect = init.shape[1] / init.shape[0]
    width, height = generation_size(aspect, opts.width, opts.height)
    needed = used if template is not None else {"checkpoint", "lora", "lcm_lora", "vae"}
    models = _models(client, opts, needed)
    prepared = prepare_init_image(init, width, height) if init is not None else None
    image_name = client.upload_image(prepared) if prepared is not None else None
    mask_name, placed_mask = None, None
    if mask is not None:
        m = mask_array(mask)
        if m.shape != init.shape[:2]:
            raise ValueError(f"The mask is {m.shape[1]}x{m.shape[0]} but the init image is "
                             f"{init.shape[1]}x{init.shape[0]}")
        grey = np.rint(m * 255).astype(np.uint8)
        rgba = np.dstack([grey, grey, grey, np.full_like(grey, 255)])
        placed = prepare_init_image(rgba, width, height, background=(0, 0, 0))  # same placement
        placed_mask = np.asarray(placed.convert("L"), np.float32) / 255
        mask_name = client.upload_image(placed)
    samp = opts.sampling()

    base = random.randrange(2**32) if seed is None else int(seed)
    results = []
    for i in range(count):
        s = (base + i) % 2**64
        if template is not None:
            wf = fill_template(template, {
                "prompt": prompt, "negative": negative, "seed": s, "width": width,
                "height": height, "image": image_name,
                "denoise": denoise if image_name else 1.0, "steps": samp["steps"],
                "cfg": samp["cfg"], "sampler": samp["sampler"], "scheduler": samp["scheduler"],
                "checkpoint": models["checkpoint"], "lora": models["lora"],
                "lora_strength": samp["lora_strength"], "prefix": opts.filename_prefix})
        else:
            wf = build_workflow(prompt, negative, seed=s, width=width, height=height,
                                image=image_name, denoise=denoise, checkpoint=models["checkpoint"],
                                lora=models["lora"], lora_strength=samp["lora_strength"],
                                lcm_lora=models["lcm_lora"], vae=models["vae"],
                                steps=samp["steps"], cfg=samp["cfg"], sampler=samp["sampler"],
                                scheduler=samp["scheduler"], filename_prefix=opts.filename_prefix,
                                mask=mask_name, grow_mask_by=grow_mask_by)
            _apply_project_style(wf, opts)
        pid = client.queue(wf)
        log.info("Queued ComfyUI prompt %s (seed %d)", pid, s)
        entry = client.wait(pid, opts.timeout)
        images = client.fetch_images(pid, entry)
        if placed_mask is not None:
            images = [composite_masked(prepared, img, placed_mask) for img in images]
        results += [(img, s) for img in images]
    return results


def generate_sprite(prompt: str, settings: Settings, *, client: ComfyClient | None = None,
                    subject: str | None = None, system: str | None = None,
                    seed: int | None = None, count: int = 1,
                    init_image: ImageSource | None = None, denoise: float = 0.6,
                    palette: np.ndarray | None = None, style=None,
                    **gen_opts) -> list[Generated]:
    """Prompt -> ComfyUI -> convert() -> sprites on the settings' exact grid and palette.

    subject: person | animal | object | background, shapes the prompt (default: guessed
    from the settings). system: retro look name (snes, nes, ...) adds style words; the
    caller still applies the system's settings. init_image: sketch for img2img; denoise =
    how much the GPU may change it. palette: force one palette (e.g. shared across a set).
    style: the house style (assets.Style) appended to shaped prompts (None = the current
    project's, False = none). gen_opts: see GenOptions. Returns [(sprite, raw_image, seed)].
    Raises ComfyUIUnavailable / ComfyUIModelMissing / ComfyUIWorkflowError (all
    ComfyUIError), ValueError for bad arguments, TypeError for unknown options.
    """
    opts = gen_options(**gen_opts)
    subject = subject or guess_subject(settings)
    if opts.shape_prompt:
        positive, negative = sprite_prompt(prompt, subject, system=system,
                                           size=(settings.width, settings.height))
        if style is not False:                                  # house style (assets.py)
            from .assets import project_style, with_style
            positive, negative = with_style(positive, negative, style or project_style(),
                                            "character" if subject in ("person", "animal")
                                            else "sprite")
    else:
        positive, negative = (prompt or "").strip(), ""
        if not positive:
            raise ValueError("Empty prompt: describe what to draw.")
    if opts.negative is not None:
        negative = opts.negative
    aspect = settings.width / settings.height if settings.height else None
    if init_image is not None:
        aspect = None  # follow the sketch's shape
    rest = {k: v for k, v in gen_opts.items() if k != "negative"}
    raw = generate_images(positive, negative, client=client, seed=seed, count=count,
                          init_image=init_image, denoise=denoise, aspect=aspect, **rest)
    return [Generated(convert(img, settings, palette), img, s) for img, s in raw]


def check_setup(client: ComfyClient | None = None, **gen_opts) -> tuple[bool, str]:
    """Is ComfyUI running with the models these options need? -> (ok, report text).
    Raises ComfyUIUnavailable if ComfyUI isn't running."""
    opts = gen_options(**gen_opts)
    client = client or ComfyClient()
    stats = client.ping()
    lines = [f"{client.base_url}: {gpu_summary(stats)}"]
    ok = True
    checks = [("checkpoint", "checkpoints", opts.checkpoint, DEFAULT_CHECKPOINT, True)]
    if opts.use_lora:
        checks.append(("pixel-art LoRA", "loras", opts.lora, DEFAULT_LORA, True))
    checks.append(("LCM LoRA (fast mode)", "loras", opts.lcm_lora, DEFAULT_LCM_LORA, opts.fast))
    if opts.vae:
        checks.append(("VAE", "vae", opts.vae, None, True))
    cache: dict[str, list[str]] = {}
    for label, kind, wanted, default, required in checks:
        if kind not in cache:
            cache[kind] = client.list_models(kind)
        try:
            got = resolve_model(client, kind, wanted, default, auto=opts.auto_models,
                                installed=cache[kind])
            lines.append(f"  {label}: {got}" + (f" (used for {wanted})" if _base(got) != wanted else ""))
        except ComfyUIModelMissing:
            if required:
                ok = False
                lines.append(f"  {label}: MISSING. {_missing_hint(kind, wanted, cache[kind])}")
            else:
                lines.append(f"  {label}: not installed (optional; download: "
                             f"{DOWNLOADS.get(wanted, wanted)})")
    lines.append("Ready." if ok else "Not ready: install the missing model(s) above.")
    return ok, "\n".join(lines)


def _apply_project_style(wf: dict, opts) -> None:
    """Chain the project's trained style LoRA (Train style) into a built-in workflow and put its
    trigger word in the prompt. opts.style_lora: None = the current project's, "" = off."""
    lora, strength, trigger = opts.style_lora, opts.style_lora_strength, ""
    if lora is None:
        try:
            from .styletrain import project_style_lora
            found = project_style_lora()
        except Exception:  # noqa: BLE001 - no project, no style: plain generation
            found = None
        if not found:
            return
        lora, trigger, strength = found["lora"], found.get("trigger", ""), found.get("strength", strength)
    if not lora:
        return
    from .styletrain import apply_style_lora, styled_prompt
    apply_style_lora(wf, lora, strength)
    if trigger:
        for n in wf.values():
            if n.get("class_type") == "CLIPTextEncode" and n.get("_meta", {}).get("title") == "Prompt":
                n["inputs"]["text"] = styled_prompt(n["inputs"]["text"], trigger)
