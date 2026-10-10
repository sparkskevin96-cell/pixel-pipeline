"""Prompt -> sprite from online pixel-art AIs, side by side with your own GPU.

Providers:
    pixellab        PixelLab (pixellab.ai) REST v2: create-image-pixen (fast), pixflux,
                    bitforge, or generate-image-v2 ("pro", queued as a background job).
    retrodiffusion  Retro Diffusion (retrodiffusion.ai) v2: rd_fast / rd_plus / rd_pro styles,
                    always queued; ~$0.03 (fast), $0.06 (plus), $0.18 (pro) per image.
    gpu             ComfyUI on this PC (generate.py), free.

Only the prompt text (and size/seed/options) is sent; nothing of yours is uploaded. Keys come
from the environment (PIXELLAB_API_KEY, RETRO_DIFFUSION_API_KEY) or ~/.pixel-pipeline/keys.json
(the desktop app's Keys button writes it). The services draw real pixel art at the sprite's
size, so their results are only snapped to your palette (no resampling); results at another
size go through the normal converter.
"""
from __future__ import annotations

import base64
import io
import json
import os
import textwrap
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .export import pixel_scale
from .pipeline import Prepared, Sprite, convert, finish, load_image
from .settings import Settings

PROVIDERS = {
    "pixellab": {"name": "PixelLab", "env": "PIXELLAB_API_KEY", "url_env": "PIXELLAB_BASE_URL",
                 "url": "https://api.pixellab.ai/v2", "signup": "https://www.pixellab.ai/pixellab-api"},
    "retrodiffusion": {"name": "Retro Diffusion", "env": "RETRO_DIFFUSION_API_KEY",
                       "url_env": "RETRO_DIFFUSION_BASE_URL", "url": "https://api.retrodiffusion.ai/v2",
                       "signup": "https://retrodiffusion.ai/app/devtools"},
    "gpu": {"name": "Your GPU (ComfyUI)"},
}
PIXELLAB_MODELS = {"pixen": "/create-image-pixen", "pixflux": "/create-image-pixflux",
                   "bitforge": "/create-image-bitforge", "pro": "/generate-image-v2"}
PIXELLAB_VIEWS = ("side", "low top-down", "high top-down")
PIXELLAB_DIRECTIONS = ("south", "south-east", "east", "north-east", "north", "north-west",
                       "west", "south-west")
RD_STYLES = ("rd_fast__default", "rd_fast__retro", "rd_fast__simple", "rd_plus__default",
             "rd_plus__retro", "rd_plus__isometric", "rd_pro__default", "rd_pro__fantasy",
             "rd_pro__painterly", "rd_pro__isometric")
# Size range per Retro Diffusion family (GET /v2/styles/selector has the live numbers).
RD_SIZES = {"rd_fast": (64, 384), "rd_plus": (64, 384), "rd_pro": (16, 256)}
KEYS_FILE = Path.home() / ".pixel-pipeline" / "keys.json"
POLL_SECONDS = 2.0      # queued jobs: how often to ask


class OnlineError(RuntimeError):
    """A service said no (bad key, no credits, bad size) or couldn't be reached."""


# --------------------------------------------------------------------------- keys

def load_keys(path: Path | None = None, env: bool = True) -> dict[str, str]:
    """{provider: key} from the keys file, overridden by environment variables (unless not env).
    The file also holds "github", the token the desktop app checks for updates with."""
    keys: dict[str, str] = {}
    path = path or KEYS_FILE
    try:
        keys.update({k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if v})
    except (OSError, ValueError):
        pass
    for prov, info in PROVIDERS.items() if env else ():
        if "env" in info and os.environ.get(info["env"]):
            keys[prov] = os.environ[info["env"]].strip()
    return keys


def save_keys(keys: dict[str, str], path: Path | None = None) -> Path:
    """Store keys for this user only (the desktop app and the command line share them)."""
    path = path or KEYS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v.strip() for k, v in keys.items() if v and v.strip()}, indent=2),
                    encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


def _key(provider: str, keys: dict | None) -> str:
    key = (keys if keys is not None else load_keys()).get(provider)
    if not key:
        info = PROVIDERS[provider]
        raise OnlineError(f"No {info['name']} API key. Get one at {info['signup']}, then add it "
                          f"with the Keys button (or set {info['env']}).")
    return key


def _base(provider: str) -> str:
    info = PROVIDERS[provider]
    return os.environ.get(info["url_env"], info["url"]).rstrip("/")


# --------------------------------------------------------------------------- HTTP

def _http(method: str, url: str, headers: dict, body: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Accept": "application/json", "User-Agent": "pixel-pipeline",
                                          **({"Content-Type": "application/json"} if data else {}),
                                          **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(detail)
            detail = parsed.get("detail") or parsed.get("error") or parsed.get("message") or detail
        except ValueError:
            pass
        hint = {401: " (check the API key)", 402: " (out of credits)", 429: " (too many requests; "
                "wait a moment)"}.get(e.code, "")
        raise OnlineError(f"{url.split('/v2')[0]} answered {e.code}{hint}: {str(detail)[:400]}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise OnlineError(f"Couldn't reach {url.split('/v2')[0]}: {e}") from None
    try:
        return json.loads(raw) if raw else {}
    except ValueError:
        raise OnlineError(f"{url} didn't answer with JSON") from None


def _images(payload, size: tuple[int, int] | None = None) -> list[Image.Image]:
    """Every base64 image anywhere in a response (PNG, or raw RGBA of the requested size)."""
    found: list[str] = []

    def walk(v):
        if isinstance(v, dict):
            if isinstance(v.get("base64"), str):
                found.append(v["base64"])
            for k, x in v.items():
                if k == "base64_images" and isinstance(x, list):
                    found.extend(s for s in x if isinstance(s, str))
                elif k != "base64":
                    walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    walk(payload)
    out = []
    for b64 in found:
        if "," in b64[:64] and "base64" in b64[:64]:
            b64 = b64.split(",", 1)[1]
        data = base64.b64decode(b64)
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            out.append(img.convert("RGBA"))
        except Exception:  # noqa: BLE001 - PixelLab can send raw RGBA bytes
            if size and len(data) == size[0] * size[1] * 4:
                out.append(Image.frombytes("RGBA", size, data))
    return out


def _cost(payload) -> str:
    """Whatever the response says it cost."""
    if not isinstance(payload, dict):
        return ""
    for part in (payload, payload.get("last_response")):
        usage = part.get("usage") if isinstance(part, dict) else None
        if isinstance(usage, dict) and isinstance(usage.get("usd"), (int, float)):
            return f"${float(usage['usd']):.3f}"
    result = payload.get("result")
    if isinstance(result, dict) and "balance_cost" in result:
        left = result.get("remaining_balance")
        return f"{result['balance_cost']} credits" + (f" ({left} left)" if left is not None else "")
    return ""


# --------------------------------------------------------------------------- PixelLab

def pixellab_generate(prompt: str, width: int, height: int, *, key: str | None = None,
                      model: str = "pixen", view: str | None = None, direction: str | None = None,
                      transparent: bool = True, seed: int | None = None, poll: float | None = None,
                      timeout: float = 300, keys: dict | None = None) -> tuple[list[Image.Image], str]:
    """Text -> pixel art on PixelLab. Returns ([images], cost text)."""
    if model not in PIXELLAB_MODELS:
        raise ValueError(f"PixelLab model must be one of {', '.join(PIXELLAB_MODELS)}")
    key = key or _key("pixellab", keys)
    headers = {"Authorization": f"Bearer {key}"}
    body: dict = {"description": prompt, "image_size": {"width": width, "height": height},
                  "no_background": transparent}
    if seed is not None:
        body["seed"] = int(seed) % 2**31
    if view:
        body["view"] = view
    if direction:
        body["direction"] = direction
    base = _base("pixellab")
    resp = _http("POST", base + PIXELLAB_MODELS[model], headers, body)
    job = resp.get("background_job_id") or resp.get("job_id")
    if job:
        resp = _poll(lambda: _http("GET", f"{base}/background-jobs/{job}", headers),
                     ("completed", "complete", "succeeded", "success"), ("failed", "error"),
                     poll or POLL_SECONDS, timeout, "PixelLab")
        resp = resp.get("last_response", resp)
    images = _images(resp, (width, height))
    if not images:
        raise OnlineError(f"PixelLab answered without an image: {str(resp)[:300]}")
    return images, _cost(resp)


def pixellab_balance(key: str | None = None, keys: dict | None = None) -> str:
    resp = _http("GET", _base("pixellab") + "/balance",
                 {"Authorization": f"Bearer {key or _key('pixellab', keys)}"})
    usd = resp.get("usd", resp.get("balance"))
    return f"${float(usd):.2f}" if isinstance(usd, (int, float)) else str(resp)[:100]


# --------------------------------------------------------------------------- Retro Diffusion

def rd_request_size(style: str, width: int, height: int) -> tuple[int, int]:
    """The sprite size, or the smallest whole multiple of it the style accepts."""
    lo, hi = RD_SIZES.get(style.split("__")[0], (12, 512))
    k = max(1, -(-lo // min(width, height)))
    w, h = width * k, height * k
    if max(w, h) > hi:
        scale = hi / max(w, h)
        w, h = max(lo, int(w * scale)), max(lo, int(h * scale))
    return w, h


def retrodiffusion_generate(prompt: str, width: int, height: int, *, key: str | None = None,
                            style: str = "rd_fast__default", transparent: bool = True,
                            seed: int | None = None, count: int = 1, poll: float | None = None,
                            timeout: float = 300, keys: dict | None = None,
                            check_cost: bool = False) -> tuple[list[Image.Image], str]:
    """Text -> pixel art on Retro Diffusion. Returns ([images], cost text)."""
    key = key or _key("retrodiffusion", keys)
    w, h = rd_request_size(style, width, height)
    body: dict = {"prompt": prompt, "prompt_style": style, "width": w, "height": h,
                  "num_images": count, "remove_bg": transparent}
    if seed is not None:
        body["seed"] = int(seed) % 2**31
    if check_cost:
        body["check_cost"] = True
    base = _base("retrodiffusion")
    headers = {"X-RD-Token": key}
    resp = _http("POST", base + "/inferences",
                 {**headers, **({} if check_cost else {"Idempotency-Key": str(uuid.uuid4())})}, body)
    if check_cost:
        return [], _cost(resp) or str(resp.get("balance_cost", resp))[:80]
    task = resp.get("task_id")
    if task:
        resp = _poll(lambda: _http("GET", f"{base}/inferences/tasks/{task}", headers),
                     ("succeeded",), ("failed",), poll or POLL_SECONDS, timeout, "Retro Diffusion")
    images = _images(resp.get("result", resp), (w, h))
    if not images:
        raise OnlineError(f"Retro Diffusion answered without an image: {str(resp)[:300]}")
    return images, _cost(resp)


def retrodiffusion_balance(key: str | None = None, keys: dict | None = None) -> str:
    resp = _http("GET", _base("retrodiffusion") + "/inferences/credits",
                 {"X-RD-Token": key or _key("retrodiffusion", keys)})
    return f"{resp.get('balance', resp.get('credits', '?'))} credits"


def _poll(get, done: tuple, failed: tuple, every: float, timeout: float, who: str) -> dict:
    end = time.monotonic() + timeout
    while True:
        resp = get()
        status = str(resp.get("status", "")).lower()
        if status in done:
            return resp
        if status in failed:
            raise OnlineError(f"{who} failed: {resp.get('error') or resp.get('detail') or resp}")
        if time.monotonic() > end:
            raise OnlineError(f"{who} took longer than {timeout:.0f} s; try again later")
        time.sleep(every)


# --------------------------------------------------------------------------- results

def to_sprite(raw: Image.Image | np.ndarray, settings: Settings,
              palette: np.ndarray | None = None) -> Sprite:
    """A service's pixel art -> sprite on your settings. At the sprite's size (or a clean
    whole-number enlargement of it) the pixels are kept and only snapped to the palette;
    anything else goes through the converter."""
    rgba = load_image(raw)
    k = pixel_scale(rgba)
    if k > 1:
        rgba = np.ascontiguousarray(rgba[::k, ::k])
    h, w = rgba.shape[:2]
    want_h = settings.height or round(settings.width * h / w)
    if (w, h) != (settings.width, want_h):
        return convert(rgba, settings, palette)
    mask = rgba[..., 3] >= settings.alpha_threshold
    rgb = rgba[..., :3].copy()
    if settings.system not in (None, "off"):
        from .systems import tone_map
        rgb = tone_map(settings.system, rgb, mask)
    # already drawn as pixel art: keep its outline and highlights
    return finish(Prepared(rgb, mask), replace(settings, outline=False, cleanup=False), palette)


@dataclass
class Candidate:
    provider: str
    label: str
    raw: Image.Image | None = None
    sprite: Sprite | None = None
    seed: int | None = None
    cost: str = ""
    seconds: float = 0.0
    error: str = ""


def prompt_candidates(prompt: str, settings: Settings, providers: list[str], *,
                      seed: int | None = None, keys: dict | None = None,
                      pixellab_model: str = "pixen", rd_style: str = "rd_fast__default",
                      view: str | None = None, direction: str | None = None,
                      comfy_client=None, progress=lambda text: None) -> list[Candidate]:
    """Make the same prompt with every provider; one failing doesn't stop the others."""
    if not prompt or not prompt.strip():
        raise ValueError("Describe what to make.")
    keys = load_keys() if keys is None else keys
    w, h = settings.width, settings.height or settings.width
    out = []
    for prov in providers:
        if prov not in PROVIDERS:
            raise ValueError(f"Unknown provider '{prov}' (use {', '.join(PROVIDERS)})")
        label = {"pixellab": f"model: {pixellab_model}", "retrodiffusion": f"style: {rd_style}",
                 "gpu": "SDXL + pixel LoRA"}[prov]
        cand = Candidate(prov, label, seed=seed)
        progress(f"{PROVIDERS[prov]['name']}: making “{prompt}”…")
        t0 = time.monotonic()
        try:
            if prov == "pixellab":
                images, cand.cost = pixellab_generate(prompt, w, h, model=pixellab_model, view=view,
                                                      direction=direction, seed=seed, keys=keys)
                cand.raw, cand.sprite = images[0], to_sprite(images[0], settings)
            elif prov == "retrodiffusion":
                images, cand.cost = retrodiffusion_generate(prompt, w, h, style=rd_style, seed=seed,
                                                            keys=keys)
                cand.raw, cand.sprite = images[0], to_sprite(images[0], settings)
            else:   # your GPU makes a big picture; the converter turns it into the sprite
                from .generate import generate_sprite
                (gen,) = generate_sprite(prompt, settings, client=comfy_client, seed=seed)
                cand.sprite, cand.raw, cand.seed, cand.cost = gen.sprite, gen.image, gen.seed, "free"
        except Exception as e:  # noqa: BLE001 - shown next to the others
            cand.error = str(e) or type(e).__name__
        cand.seconds = time.monotonic() - t0
        out.append(cand)
    return out


def compare_sheet(cands: list[Candidate], scale: int | None = None) -> Image.Image:
    """One row per provider: what the service drew, and the sprite on your settings. A row is as
    tall as its pictures or its text, whichever is taller, so error messages never overlap."""
    sprites = [c.sprite for c in cands if c.sprite is not None]
    sw = max((s.rgba.shape[1] for s in sprites), default=48)
    sh = max((s.rgba.shape[0] for s in sprites), default=48)
    k = scale or max(2, min(8, 192 // max(sw, sh)))
    cell_w, cell_h, text_w, gap, line_h = sw * k, sh * k, 230, 12, 14
    rows = []
    for c in cands:
        lines = [PROVIDERS[c.provider]["name"], c.label]
        if c.error:
            lines += textwrap.wrap(c.error, 36, break_long_words=True)[:8]
        else:
            lines.append(f"{c.seconds:.0f} s  {c.cost}".strip())
        rows.append((c, lines, max(cell_h, 8 + line_h * len(lines))))
    width = text_w + 2 * (cell_w + gap) + gap
    height = gap + sum(h + gap for *_, h in rows)
    sheet = Image.new("RGB", (width, height), (34, 38, 52))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    y = gap
    for c, lines, row_h in rows:
        for j, line in enumerate(lines):
            draw.text((gap, y + 4 + line_h * j), line,
                      fill=(235, 235, 235) if j == 0 else (170, 178, 196), font=font)
        for col, img in enumerate((c.raw, c.sprite.rgba if c.sprite else None)):
            if img is None:
                continue
            im = img if isinstance(img, Image.Image) else Image.fromarray(img, "RGBA")
            fit = min(cell_w / im.width, cell_h / im.height)
            size = (max(1, int(im.width * fit)), max(1, int(im.height * fit)))
            im = im.convert("RGBA").resize(size, Image.NEAREST)
            x = text_w + gap + col * (cell_w + gap)
            bg = Image.new("RGBA", (cell_w, cell_h), (52, 58, 76, 255))
            bg.alpha_composite(im, ((cell_w - size[0]) // 2, cell_h - size[1]))
            sheet.paste(bg.convert("RGB"), (x, y))
        y += row_h + gap
    return sheet


def write_candidates(cands: list[Candidate], out_png: Path) -> list[Path]:
    """<name>_<provider>.png (your settings), <name>_<provider>.raw.png, <name>_compare.png."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for c in cands:
        if c.sprite is None:
            continue
        p = out_png.with_name(f"{out_png.stem}_{c.provider}.png")
        c.sprite.save(p, 1)
        raw = p.with_suffix(".raw.png")
        c.raw.save(raw)
        written += [p, raw]
    sheet = out_png.with_name(f"{out_png.stem}_compare.png")
    compare_sheet(cands).save(sheet)
    return written + [sheet]
