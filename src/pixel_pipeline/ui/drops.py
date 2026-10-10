"""Pictures dropped on the window straight from a web browser.

A browser drag carries the picture's web address (and often an <img> tag); some also carry the
picture itself. The app downloads it (nothing is uploaded) and saves it in Pictures/Pixel
Pipeline, so every tool gets an ordinary file, the same as a picture dropped from a folder.
No Qt here (app.py reads the drop and calls these).
"""
from __future__ import annotations

import base64
import html
import io
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

MAX_BYTES = 60 << 20
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PixelPipeline"   # some sites refuse Python's
_IMG = re.compile(r"""<img\b[^>]*?\bsrc\s*=\s*(["'])(.*?)\1""", re.IGNORECASE | re.DOTALL)
_EXT = {"PNG": ".png", "JPEG": ".jpg", "GIF": ".gif", "WEBP": ".webp", "BMP": ".bmp"}


class DropError(Exception):
    pass


def web_sources(urls: list[str], html_text: str = "", text: str = "") -> list[str]:
    """Where the dropped picture can be fetched from, best first: the <img> the browser dragged,
    then the dragged addresses, then plain text that is an address."""
    links = [u for u in urls if u.lower().startswith(("http://", "https://", "data:image/"))]
    base = next((u for u in links if u.lower().startswith("http")), "")
    found = [urllib.parse.urljoin(base, html.unescape(m.group(2).strip()))
             for m in _IMG.finditer(html_text or "")]
    plain = (text or "").strip()
    if plain.lower().startswith(("http://", "https://", "data:image/")) and not any(c.isspace() for c in plain):
        links.append(plain)
    out = []
    for s in found + links:
        if s.lower().startswith(("http://", "https://", "data:image/")) and s not in out:
            out.append(s)
    return out


def _name_from(source: str) -> str:
    if source.lower().startswith("data:"):
        return "web-picture"
    stem = Path(urllib.parse.unquote(urllib.parse.urlparse(source).path)).stem
    stem = re.sub(r"[^\w\-]+", "-", stem).strip("-")[:60]
    return stem or "web-picture"


def save_picture(data: bytes, dest: Path, name: str = "web-picture") -> Path:
    """Check the bytes are a picture, then save them under a name not taken yet."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt = im.format
            im.load()
    except Exception as e:  # noqa: BLE001 - PIL raises many kinds
        raise DropError("That isn't a picture the app can open (PNG, JPEG, GIF, WebP or BMP).") from e
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    ext = _EXT.get(fmt or "", ".png")
    path = dest / f"{name}{ext}"
    n = 2
    while path.exists():
        path = dest / f"{name}-{n}{ext}"
        n += 1
    if ext == ".png" and fmt != "PNG":              # unknown format: store what PIL decoded
        Image.open(io.BytesIO(data)).save(path)
    else:
        path.write_bytes(data)
    return path


def download(source: str, timeout: float = 20) -> bytes:
    if source.lower().startswith("data:"):
        head, _, body = source.partition(",")
        try:
            return base64.b64decode(body) if ";base64" in head.lower() else urllib.parse.unquote_to_bytes(body)
        except ValueError as e:
            raise DropError("The dropped picture's data is damaged.") from e
    req = urllib.request.Request(source, headers={"User-Agent": USER_AGENT,
                                                  "Accept": "image/avif,image/webp,image/*,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as e:
        raise DropError(f"The website wouldn't give the picture ({e.code} {e.reason}). Save it "
                        "from the browser (right-click > Save image as) and drop the file.") from e
    except (urllib.error.URLError, OSError) as e:
        raise DropError(f"Couldn't download the picture ({getattr(e, 'reason', e)}).") from e
    if len(data) > MAX_BYTES:
        raise DropError("That picture is too big to drop (over 60 MB).")
    return data


def fetch(sources: list[str], dest: Path) -> Path:
    """The first source that gives a picture, saved in dest."""
    if not sources:
        raise DropError("The drop had no picture in it.")
    error = None
    for source in sources:
        try:
            return save_picture(download(source), dest, _name_from(source))
        except DropError as e:
            error = error or e
    raise error
