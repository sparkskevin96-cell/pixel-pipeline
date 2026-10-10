"""Pictures dropped straight from a web browser (ui/drops.py)."""
import base64
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from pixel_pipeline.ui import drops
from pixel_pipeline.ui.drops import DropError


def png_bytes(size=(12, 10), colour=(200, 40, 40, 255), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    im = Image.new("RGBA", size, colour)
    (im.convert("RGB") if fmt == "JPEG" else im).save(buf, fmt)
    return buf.getvalue()


class Site(BaseHTTPRequestHandler):
    pages = {}
    agents = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.agents.append(self.headers.get("User-Agent", ""))
        body = self.pages.get(self.path)
        if body is None:
            self.send_response(403)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def site():
    Site.pages = {"/art/knight%20sprite.png": png_bytes(), "/photo.jpg": png_bytes(fmt="JPEG"),
                  "/page.html": b"<html>not a picture</html>"}
    Site.agents = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), Site)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_sources_prefer_the_dragged_img_and_resolve_relative_ones():
    html = ('<html><body><!--StartFragment--><a href="/wiki/Knight">'
            '<img alt="x" src="/media/knight.png?w=200&amp;h=100"></a><!--EndFragment--></body></html>')
    got = drops.web_sources(["https://example.com/wiki/Knight", "file:///C:/x.png"], html, "hello there")
    assert got == ["https://example.com/media/knight.png?w=200&h=100", "https://example.com/wiki/Knight"]
    assert drops.web_sources([], "", " https://e.com/a.webp ") == ["https://e.com/a.webp"]
    assert drops.web_sources([], "", "javascript:alert(1)") == []


def test_fetch_downloads_names_and_never_overwrites(site, tmp_path):
    page, art = f"{site}/page.html", f"{site}/art/knight%20sprite.png"
    first = drops.fetch([page, art], tmp_path)          # the page isn't a picture: next source
    assert first.name == "knight-sprite.png"
    assert np.asarray(Image.open(first)).shape == (10, 12, 4)
    assert drops.fetch([art], tmp_path).name == "knight-sprite-2.png"
    assert drops.fetch([f"{site}/photo.jpg"], tmp_path).suffix == ".jpg"
    assert all("Mozilla" in a for a in Site.agents)      # sites that refuse Python still answer


def test_data_uris_and_plain_errors(site, tmp_path):
    uri = "data:image/png;base64," + base64.b64encode(png_bytes()).decode()
    assert drops.fetch([uri], tmp_path).name == "web-picture.png"
    with pytest.raises(DropError, match="wouldn't give the picture"):
        drops.fetch([f"{site}/secret.png"], tmp_path)
    with pytest.raises(DropError, match="isn't a picture"):
        drops.fetch([f"{site}/page.html"], tmp_path)
    with pytest.raises(DropError, match="no picture"):
        drops.fetch([], tmp_path)
    tiff = io.BytesIO()
    Image.new("RGB", (4, 4)).save(tiff, "TIFF")
    assert drops.save_picture(tiff.getvalue(), tmp_path, "scan").name == "scan.png"   # stored as PNG
