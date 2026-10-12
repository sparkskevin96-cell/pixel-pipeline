"""Online providers against fake PixelLab / Retro Diffusion servers (no keys, no network)."""
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

import pixel_pipeline.online as online
from test_generate import comfy  # noqa: F401  (pytest fixture)
from pixel_pipeline import settings_for
from pixel_pipeline.cli import main
from pixel_pipeline.online import (OnlineError, compare_sheet, load_keys, pixellab_generate,
                                   prompt_candidates, rd_request_size, retrodiffusion_generate,
                                   save_keys, to_sprite, write_candidates)

RED, SKIN, DARK = (200, 40, 40), (240, 190, 150), (30, 20, 30)


def pixel_figure(w=48, h=64):
    """Pixel art as a service would draw it: outlined figure on transparent."""
    img = np.zeros((h, w, 4), np.uint8)
    img[4:h - 4, w // 4: 3 * w // 4] = (*DARK, 255)
    img[5:h // 3, w // 4 + 1: 3 * w // 4 - 1] = (*SKIN, 255)
    img[h // 3:h - 5, w // 4 + 1: 3 * w // 4 - 1] = (*RED, 255)
    img[h // 2, w // 4 + 2] = (255, 255, 255, 255)                    # a lone highlight pixel
    return img


def png_b64(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class Fake(BaseHTTPRequestHandler):
    log: list = []
    polls: dict = {}

    def log_message(self, *a):
        pass

    def _send(self, code, payload):
        data = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n)) if n else {}

    def do_POST(self):
        body = self._body()
        Fake.log.append((self.path, dict(self.headers), body))
        if self.path.startswith("/pl/"):
            if self.headers.get("Authorization") != "Bearer pl-key":
                return self._send(401, {"detail": "Invalid API token"})
            w, h = body["image_size"]["width"], body["image_size"]["height"]
            if self.path == "/pl/create-image-pixen":
                image = {"type": "base64", "base64": png_b64(pixel_figure(w, h)), "format": "png"}
                return self._send(200, {"images": [image], "usage": {"type": "usd", "usd": 0.008}})
            if self.path == "/pl/generate-image-v2":
                Fake.polls["job-1"] = (0, w, h)
                return self._send(202, {"background_job_id": "job-1", "status": "processing"})
        if self.path == "/rd/inferences":
            if self.headers.get("X-RD-Token") == "rd-broke":
                return self._send(402, {"detail": "Insufficient balance"})
            if body.get("check_cost"):
                return self._send(200, {"balance_cost": 0.03, "model": "check_cost"})
            Fake.polls["t-1"] = (0, body["width"], body["height"])
            return self._send(200, {"status": "accepted", "task_id": "t-1"})
        self._send(404, {"detail": "no such path"})

    def do_GET(self):
        Fake.log.append((self.path, dict(self.headers), None))
        if self.path == "/pl/background-jobs/job-1":
            n, w, h = Fake.polls["job-1"]
            Fake.polls["job-1"] = (n + 1, w, h)
            if n == 0:
                return self._send(200, {"status": "processing"})
            raw = base64.b64encode(pixel_figure(w, h).tobytes()).decode()      # raw RGBA, not PNG
            return self._send(200, {"status": "completed", "last_response": {"images": [{"base64": raw}]}})
        if self.path == "/pl/balance":
            return self._send(200, {"type": "usd", "usd": 4.2})
        if self.path == "/rd/inferences/tasks/t-1":
            n, w, h = Fake.polls["t-1"]
            Fake.polls["t-1"] = (n + 1, w, h)
            if n == 0:
                return self._send(200, {"status": "running"})
            art = pixel_figure(w // 2, h // 2) if w >= 96 else pixel_figure(w, h)
            big = np.repeat(np.repeat(art, 2, 0), 2, 1) if w >= 96 else art
            return self._send(200, {"status": "succeeded", "result": {
                "base64_images": [png_b64(big)], "balance_cost": 0.03, "remaining_balance": 9.97}})
        if self.path == "/rd/inferences/credits":
            return self._send(200, {"credits": 0, "balance": 9.97})
        self._send(404, {"detail": "no such path"})


@pytest.fixture
def services(monkeypatch, tmp_path):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("PIXELLAB_BASE_URL", base + "/pl")
    monkeypatch.setenv("RETRO_DIFFUSION_BASE_URL", base + "/rd")
    monkeypatch.setattr(online, "KEYS_FILE", tmp_path / "keys.json")
    monkeypatch.setattr(online, "POLL_SECONDS", 0.01)
    for var in ("PIXELLAB_API_KEY", "RETRO_DIFFUSION_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    Fake.log.clear()
    yield {"pixellab": "pl-key", "retrodiffusion": "rd-key"}
    server.shutdown()


def test_pixellab_sync_and_queued(services):
    images, cost = pixellab_generate("a knight", 48, 64, keys=services, view="low top-down", seed=7)
    assert images[0].size == (48, 64) and cost == "$0.008"
    path, headers, body = Fake.log[0]
    assert path == "/pl/create-image-pixen" and headers["Authorization"] == "Bearer pl-key"
    assert body == {"description": "a knight", "image_size": {"width": 48, "height": 64},
                    "no_background": True, "seed": 7, "view": "low top-down"}
    images, _ = pixellab_generate("a knight", 32, 32, keys=services, model="pro")
    assert images[0].size == (32, 32) and np.asarray(images[0])[16, 16, :3].tolist() == list(RED)
    with pytest.raises(OnlineError, match="401.*check the API key"):
        pixellab_generate("a knight", 32, 32, keys={"pixellab": "wrong"})


def test_retrodiffusion_polls_and_reports_cost(services):
    images, cost = retrodiffusion_generate("a knight", 48, 64, keys=services, style="rd_pro__default")
    assert images[0].size == (48, 64) and cost == "0.03 credits (9.97 left)"
    path, headers, body = Fake.log[0]
    assert headers["X-Rd-Token"] == "rd-key" and headers.get("Idempotency-Key")
    assert body == {"prompt": "a knight", "prompt_style": "rd_pro__default", "width": 48, "height": 64,
                    "num_images": 1, "remove_bg": True}
    assert rd_request_size("rd_fast__default", 48, 64) == (96, 128)       # fast draws >= 64 px
    with pytest.raises(OnlineError, match="402.*out of credits"):
        retrodiffusion_generate("x", 64, 64, keys={"retrodiffusion": "rd-broke"})
    assert retrodiffusion_generate("x", 64, 64, keys=services, check_cost=True) == ([], "0.03")


def test_to_sprite_keeps_service_pixels():
    s = settings_for("person", width=48, height=64, colors=16)
    art = pixel_figure()
    sprite = to_sprite(Image.fromarray(art), s)
    assert sprite.rgba.shape == (64, 48, 4)
    assert ((sprite.rgba[..., 3] > 0) == (art[..., 3] > 0)).all()               # no resampling
    assert np.abs(sprite.rgba[..., :3].astype(int) - art[..., :3])[art[..., 3] > 0].max() <= 12
    assert tuple(sprite.rgba[32, 14, :3]) == (255, 255, 255)                     # highlight kept
    doubled = np.repeat(np.repeat(art, 2, 0), 2, 1)
    assert to_sprite(doubled, s).rgba.shape == (64, 48, 4)                       # 2x undone


def test_candidates_compare_and_files(services, tmp_path):
    s = settings_for("person", width=48, height=64)
    cands = prompt_candidates("a knight", s, ["pixellab", "retrodiffusion"], keys=services)
    assert [c.error for c in cands] == ["", ""]
    assert all(c.sprite.rgba.shape == (64, 48, 4) for c in cands)
    rd = next(c for c in cands if c.provider == "retrodiffusion")
    assert rd.raw.size == (96, 128)                                              # made 2x, shrunk back
    broken = prompt_candidates("a knight", s, ["pixellab", "retrodiffusion"],
                               keys={"retrodiffusion": "rd-key"})
    assert "No PixelLab API key" in broken[0].error and broken[1].sprite is not None
    sheet = compare_sheet(broken)
    assert sheet.width > 300 and sheet.height > 2 * 64
    names = [p.name for p in write_candidates(cands, tmp_path / "knight.png")]
    assert names == ["knight_pixellab.png", "knight_pixellab.raw.png", "knight_retrodiffusion.png",
                     "knight_retrodiffusion.raw.png", "knight_compare.png"]


def test_compare_sheet_rows_fit_error_text():
    """The app shows the sheet at scale 1 (64 px rows); long errors must not run into the next row."""
    s = settings_for("person", width=48, height=64)
    cands = prompt_candidates("a knight", s, ["pixellab", "retrodiffusion"], keys={})
    assert all(c.error for c in cands)
    sheet = np.asarray(compare_sheet(cands, scale=1).convert("RGB"))
    text = (np.abs(sheet[:, :230].astype(int) - (34, 38, 52)).sum(-1) > 0).any(1)
    edges = np.flatnonzero(np.diff(text.astype(int)))
    blocks = [(a, b) for a, b in zip(edges[::2], edges[1::2])]
    gaps = [b2 - a1 for (_, a1), (b2, _) in zip(blocks, blocks[1:])]
    assert max(gaps) >= 12                                   # a clear gap between the two rows


def test_gpu_provider_uses_comfyui(tmp_path, comfy):
    fake, client = comfy
    s = settings_for("person", width=24, height=32)
    (cand,) = prompt_candidates("a knight", s, ["gpu"], keys={}, comfy_client=client, seed=3)
    assert cand.error == "" and cand.sprite.rgba.shape == (32, 24, 4) and cand.seed == 3


def test_keys_file_and_env(tmp_path, monkeypatch):
    monkeypatch.delenv("PIXELLAB_API_KEY", raising=False)
    monkeypatch.delenv("RETRO_DIFFUSION_API_KEY", raising=False)
    path = save_keys({"pixellab": " a ", "retrodiffusion": ""}, tmp_path / "k.json")
    assert load_keys(path) == {"pixellab": "a"}
    monkeypatch.setenv("RETRO_DIFFUSION_API_KEY", "env-key")
    assert load_keys(path) == {"pixellab": "a", "retrodiffusion": "env-key"}


def test_cli_ai(services, tmp_path, capsys):
    assert main(["ai", "--save-key", "pixellab=pl-key", "--save-key", "retrodiffusion=rd-key"]) == 0
    assert main(["ai", "--balance", "--providers", "pixellab,retrodiffusion"]) == 0
    out = capsys.readouterr().out
    assert "PixelLab: $4.20" in out and "Retro Diffusion: 9.97 credits" in out
    assert main(["ai", "a knight", "-o", str(tmp_path / "k.png"), "--providers", "pixellab,retrodiffusion",
                 "-W", "32", "-H", "32", "--rd-style", "rd_pro__default"]) == 0
    assert (tmp_path / "k_compare.png").exists() and (tmp_path / "k_retrodiffusion.png").exists()


def test_prompt_tab(services, tmp_path):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.app import MainWindow
    save_keys(services)
    win = MainWindow()
    tab = win.prompt_tab
    tab.sync = True
    tab.prompt.setText("a knight")
    tab.use_gpu.setChecked(False)
    tab.make()
    assert [c.error for c in tab.cands] == ["", ""] and tab.pick.count() == 2
    assert {p.name for p in tab.save(tmp_path / "knight.png")} >= {"knight_compare.png"}
    tab.pick.setCurrentIndex(1)
    tab.send()
    assert win.tabs.currentIndex() == 0 and win.source is not None and win.sprite is not None


def test_prompt_tab_defaults_to_the_gpu(services):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6")
    from conftest import qt_app
    qt_app()
    from pixel_pipeline.ui.tools import PromptTab
    tab = PromptTab(None)                               # no keys saved: only the free GPU
    assert tab.use_gpu.isChecked() and not tab.use_pl.isChecked() and not tab.use_rd.isChecked()
    save_keys({"pixellab": "pl-key"})
    tab = PromptTab(None)
    assert tab.use_gpu.isChecked() and tab.use_pl.isChecked() and not tab.use_rd.isChecked()


def test_mcp_online_sprites(services, tmp_path):
    import asyncio
    pytest.importorskip("mcp")
    from pixel_pipeline.mcp_server import server
    save_keys(services)
    result = asyncio.run(server.call_tool("online_sprites", {
        "prompt": "a knight", "out_png": str(tmp_path / "k.png"), "width": 32, "height": 32,
        "rd_style": "rd_pro__default"}))
    content = result[0] if isinstance(result, tuple) else getattr(result, "content", result)
    assert content[0].text.startswith("pixellab:") and getattr(content[1], "type", "") == "image"
