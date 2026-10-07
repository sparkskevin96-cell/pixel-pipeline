"""End-to-end: launch the MCP server over stdio and drive it with the SDK client."""
import asyncio
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

mcp = pytest.importorskip("mcp")
from mcp import Client, StdioServerParameters, stdio_client

SRC = Path(__file__).resolve().parents[1] / "src"


def _cut_out_photo(path: Path) -> Path:
    """Shaded ellipse on a transparent background, so no background-removal model is needed."""
    h, w = 120, 150
    yy, xx = np.mgrid[0:h, 0:w]
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[..., 0] = np.clip(120 + xx * 0.8, 0, 255)
    rgba[..., 1] = np.clip(60 + yy * 0.6, 0, 255)
    rgba[..., 2] = 40
    rgba[..., 3] = np.where(((xx - w / 2) / 50) ** 2 + ((yy - h / 2) / 45) ** 2 <= 1, 255, 0)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def _clip(path: Path) -> Path:
    import cv2
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (96, 64))
    for i in range(12):
        f = np.full((64, 96, 3), 200, np.uint8)
        cv2.rectangle(f, (10 + i * 3, 20), (40 + i * 3, 50), (40, 60, 200), -1)
        vw.write(f)
    vw.release()
    return path


def _text(result) -> str:
    return "\n".join(b.text for b in result.content if b.type == "text")


def _images(result) -> list[Image.Image]:
    return [Image.open(io.BytesIO(base64.b64decode(b.data)))
            for b in result.content if b.type == "image"]


async def _session(tmp_path: Path) -> None:
    photo = _cut_out_photo(tmp_path / "blob.png")
    out = tmp_path / "out"
    params = StdioServerParameters(command=sys.executable, args=["-m", "pixel_pipeline.mcp_server"],
                                   cwd=str(tmp_path), env={"PYTHONPATH": str(SRC)})
    async with Client(stdio_client(params), read_timeout_seconds=60) as client:
        assert client.server_info.name == "pixel-pipeline"
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"list_palettes", "convert_photos", "make_sprite_sheet", "make_tileset",
                "preview_image"} <= names

        palettes = await client.call_tool("list_palettes", {})
        assert not palettes.is_error
        assert "pico-8" in _text(palettes) and "lospec:" in _text(palettes)

        result = await client.call_tool("convert_photos", {
            "paths": [str(photo)], "subject": "animal", "width": 20, "height": 16,
            "palette": "sweetie-16", "remove_background": False, "scale": 4,
            "out_dir": str(out), "aseprite": True})
        assert not result.is_error, _text(result)
        sprite, one_x = out / "blob.sprite.png", out / "blob.sprite@1x.png"
        assert Image.open(sprite).size == (80, 64)
        assert Image.open(one_x).size == (20, 16)
        assert (out / "blob.sprite.aseprite").exists()
        assert str(one_x) in _text(result) and "#1a1c2c" in _text(result)
        (preview,) = _images(result)
        assert max(preview.size) <= 1024 and preview.size[0] > 20

        sheet = await client.call_tool("make_sprite_sheet", {
            "frame_paths": [str(one_x), str(one_x)], "out_png": str(tmp_path / "sheet.png"),
            "tag": "idle"})
        assert not sheet.is_error, _text(sheet)
        assert (tmp_path / "sheet.json").exists() and _images(sheet)

        tiles = await client.call_tool("make_tileset", {
            "image_path": str(one_x), "out_png": str(tmp_path / "tiles.png"), "tile": 8})
        assert not tiles.is_error, _text(tiles)
        assert (tmp_path / "tiles.tmj").exists()

        look = await client.call_tool("preview_image", {"path": str(one_x)})
        assert not look.is_error and "20x16" in _text(look) and _images(look)

        missing = await client.call_tool("convert_photos", {"paths": [str(tmp_path / "nope.jpg")]})
        assert missing.is_error and "not found" in _text(missing)

        bad = await client.call_tool("convert_photos", {"paths": [str(photo)], "palette": "nope",
                                                        "remove_background": False})
        assert bad.is_error and "Unknown palette" in _text(bad)

        idle = await client.call_tool("idle_animation", {
            "sprite_path": str(sprite), "out_png": str(tmp_path / "idle.png"), "godot": True})
        assert not idle.is_error, _text(idle)
        assert Image.open(tmp_path / "idle.png").size == (80, 16)  # enlarged input shrunk to 1x
        assert (tmp_path / "idle.tres").exists() and (tmp_path / "idle.aseprite").exists()

        clip = _clip(tmp_path / "clip.mp4")
        walk = await client.call_tool("video_to_animation", {
            "video_path": str(clip), "out_png": str(tmp_path / "walk.png"), "frames": 4,
            "width": 20, "height": 12, "remove_background": False, "godot": True})
        assert not walk.is_error, _text(walk)
        assert Image.open(tmp_path / "walk.png").size == (80, 12)
        assert '&"walk"' in (tmp_path / "walk.tres").read_text() and _images(walk)

        depth = tmp_path / "depth.png"
        Image.fromarray(np.tile(np.linspace(0, 255, 96, dtype=np.uint8)[:, None], (1, 160))).save(depth)
        scene = tmp_path / "scene.png"
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (96, 160, 3), dtype=np.uint8)).save(scene)
        plx = await client.call_tool("parallax_layers", {
            "photo_path": str(scene), "out_dir": str(tmp_path / "plx"), "width": 64, "height": 36,
            "depth_path": str(depth)})
        assert not plx.is_error, _text(plx)
        assert (tmp_path / "plx" / "scene_parallax.tscn").exists() and _images(plx)


def test_mcp_server_over_stdio(tmp_path):
    asyncio.run(_session(tmp_path))
