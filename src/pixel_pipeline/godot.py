"""Godot 4 resources: SpriteFrames (.tres) for AnimatedSprite2D, TileSet (.tres) for TileMapLayer.

Texture paths are written relative to the .tres, so keep the PNG next to it
anywhere inside your Godot project. For crisp pixels set Project Settings ->
Rendering -> Textures -> Default Texture Filter = Nearest.
"""
from __future__ import annotations

from pathlib import Path


def _write(path, text: str) -> Path:
    path = Path(path)
    path.write_text(text, encoding="utf-8")
    return path


def write_sprite_frames(out_tres, sheet_png, rects: list[tuple[int, int, int, int]],
                        animations: dict[str, dict] | None = None, name: str = "default",
                        duration_ms: int = 100, loop: bool = True) -> Path:
    """rects: (x, y, w, h) of every frame in the sheet.
    animations: {name: {"frames": [frame indices], "duration_ms": int, "loop": bool}};
    default is one animation `name` using every frame."""
    if animations is None:
        animations = {name: {"frames": list(range(len(rects))), "duration_ms": duration_ms,
                             "loop": loop}}
    lines = [f'[gd_resource type="SpriteFrames" load_steps={len(rects) + 2} format=3]', "",
             f'[ext_resource type="Texture2D" path="{Path(sheet_png).name}" id="1_sheet"]', ""]
    for i, (x, y, w, h) in enumerate(rects):
        lines += [f'[sub_resource type="AtlasTexture" id="AtlasTexture_{i}"]',
                  'atlas = ExtResource("1_sheet")',
                  f"region = Rect2({x}, {y}, {w}, {h})", ""]
    anims = []
    for anim_name, a in animations.items():
        frames = ",\n".join(
            '{\n"duration": 1.0,\n' f'"texture": SubResource("AtlasTexture_{i}")\n' "}"
            for i in a["frames"])
        speed = 1000.0 / max(1, a.get("duration_ms", duration_ms))
        anims.append("{\n" f'"frames": [{frames}],\n'
                     f'"loop": {"true" if a.get("loop", True) else "false"},\n'
                     f'"name": &"{anim_name}",\n'
                     f'"speed": {speed:.4g}\n' "}")
    lines += ["[resource]", f"animations = [{', '.join(anims)}]", ""]
    return _write(out_tres, "\n".join(lines))


def write_tileset(out_tres, tileset_png, tile: int, columns: int, count: int) -> Path:
    """TileSet with one atlas source containing `count` tiles laid out `columns` wide."""
    tiles = "\n".join(f"{i % columns}:{i // columns}/0 = 0" for i in range(count))
    text = f"""[gd_resource type="TileSet" load_steps=3 format=3]

[ext_resource type="Texture2D" path="{Path(tileset_png).name}" id="1_tiles"]

[sub_resource type="TileSetAtlasSource" id="TileSetAtlasSource_1"]
texture = ExtResource("1_tiles")
texture_region_size = Vector2i({tile}, {tile})
{tiles}

[resource]
tile_size = Vector2i({tile}, {tile})
sources/0 = SubResource("TileSetAtlasSource_1")
"""
    return _write(out_tres, text)
