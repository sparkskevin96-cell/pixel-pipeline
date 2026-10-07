"""Draws the launcher icon: a 16x16 pixel-art camera, upscaled with nearest neighbour.

Writes the legacy mipmaps, the adaptive-icon foreground/monochrome layers (Android 8+ /
13+ themed icons) and the in-app splash sprite. Re-run after changing the sprite:

    ../.venv/bin/python tool/make_icon.py        (from mobile/; needs Pillow)
"""

from pathlib import Path

from PIL import Image, ImageDraw

SPRITE = [
    "................",
    "................",
    "...KKKK.........",
    "..KYYYYK....KKK.",
    ".KKKKKKKKKKKRRRK",
    ".KWWWWWWWWWWWWWK",
    ".KWWWWWKKKKWWWWK",
    ".KWWWWKLLLLKWWWK",
    ".KWWWKLHBBBLKWWK",
    ".KWWWKLBBBBLKWWK",
    ".KGGGKLBBBDLKGGK",
    ".KGGGGKLDDLKGGGK",
    ".KGGGGGKKKKGGGGK",
    "..KKKKKKKKKKKKK.",
    "................",
    "................",
]
PALETTE = {
    "K": (26, 28, 44),     # outline (sweetie-16 black)
    "W": (244, 244, 244),  # body
    "G": (148, 176, 194),  # body shade
    "L": (51, 60, 87),     # lens ring
    "B": (65, 166, 246),   # glass
    "D": (59, 93, 201),    # glass shade (the web app's accent)
    "H": (255, 255, 255),  # glint
    "Y": (255, 205, 117),  # flash
    "R": (177, 62, 83),    # shutter
}
BACKGROUND = (59, 93, 201)  # #3b5dc9, the web app's accent

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "android" / "app" / "src" / "main" / "res"
DENSITIES = {"mdpi": 1, "hdpi": 1.5, "xhdpi": 2, "xxhdpi": 3, "xxxhdpi": 4}


def sprite(scale: int, color=None) -> Image.Image:
    im = Image.new("RGBA", (16, 16), (0, 0, 0, 0))
    for y, row in enumerate(SPRITE):
        for x, ch in enumerate(row):
            if ch != ".":
                im.putpixel((x, y), (*(color or PALETTE[ch]), 255))
    return im.resize((16 * scale, 16 * scale), Image.NEAREST)


def legacy(size: int) -> Image.Image:
    """Square icon on a 24-cell grid: rounded accent tile, sprite inset 4 cells."""
    cell = size // 24
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    # Rounded tile with stair-stepped (pixel) corners.
    for y in range(1, 23):
        inset = {1: 3, 2: 1, 3: 1, 20: 1, 21: 1, 22: 3}.get(y, 0)
        d.rectangle([(1 + inset) * cell, y * cell, (23 - inset) * cell - 1, (y + 1) * cell - 1], fill=BACKGROUND)
    im.alpha_composite(sprite(cell), (4 * cell, 4 * cell))
    return im


def adaptive(size: int, color=None) -> Image.Image:
    """108dp foreground; the sprite stays inside the 66dp safe circle."""
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    scale = int(size * 66 / 108 / 16 / 1.2)  # camera body is 15x12 of 16x16; keep its corners in the circle
    sp = sprite(max(scale, 1), color)
    im.alpha_composite(sp, ((size - sp.width) // 2, (size - sp.height) // 2 + scale // 2))
    return im


def main() -> None:
    for name, k in DENSITIES.items():
        folder = RES / f"mipmap-{name}"
        folder.mkdir(parents=True, exist_ok=True)
        legacy(int(48 * k)).save(folder / "ic_launcher.png")
        adaptive(int(108 * k)).save(folder / "ic_launcher_foreground.png")
        adaptive(int(108 * k), color=(255, 255, 255)).save(folder / "ic_launcher_monochrome.png")
    any_dpi = RES / "mipmap-anydpi-v26"
    any_dpi.mkdir(exist_ok=True)
    (any_dpi / "ic_launcher.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<adaptive-icon xmlns:android="http://schemas.android.com/apk/res/android">\n'
        '    <background android:drawable="@color/ic_launcher_background" />\n'
        '    <foreground android:drawable="@mipmap/ic_launcher_foreground" />\n'
        '    <monochrome android:drawable="@mipmap/ic_launcher_monochrome" />\n'
        "</adaptive-icon>\n"
    )
    # In-app splash sprite (Flutter asset, drawn with FilterQuality.none).
    (ROOT / "assets" / "icon").mkdir(parents=True, exist_ok=True)
    sprite(1).save(ROOT / "assets" / "icon" / "sprite.png")


if __name__ == "__main__":
    main()
