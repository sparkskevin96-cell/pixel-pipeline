"""Render every premium effect: a contact sheet (frames across, dark + light ground) and an
animated GIF (real timing, bloom) -> ~/Pictures/PixelPipeline/fx_preview.png / .gif.

    python tools/fx_preview.py [size] [theme]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixel_pipeline import spellfx  # noqa: E402

size = sys.argv[1] if len(sys.argv) > 1 else "medium"
theme = sys.argv[2] if len(sys.argv) > 2 else None
out = Path.home() / "Pictures" / "PixelPipeline"
out.mkdir(parents=True, exist_ok=True)
fxs = [spellfx.render_spell(k, size=size, theme=theme) for k in spellfx.SPELL_KINDS]
spellfx.contact_sheet(fxs, out / "fx_preview.png", scale=1, max_frames=32)
spellfx.preview_gif(fxs, out / "fx_preview.gif", scale=2, columns=5)
print(out / "fx_preview.png", out / "fx_preview.gif")
