"""pixel-pipeline: photos -> game-ready pixel-art sprites."""

__version__ = "0.1.0"

try:  # let Pillow open iPhone .heic/.heif photos
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

from .pipeline import Sprite, convert, convert_batch, load_image  # noqa: E402
from .settings import PRESETS, SUBJECTS, Settings, settings_for  # noqa: E402

__all__ = ["Settings", "Sprite", "PRESETS", "SUBJECTS", "convert", "convert_batch",
           "load_image", "settings_for", "__version__"]
