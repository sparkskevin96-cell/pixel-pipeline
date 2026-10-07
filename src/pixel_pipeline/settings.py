"""Conversion settings and per-subject presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields

SUBJECTS = ("person", "animal", "object", "background")
FITS = ("contain", "cover", "stretch")
ANCHORS = ("bottom", "center")


@dataclass
class Settings:
    # Output size in sprite pixels. height None or 0 keeps the photo's aspect ratio.
    width: int = 32
    height: int | None = None
    fit: str = "contain"            # contain (pad) | cover (crop) | stretch
    anchor: str = "bottom"          # where a contained subject sits: bottom (feet on floor) | center

    # Subject isolation
    remove_background: bool = True
    cutout_model: str = "isnet-general-use"
    crop_to_subject: bool = True
    margin: float = 0.04            # padding around the subject when cropping (fraction of size)

    # Pixelization
    patch: int = 8                  # photo pixels sampled per sprite pixel (before downscale)
    detail: int = 2                 # edge thickening (0 = off); higher keeps more linework
    downscale: str = "contrast"     # contrast | median | center | area
    brightness: float = 0.0         # -1..1
    contrast: float = 1.0
    saturation: float = 1.0
    tileable: bool = False          # make the result wrap seamlessly (backgrounds/tiles)

    # Colour
    palette: str = "auto"           # auto | built-in | lospec:<slug> | path to .gpl/.hex/.png
    colors: int = 16                # palette size when palette == "auto"
    dither: str = "none"            # none | floyd | bayer2 | bayer4 | bayer8
    dither_strength: float = 1.0

    # Sprite finishing
    outline: bool = True            # 1px outline around the silhouette
    outline_color: str = "auto"     # auto (darkest palette colour) | #rrggbb
    alpha_threshold: int = 128

    def resolve_size(self, src_w: int, src_h: int) -> tuple[int, int]:
        if self.height:
            return self.width, self.height
        return self.width, max(1, round(self.width * src_h / max(1, src_w)))

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict] = {
    "person": dict(width=32, height=48, anchor="bottom", detail=2, colors=16, outline=True),
    "animal": dict(width=32, height=32, anchor="bottom", detail=2, colors=16, outline=True),
    "object": dict(width=32, height=32, anchor="center", detail=1, colors=12, outline=True),
    "background": dict(width=160, height=90, fit="cover", anchor="center",
                       remove_background=False, crop_to_subject=False, detail=1,
                       colors=32, outline=False, dither="bayer4", dither_strength=0.6),
}


def settings_for(subject: str = "object", **overrides) -> Settings:
    """Preset defaults for a subject type; any non-None override wins."""
    if subject not in PRESETS:
        raise ValueError(f"Unknown subject '{subject}' (use {', '.join(SUBJECTS)})")
    names = {f.name for f in fields(Settings)}
    unknown = set(overrides) - names
    if unknown:
        raise TypeError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
    values = dict(PRESETS[subject])
    values.update({k: v for k, v in overrides.items() if v is not None})
    return Settings(**values)
