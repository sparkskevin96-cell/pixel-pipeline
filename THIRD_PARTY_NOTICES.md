# Third-party notices

## PixelOE

`src/pixel_pipeline/pixelize.py` reimplements (numpy/OpenCV, no PyTorch) the
contrast-aware outline expansion and contrast-based downscale from PixelOE.

- Project: https://github.com/KohakuBlueleaf/PixelOE
- Copyright: Shih-Ying Yeh (KohakuBlueLeaf)
- License: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0

Changes: ported from PyTorch chunk ops to OpenCV filters (median/min/max windows),
added wrap-around padding for tileable output, and alpha-aware colour matching.

## Optional dependencies

- rembg (MIT): background removal; models are downloaded by rembg on first use and
  carry their own licences.
- PySide6 (LGPLv3): desktop UI.
