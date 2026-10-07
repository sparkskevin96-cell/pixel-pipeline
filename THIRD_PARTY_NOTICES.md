# Third-party notices

## PixelOE

`src/pixel_pipeline/pixelize.py` reimplements (numpy/OpenCV, no PyTorch) the
contrast-aware outline expansion and contrast-based downscale from PixelOE.

- Project: https://github.com/KohakuBlueleaf/PixelOE
- Copyright: Shih-Ying Yeh (KohakuBlueLeaf)
- License: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0

Changes: ported from PyTorch chunk ops to OpenCV filters (median/min/max windows),
added wrap-around padding for tileable output, and alpha-aware colour matching.

## Depth Anything V2 (Small)

`src/pixel_pipeline/depth.py` downloads and runs Depth Anything V2 **Small** (ViT-S) as
ONNX for parallax layers. Only the Small model is used; Base/Large are CC-BY-NC.

- Model: https://github.com/DepthAnything/Depth-Anything-V2, Apache License 2.0
- ONNX export: https://github.com/fabio-sim/Depth-Anything-ONNX, Apache License 2.0

## Optional dependencies

- onnxruntime (MIT): runs the background-removal and depth models.
- Background-removal models (U²-Net, IS-Net, BiRefNet) are downloaded on first use from
  rembg's GitHub releases (rembg is MIT; the models carry their own licences). The
  pre/post-processing in `cutout.py` mirrors rembg's.
- PySide6 (LGPLv3): desktop UI.
- mcp (MIT): MCP server.
