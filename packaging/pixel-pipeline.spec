# PyInstaller spec: one folder, two programs sharing the same libraries.
#
#   PixelPipeline(.exe)    desktop app, no console window
#   pixel-pipeline(.exe)   command line
#
# Build (from the repo root, with pip install -e ".[ui,cutout,3d,build]"):
#   pyinstaller packaging/pixel-pipeline.spec --noconfirm
# Output: dist/PixelPipeline/. Background-removal models are not bundled; they download
# on first use (see pixel_pipeline/cutout.py).
import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

HERE = Path(SPECPATH)
SRC = HERE.parent / "src"

# Our own modules, including ones only imported lazily. The MCP server needs the `mcp`
# extra and runs from a normal Python install, so it stays out.
HIDDEN = collect_submodules("pixel_pipeline", filter=lambda name: name != "pixel_pipeline.mcp_server")
# trimesh (3D models for turnarounds) loads its file formats lazily and reads its own resources.
HIDDEN += collect_submodules("trimesh.exchange") + collect_submodules("trimesh.visual")
DATAS = collect_data_files("trimesh")
# The window/taskbar icon the app loads at runtime.
DATAS += [(str(SRC / "pixel_pipeline" / "ui" / "icon.png"), "pixel_pipeline/ui")]
ICON = str(HERE / "icon.ico")

# Qt ships nearly everything; the app only uses QtCore / QtGui / QtWidgets (QtDBus stays:
# QtGui and QtWidgets import it).
QT_UNUSED = [
    "Qt3DAnimation", "Qt3DCore", "Qt3DExtras", "Qt3DInput", "Qt3DLogic", "Qt3DRender",
    "QtBluetooth", "QtCharts", "QtConcurrent", "QtDataVisualization", "QtDesigner",
    "QtGraphs", "QtGraphsWidgets", "QtHelp", "QtHttpServer", "QtLocation", "QtMultimedia",
    "QtMultimediaWidgets", "QtNetwork", "QtNetworkAuth", "QtNfc", "QtOpenGL", "QtOpenGLWidgets",
    "QtPdf", "QtPdfWidgets", "QtPositioning", "QtPrintSupport", "QtQml", "QtQuick", "QtQuick3D",
    "QtQuickControls2", "QtQuickTest", "QtQuickWidgets", "QtRemoteObjects", "QtScxml",
    "QtSensors", "QtSerialBus", "QtSerialPort", "QtSpatialAudio", "QtSql", "QtStateMachine",
    "QtSvg", "QtSvgWidgets", "QtTest", "QtTextToSpeech", "QtUiTools", "QtWebChannel",
    "QtWebEngineCore", "QtWebEngineQuick", "QtWebEngineWidgets", "QtWebSockets", "QtWebView",
    "QtXml",
]
EXCLUDES = [f"PySide6.{m}" for m in QT_UNUSED] + [
    # never needed at runtime (dev tools, or rembg's stack that cutout.py replaced)
    "tkinter", "matplotlib", "scipy", "skimage", "rembg", "pymatting", "numba", "llvmlite",
    "torch", "sympy", "pandas", "IPython", "pytest", "PyInstaller",
    "mcp", "pydantic", "pydantic_core", "starlette", "uvicorn", "httpx",
]


# Qt plugins the hooks collect that drag in unused Qt libraries (virtual keyboard ->
# Quick/Qml, PDF image plugin -> QtPdf, VNC/TUIO -> QtNetwork), those libraries, and the
# software OpenGL fallback (Windows, ~20 MB; a widgets app doesn't use OpenGL).
DROP_BINARIES = re.compile(
    r"qtvirtualkeyboardplugin|qpdf|qvnc|qtuiotouchplugin"
    r"|qt6(quick|qml|virtualkeyboard|pdf|network)|opengl32sw", re.IGNORECASE)


def trim(toc):
    """Drop unused Qt pieces and Qt's UI translations (the app is English only)."""
    return [entry for entry in toc
            if not DROP_BINARIES.search(Path(entry[0]).name)
            and "translations" not in Path(entry[0]).parts]


def analysis(script):
    a = Analysis(
        [str(HERE / script)],
        pathex=[str(SRC)],
        hiddenimports=HIDDEN,
        datas=DATAS,
        excludes=EXCLUDES,
        noarchive=False,
    )
    a.binaries = trim(a.binaries)
    a.datas = trim(a.datas)
    return a


gui = analysis("gui_main.py")
cli = analysis("cli_main.py")

gui_exe = EXE(
    PYZ(gui.pure),
    gui.scripts,
    exclude_binaries=True,
    name="PixelPipeline",
    console=False,
    icon=ICON,
    upx=False,
)
cli_exe = EXE(
    PYZ(cli.pure),
    cli.scripts,
    exclude_binaries=True,
    name="pixel-pipeline",
    console=True,
    icon=ICON,
    upx=False,
)
COLLECT(
    gui_exe, gui.binaries, gui.datas,
    cli_exe, cli.binaries, cli.datas,
    name="PixelPipeline",
    upx=False,
)
