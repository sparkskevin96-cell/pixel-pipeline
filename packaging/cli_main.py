"""PyInstaller entry point for the command line (pixel-pipeline.exe)."""
import sys

from pixel_pipeline.cli import main

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()   # worker processes (parallel.py) start this exe too
    sys.exit(main())
