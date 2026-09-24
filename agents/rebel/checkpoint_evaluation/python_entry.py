"""Run a project script after loading the active Python's NumPy/PyTorch."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy  # noqa: F401
import torch  # noqa: F401


script = Path(sys.argv[1]).resolve()
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
