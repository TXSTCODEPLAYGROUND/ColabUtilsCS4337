from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

from .. import _colab

TRACKED = (
    "torch",
    "torchvision",
    "fastai",
    "fastcore",
    "numpy",
    "matplotlib",
    "pillow",
    "ipywidgets",
    "timm",
    "opencv-python",
)
EXTRAS = ("torchinfo", "netron", "onnx", "onnxscript", "grad-cam")


def is_colab() -> bool:
    return _colab.is_colab()


def notebook_name(default: str = _colab.DEFAULT_NOTEBOOK_NAME) -> str:
    return _colab.notebook_name(default)


def installed_version(dist: str) -> str | None:
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def gpu_info() -> list[str]:
    """GPU names and memory from ``nvidia-smi`` (no torch import needed)."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def runtime_info() -> dict:
    """Snapshot of the runtime: Colab or not, Python, OS, GPU, and package versions."""
    from .. import __version__

    return {
        "colabutilscs4337": __version__,
        "is_colab": is_colab(),
        "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"),
        "python": platform.python_version(),
        "executable": sys.executable,
        "os": platform.platform(),
        "gpus": gpu_info(),
        "packages": {d: installed_version(d) for d in (*TRACKED, *EXTRAS)},
    }
