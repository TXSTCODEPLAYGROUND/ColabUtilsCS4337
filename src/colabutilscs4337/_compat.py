"""Lazy imports with actionable error messages."""

from __future__ import annotations

import importlib
from types import ModuleType

PACKAGE = "colabutilscs4337"

# Distributions Colab preinstalls; never declared as dependencies of this package.
COLAB_PROVIDED = (
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

# import name -> (pip distribution, extra that provides it)
_KNOWN = {
    "torchinfo": ("torchinfo", "tools"),
    "netron": ("netron", "tools"),
    "pytorch_grad_cam": ("grad-cam", "cam"),
    "fastai": ("fastai", "fastai"),
    "torch": ("torch", None),
    "torchvision": ("torchvision", None),
    "numpy": ("numpy", None),
    "matplotlib": ("matplotlib", None),
    "PIL": ("pillow", None),
}


class MissingDependencyError(ImportError):
    pass


def install_hint(module: str, extra: str | None = None) -> str:
    dist, known_extra = _KNOWN.get(module, (module, None))
    extra = extra or known_extra
    if extra:
        return (
            f"Run `import {PACKAGE} as cu; cu.install({extra!r})` "
            f'(or `%pip install "{PACKAGE}[{extra}]"`), then restart the session if asked.'
        )
    if dist in COLAB_PROVIDED:
        return (
            f"`{dist}` is normally preinstalled in Google Colab. Use a standard Colab runtime, "
            f"or install it with `%pip install {dist}`."
        )
    return f"Install it with `%pip install {dist}`."


def require(module: str, extra: str | None = None, purpose: str | None = None) -> ModuleType:
    """Import ``module`` or raise :class:`MissingDependencyError` explaining how to install it."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        what = f" (needed for {purpose})" if purpose else ""
        raise MissingDependencyError(
            f"`{module}` is not installed{what}. {install_hint(module, extra)}"
        ) from exc


def has_module(module: str) -> bool:
    try:
        importlib.import_module(module)
    except ImportError:
        return False
    return True
