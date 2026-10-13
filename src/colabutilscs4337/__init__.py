"""Google Colab utilities for CS4337.

Subpackages are imported lazily so ``import colabutilscs4337`` stays fast and never
imports torch or fastai until a feature that needs them is used::

    import colabutilscs4337 as cu
    cu.install("tools", "cam")
    cu.env.check()
    run = cu.drive.new_run("baseline")
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - only when running from a source tree without a build
    __version__ = "0.0.0+unknown"

if TYPE_CHECKING:
    from . import drive, env, models, tools, viz  # noqa: F401
    from ._install import install  # noqa: F401

_SUBMODULES = {"env", "drive", "tools", "models", "viz"}

__all__ = ["__version__", "install", *sorted(_SUBMODULES)]


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name == "install":
        from ._install import install

        globals()["install"] = install
        return install
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
