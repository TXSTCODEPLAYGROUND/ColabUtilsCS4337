"""Load fastai and PyTorch models from disk into a plain ``nn.Module``."""

from __future__ import annotations

import os
import pickle
import warnings
import zipfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .._compat import require

ArchLike = Any  # nn.Module instance, fastai Learner, or a zero-argument callable returning one


def detect_format(path: str | os.PathLike) -> str:
    """Guess the file format: ``fastai_export``, ``torchscript``, ``torch_export``, or ``torch``."""
    path = Path(path)
    if path.suffix == ".pt2":
        return "torch_export"
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
        if any("/code/" in n or n.endswith("constants.pkl") for n in names):
            return "torchscript"
        if any(n.endswith("serialized_exported_program.json") or "/models/" in n for n in names):
            return "torch_export"
    if path.suffix == ".pkl":
        return "fastai_export"
    return "torch"


def _torch_load(path: Path, device: str, weights_only: bool | None):
    torch = require("torch")
    if weights_only is not None:
        return torch.load(path, map_location=device, weights_only=weights_only)
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except pickle.UnpicklingError:
        warnings.warn(
            f"{path.name} contains pickled Python objects (a full model, not just weights). "
            "Loading it with weights_only=False; only do this for files you trust.",
            stacklevel=3,
        )
        return torch.load(path, map_location=device, weights_only=False)


def extract_state_dict(obj: Mapping) -> dict:
    """Pull the weights out of common checkpoint layouts.

    Handles fastai ``learn.save`` (``{"model": ..., "opt": ...}``), Lightning-style
    ``{"state_dict": ...}``, ``{"model_state_dict": ...}``, and bare state dicts. A
    ``module.`` prefix from ``nn.DataParallel`` is removed.
    """
    for key in ("model", "state_dict", "model_state_dict"):
        inner = obj.get(key)
        if isinstance(inner, Mapping):
            obj = inner
            break
    state = dict(obj)
    if state and all(k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _build_arch(arch: ArchLike):
    torch = require("torch")
    if isinstance(arch, torch.nn.Module):
        return arch
    if isinstance(getattr(arch, "model", None), torch.nn.Module):
        return arch.model
    if callable(arch):
        built = arch()
        return _build_arch(built)
    raise TypeError("arch must be an nn.Module, a fastai Learner, or a callable returning one")


def load_model(
    path: str | os.PathLike,
    arch: ArchLike | Callable[[], Any] | None = None,
    *,
    device: str | None = None,
    weights_only: bool | None = None,
    strict: bool = True,
    learner: bool = False,
    eval: bool = True,
):
    """Load a model file and return a ``torch.nn.Module`` (in eval mode by default).

    Supported files:

    * fastai ``learn.export()`` ``.pkl`` (returns ``learn.model``, or the Learner with
      ``learner=True``; needs the same custom functions/classes defined as when exported)
    * fastai ``learn.save()`` ``.pth`` and PyTorch state dicts (need ``arch``)
    * full pickled PyTorch models (``torch.save(model, ...)``)
    * TorchScript (``torch.jit.save``) and ``torch.export`` ``.pt2`` archives

    ``arch`` is a model instance (for example ``torchvision.models.resnet18(num_classes=37)``),
    a fastai Learner, or a zero-argument callable that builds one. ``device`` defaults
    to CUDA when available.
    """
    torch = require("torch")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    fmt = detect_format(path)

    if fmt == "fastai_export":
        fastai_learner = require("fastai.learner", extra="fastai", purpose="fastai .pkl exports")
        learn = fastai_learner.load_learner(path, cpu=(device == "cpu"))
        if learner:
            return learn
        model = learn.model
    elif fmt == "torchscript":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = torch.jit.load(str(path), map_location=device)
    elif fmt == "torch_export":
        # Exported programs are frozen in inference mode and reject .eval()/.train().
        return torch.export.load(str(path)).module()
    else:
        obj = _torch_load(path, device, weights_only)
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(getattr(obj, "model", None), torch.nn.Module):
            if learner:
                return obj
            model = obj.model
        elif isinstance(obj, Mapping):
            if arch is None:
                raise ValueError(
                    f"{path.name} holds weights only (a state dict). Pass the architecture, "
                    "e.g. load_model(path, arch=torchvision.models.resnet18(num_classes=10))."
                )
            model = _build_arch(arch)
            model.load_state_dict(extract_state_dict(obj), strict=strict)
        else:
            raise TypeError(f"Don't know how to load a {type(obj).__name__} from {path.name}")

    model = model.to(device)
    if eval and hasattr(model, "eval"):
        model.eval()
    return model
