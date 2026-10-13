from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .._compat import require

DEFAULT_INPUT_SIZE = (1, 3, 224, 224)
DEFAULT_COLUMNS = ("input_size", "output_size", "num_params", "trainable")


def unwrap_model(model: Any):
    """Return the ``torch.nn.Module`` inside a fastai ``Learner`` (or the module itself)."""
    torch = require("torch")
    inner = getattr(model, "model", model)
    if not isinstance(inner, torch.nn.Module):
        raise TypeError(f"Expected a torch.nn.Module or fastai Learner, got {type(model).__name__}")
    return inner


def _model_device(module) -> Any:
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return "cpu"


def summary(
    model: Any,
    input_size: Sequence[int] | None = DEFAULT_INPUT_SIZE,
    *,
    depth: int = 3,
    col_names: Sequence[str] = DEFAULT_COLUMNS,
    verbose: int = 1,
    **kwargs: Any,
):
    """``torchinfo.summary`` with computer-vision defaults; accepts fastai Learners too.

    ``input_size`` is ``(batch, channels, height, width)``; pass ``None`` (with
    ``input_data=...``) to summarize from real inputs instead.
    """
    torchinfo = require("torchinfo", extra="tools", purpose="model summaries")
    module = unwrap_model(model)
    kwargs.setdefault("device", _model_device(module))
    if input_size is not None and "input_data" not in kwargs:
        kwargs["input_size"] = tuple(input_size)
    return torchinfo.summary(
        module, depth=depth, col_names=tuple(col_names), verbose=verbose, **kwargs
    )
