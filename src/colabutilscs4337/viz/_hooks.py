"""Capture intermediate layer outputs with forward hooks."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from .._compat import require
from ..tools._summary import unwrap_model

LayerRef = Any  # dotted module name ("layer4.1.conv2") or the nn.Module itself


def resolve_layer(model: Any, layer: LayerRef) -> tuple[str, Any]:
    """Return ``(name, module)`` for a dotted name or a module inside ``model``."""
    module = unwrap_model(model)
    if isinstance(layer, str):
        return layer, module.get_submodule(layer)
    for name, sub in module.named_modules():
        if sub is layer:
            return name or type(sub).__name__, sub
    raise ValueError(f"{type(layer).__name__} is not a submodule of {type(module).__name__}")


def layer_names(model: Any, kind: type | tuple[type, ...] | None = None) -> list[str]:
    """Dotted names of submodules, optionally only those of a given type (e.g. ``nn.Conv2d``)."""
    module = unwrap_model(model)
    return [n for n, m in module.named_modules() if n and (kind is None or isinstance(m, kind))]


def activations(
    model: Any,
    x: Any,
    layers: Iterable[LayerRef] | None = None,
    *,
    detach: bool = True,
) -> dict[str, Any]:
    """Run ``model(x)`` and return ``{layer_name: output}`` for the requested layers.

    ``layers`` defaults to every ``nn.Conv2d``. Outputs are detached and moved to CPU
    unless ``detach=False``.
    """
    torch = require("torch")
    module = unwrap_model(model)
    if layers is None:
        layers = layer_names(module, torch.nn.Conv2d)
    resolved = [resolve_layer(module, layer) for layer in layers]

    captured: dict[str, Any] = {}

    def make_hook(name: str):
        def hook(_mod, _inp, out):
            captured[name] = out.detach().cpu() if detach and torch.is_tensor(out) else out

        return hook

    handles = [sub.register_forward_hook(make_hook(name)) for name, sub in resolved]
    try:
        with torch.set_grad_enabled(not detach), eval_mode(module):
            module(plain_tensor(x).to(_device_of(module)))
    finally:
        for handle in handles:
            handle.remove()
    return {name: captured[name] for name, _ in resolved if name in captured}


@contextmanager
def eval_mode(module: Any):
    """Temporarily switch to eval mode so BatchNorm/Dropout don't change or fail on tiny batches."""
    was_training = module.training
    module.eval()
    try:
        yield module
    finally:
        module.train(was_training)


def plain_tensor(x: Any):
    """Strip tensor subclasses (fastai ``TensorImage``) that break hooks in other libraries."""
    torch = require("torch")
    return x.as_subclass(torch.Tensor) if type(x) is not torch.Tensor else x


def _device_of(module: Any):
    for tensor in (*module.parameters(), *module.buffers()):
        return tensor.device
    return "cpu"
