"""Input/output shapes of any layer or block, captured with forward hooks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .._compat import require
from ._summary import DEFAULT_INPUT_SIZE, unwrap_model


@dataclass
class LayerShapes:
    """Shapes seen by one layer during a forward pass.

    ``inputs`` has one entry per positional argument (a shape tuple, a nested list
    for lists of tensors, or ``None`` for non-tensors); ``kwargs`` holds tensors
    passed by keyword. ``outputs`` has one entry per returned value.
    """

    name: str
    type: str
    inputs: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)

    @property
    def input(self):
        """Shape of the first input (the usual case)."""
        return self.inputs[0] if self.inputs else None

    @property
    def output(self):
        """Shape of the first output (the usual case)."""
        return self.outputs[0] if self.outputs else None

    def __str__(self) -> str:
        ins = ", ".join(str(s) for s in self.inputs)
        ins += "".join(f", {k}={s}" for k, s in self.kwargs.items())
        outs = ", ".join(str(s) for s in self.outputs)
        return f"{self.name} ({self.type}): {ins} -> {outs}"


def _shape(obj: Any):
    if hasattr(obj, "shape") and hasattr(obj, "dtype"):
        return tuple(obj.shape)
    if isinstance(obj, (list, tuple)):
        return [_shape(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _shape(v) for k, v in obj.items()}
    return None


def layer_shapes(
    model: Any,
    layers: Any,
    input_size: Sequence[int] = DEFAULT_INPUT_SIZE,
    *,
    x: Any = None,
    verbose: bool = True,
):
    """Input and output shapes of one or more layers/blocks, from a real forward pass.

    Parameters
    ----------
    model
        An ``nn.Module`` or a fastai ``Learner``.
    layers
        A submodule (``backbone.features[2][0]``), a dotted name (``"features.2.0"``),
        or a list of either.
    input_size
        Shape of the dummy input, ``(batch, channels, height, width)``. Ignored if
        ``x`` is given.
    x
        A real input batch to use instead of zeros.
    verbose
        Print one line per layer.

    Returns
    -------
    LayerShapes
        For a single layer; a ``{name: LayerShapes}`` dict for a list.

    The model runs once in eval mode without gradients, so BatchNorm statistics are
    not changed. If a layer runs several times in one pass, its first call is shown.

    Examples
    --------
    >>> backbone = torchvision.models.efficientnet_b0()
    >>> s = cu.tools.layer_shapes(backbone, backbone.features[2][0], (1, 3, 160, 160))
    features.2.0 (MBConv): (1, 16, 80, 80) -> (1, 24, 40, 40)
    >>> s.input, s.output            # first input / output
    >>> s.inputs                     # every positional input
    >>> cu.tools.layer_shapes(model, ["layer1", "layer2.0"])
    """
    torch = require("torch")
    from ..viz._hooks import _device_of, eval_mode, plain_tensor, resolve_layer

    module = unwrap_model(model)
    single = not isinstance(layers, (list, tuple))
    resolved = [resolve_layer(module, layer) for layer in ([layers] if single else layers)]
    found: dict[str, LayerShapes] = {}
    handles = []

    def make_hooks(name: str, sub: Any):
        def pre(_mod, args, kwargs):
            if name not in found:
                found[name] = LayerShapes(
                    name,
                    type(sub).__name__,
                    inputs=[_shape(a) for a in args],
                    kwargs={k: _shape(v) for k, v in kwargs.items()},
                )

        def post(_mod, _args, out):
            info = found.get(name)
            if info is not None and not info.outputs:
                many = isinstance(out, (list, tuple))
                info.outputs = list(_shape(out)) if many else [_shape(out)]

        return pre, post

    for name, sub in resolved:
        pre, post = make_hooks(name, sub)
        handles.append(sub.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(sub.register_forward_hook(post))
    try:
        device = _device_of(module)
        if x is None:
            inp = torch.zeros(*input_size, device=device)
        else:
            inp = plain_tensor(x).to(device)
        with torch.no_grad(), eval_mode(module):
            module(inp)
    finally:
        for handle in handles:
            handle.remove()

    missing = [name for name, _ in resolved if name not in found]
    if missing:
        raise ValueError(f"These layers were not called during the forward pass: {missing}")
    results = {name: found[name] for name, _ in resolved}
    if verbose:
        for info in results.values():
            print(info)
    return next(iter(results.values())) if single else results
