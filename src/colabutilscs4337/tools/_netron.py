from __future__ import annotations

import os
import tempfile
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import _colab
from .._compat import require
from ._summary import DEFAULT_INPUT_SIZE, _model_device, unwrap_model

_servers: dict[str, tuple[str, int]] = {}
_SUFFIXES = {"torchscript": ".pt", "export": ".pt2", "onnx": ".onnx"}


def export_for_netron(
    model: Any,
    path: str | os.PathLike | None = None,
    input_size: Sequence[int] = DEFAULT_INPUT_SIZE,
    format: str = "torchscript",
) -> Path:
    """Save a model in a format Netron can draw as a graph.

    ``format="torchscript"`` (default) needs only torch. ``format="export"`` writes a
    ``torch.export`` ``.pt2`` archive. ``format="onnx"`` gives the cleanest graph but
    needs the ``onnx`` package, which Colab does not preinstall.
    """
    torch = require("torch")
    if format not in _SUFFIXES:
        raise ValueError(f"format must be one of {sorted(_SUFFIXES)}")
    module = unwrap_model(model)
    example = torch.zeros(*input_size, device=_model_device(module))
    suffix = _SUFFIXES[format]
    if path is None:
        path = Path(tempfile.mkdtemp(prefix="netron-")) / f"{type(module).__name__}{suffix}"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    was_training = module.training
    module.eval()
    try:
        with torch.no_grad(), warnings.catch_warnings():
            # torch>=2.10 marks torch.jit as deprecated; it still produces Netron-readable files.
            warnings.simplefilter("ignore", DeprecationWarning)
            if format == "onnx":
                require("onnx", purpose="ONNX export")
                torch.onnx.export(module, (example,), str(path))
            elif format == "export":
                torch.export.save(torch.export.export(module, (example,)), str(path))
            else:
                torch.jit.trace(module, example).save(str(path))
    finally:
        module.train(was_training)
    return path


def netron(
    model_or_path: Any,
    *,
    input_size: Sequence[int] = DEFAULT_INPUT_SIZE,
    format: str = "torchscript",
    port: int | None = None,
    height: int = 700,
) -> tuple[str, int]:
    """Open Netron for a model file or an in-memory model (``nn.Module`` / fastai Learner).

    In Colab the viewer is shown inside the cell output through Colab's port proxy.
    Returns Netron's ``(host, port)`` address; pass it to :func:`stop_netron` to stop.
    """
    netron_mod = require("netron", extra="tools", purpose="the Netron viewer")
    if isinstance(model_or_path, (str, os.PathLike)):
        path = Path(model_or_path)
        if not path.exists():
            raise FileNotFoundError(path)
    else:
        path = export_for_netron(model_or_path, input_size=input_size, format=format)

    key = str(path.resolve())
    if key in _servers:
        stop_netron(_servers.pop(key))
    address = ("localhost", port) if port else None
    address = netron_mod.start(str(path), address=address, browse=False)
    _servers[key] = address
    _colab.serve_port(address[1], height=height)
    return address


def stop_netron(address: tuple[str, int] | None = None) -> None:
    """Stop one Netron server, or all servers started by this module."""
    netron_mod = require("netron", extra="tools")
    if address is not None:
        netron_mod.stop(address)
        for key in [k for k, v in _servers.items() if tuple(v) == tuple(address)]:
            del _servers[key]
        return
    for addr in list(_servers.values()):
        netron_mod.stop(addr)
    _servers.clear()
