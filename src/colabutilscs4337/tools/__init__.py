"""Model inspection tools: torchinfo summaries and the Netron graph viewer."""

from ._netron import export_for_netron, netron, stop_netron
from ._summary import summary, unwrap_model


def install(*extras: str, **kwargs):
    """Reinstall the tools (torchinfo, netron, onnx, onnxscript) if they are missing.

    They come with ``pip install colabutilscs4337``; this is only needed if you
    uninstalled them or installed the package with ``--no-deps``.
    """
    from .._install import install as _install

    return _install(*(extras or ("tools",)), **kwargs)


__all__ = ["export_for_netron", "install", "netron", "stop_netron", "summary", "unwrap_model"]
