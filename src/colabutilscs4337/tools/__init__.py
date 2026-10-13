"""Model inspection tools: torchinfo summaries and the Netron graph viewer."""

from ._netron import export_for_netron, netron, stop_netron
from ._summary import summary, unwrap_model


def install(*extras: str, **kwargs):
    """Install the optional tools (torchinfo, netron) without touching Colab's packages."""
    from .._install import install as _install

    return _install(*(extras or ("tools",)), **kwargs)


__all__ = ["export_for_netron", "install", "netron", "stop_netron", "summary", "unwrap_model"]
