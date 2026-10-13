"""The only module that touches ``google.colab``.

Every Colab-specific call lives here so the rest of the package is plain Python and
tests can replace these functions with ``monkeypatch``. Each function has a sensible
fallback outside Colab (local Jupyter, VS Code, plain Python).
"""

from __future__ import annotations

import json
import os
import socket
import sys
import urllib.request
import warnings
from pathlib import Path

DRIVE_MOUNTPOINT = "/content/drive"
NOTEBOOK_NAME_ENV = "COLABUTILS_NOTEBOOK_NAME"
LOCAL_DRIVE_ENV = "COLABUTILS_LOCAL_DRIVE"
DEFAULT_NOTEBOOK_NAME = "untitled"


def is_colab() -> bool:
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab  # noqa: F401
    except ImportError:
        return False
    return True


def drive_root() -> Path:
    """Root of "My Drive": the mounted Drive in Colab, a local folder elsewhere."""
    if is_colab():
        return Path(DRIVE_MOUNTPOINT) / "MyDrive"
    return Path(os.environ.get(LOCAL_DRIVE_ENV, Path.cwd() / "drive")) / "MyDrive"


def is_drive_mounted() -> bool:
    return is_colab() and (Path(DRIVE_MOUNTPOINT) / "MyDrive").is_dir()


def mount(force_remount: bool = False, timeout_ms: int = 120_000) -> Path:
    """Mount Google Drive (no-op if already mounted) and return the "My Drive" path.

    Outside Colab nothing is mounted; a local ``./drive/MyDrive`` folder (or
    ``$COLABUTILS_LOCAL_DRIVE/MyDrive``) is created and returned instead.
    """
    if not is_colab():
        root = drive_root()
        if not root.is_dir():
            root.mkdir(parents=True, exist_ok=True)
            warnings.warn(
                f"Not running in Google Colab; using local folder {root} instead of Google Drive.",
                stacklevel=2,
            )
        return root

    if force_remount or not is_drive_mounted():
        from google.colab import drive

        drive.mount(DRIVE_MOUNTPOINT, force_remount=force_remount, timeout_ms=timeout_ms)
    return drive_root()


def _clean_name(name: str | None) -> str | None:
    if not name:
        return None
    name = Path(str(name)).name
    if name.endswith(".ipynb"):
        name = name[: -len(".ipynb")]
    return name.strip() or None


def _name_from_colab_frontend() -> str | None:
    from google.colab import _message  # undocumented; may change between runtimes

    reply = _message.blocking_request("get_ipynb", request="", timeout_sec=5)
    nb = (reply or {}).get("ipynb") or {}
    return _clean_name(nb.get("metadata", {}).get("colab", {}).get("name"))


def _name_from_sessions_api() -> str | None:
    host = os.environ.get("COLAB_JUPYTER_IP") or socket.gethostbyname(socket.gethostname())
    port = os.environ.get("COLAB_JUPYTER_PORT") or os.environ.get("KMP_TARGET_PORT") or "9000"
    with urllib.request.urlopen(f"http://{host}:{port}/api/sessions", timeout=3) as resp:
        sessions = json.load(resp)
    for session in sessions:
        name = _clean_name(session.get("name") or session.get("path"))
        if name:
            return name
    return None


def _name_from_local_jupyter() -> str | None:
    if os.environ.get("JPY_SESSION_NAME"):
        return _clean_name(os.environ["JPY_SESSION_NAME"])
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    ip = get_ipython()
    if ip is None:
        return None
    return _clean_name(ip.user_ns.get("__vsc_ipynb_file__"))


def notebook_name(default: str = DEFAULT_NOTEBOOK_NAME) -> str:
    """Best-effort name of the running notebook, without the ``.ipynb`` suffix.

    Colab has no official API for this, so several sources are tried in order:
    ``$COLABUTILS_NOTEBOOK_NAME``, the Colab frontend, the runtime's Jupyter
    sessions API, then local Jupyter / VS Code. Falls back to ``default``.
    """
    override = _clean_name(os.environ.get(NOTEBOOK_NAME_ENV))
    if override:
        return override

    sources = [_name_from_colab_frontend, _name_from_sessions_api] if is_colab() else []
    sources.append(_name_from_local_jupyter)
    for source in sources:
        try:
            name = source()
        except Exception:
            continue
        if name:
            return name
    return default


class SecretAccessError(RuntimeError):
    """The secret exists in Colab, but this notebook has not been granted access to it."""


def colab_secret(name: str) -> str | None:
    """Value of a secret from Colab's Secrets panel (key icon), or ``None`` if not set.

    Raises :class:`SecretAccessError` when the secret exists but the notebook's access
    toggle is off.
    """
    if not is_colab():
        return None
    from google.colab import userdata

    try:
        return userdata.get(name)
    except Exception as exc:  # userdata raises its own exception classes
        if type(exc).__name__ == "NotebookAccessError":
            raise SecretAccessError(name) from exc
        return None


def serve_port(port: int, height: int = 600, path: str = "/") -> None:
    """Display a web server running on ``localhost:port`` inside the notebook output."""
    if is_colab():
        from google.colab import output

        output.serve_kernel_port_as_iframe(port, path=path, height=str(height))
        return

    from IPython.display import IFrame, display

    display(IFrame(f"http://localhost:{port}{path}", width="100%", height=height))
