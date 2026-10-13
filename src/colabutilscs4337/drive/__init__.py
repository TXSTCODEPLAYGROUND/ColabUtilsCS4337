"""Google Drive mounting, run folders, and persistent defaults."""

from __future__ import annotations

from pathlib import Path

from .. import _colab
from ._config import (
    Config,
    config_path,
    data_dir,
    default_config,
    get_config,
    reset_config,
    set_config,
)
from ._runs import Run, latest_run, list_runs, new_run, random_slug, slugify


def mount(force_remount: bool = False) -> Path:
    """Mount Google Drive at ``/content/drive`` (no-op if mounted); returns "My Drive"."""
    return _colab.mount(force_remount=force_remount)


def is_mounted() -> bool:
    return _colab.is_drive_mounted()


__all__ = [
    "Config",
    "Run",
    "config_path",
    "data_dir",
    "default_config",
    "get_config",
    "is_mounted",
    "latest_run",
    "list_runs",
    "mount",
    "new_run",
    "random_slug",
    "reset_config",
    "set_config",
    "slugify",
]
