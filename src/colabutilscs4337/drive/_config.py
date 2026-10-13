"""Persistent defaults stored on Drive so they survive runtime resets.

Precedence (highest first): explicit arguments, environment variables
(``COLABUTILS_DATA_DIR``, ``COLABUTILS_RUNS_DIR``, ``COLABUTILS_TZ``), the config
file ``MyDrive/.colabutilscs4337/config.json``, then built-in defaults.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from .. import _colab

CONFIG_DIRNAME = ".colabutilscs4337"
CONFIG_FILENAME = "config.json"
ENV_VARS = {
    "data_dir": "COLABUTILS_DATA_DIR",
    "runs_dir": "COLABUTILS_RUNS_DIR",
    "timezone": "COLABUTILS_TZ",
}


@dataclass
class Config:
    data_dir: str
    runs_dir: str
    timezone: str = "UTC"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser()

    @property
    def runs_path(self) -> Path:
        return Path(self.runs_dir).expanduser()


def default_config() -> Config:
    # Datasets go on the runtime's local disk (fast); runs go to Drive (persistent).
    data_dir = "/content/data" if _colab.is_colab() else str(Path.cwd() / "data")
    return Config(data_dir=data_dir, runs_dir=str(_colab.drive_root() / "runs"))


def config_path() -> Path:
    return _colab.drive_root() / CONFIG_DIRNAME / CONFIG_FILENAME


def _read_file() -> dict:
    path = config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    known = {f.name for f in fields(Config)}
    return {k: v for k, v in data.items() if k in known and v}


def get_config(**overrides: str | None) -> Config:
    """Current configuration, merging defaults, the Drive config file, env vars, and overrides."""
    values = asdict(default_config())
    values.update(_read_file())
    for key, env in ENV_VARS.items():
        if os.environ.get(env):
            values[key] = os.environ[env]
    values.update({k: str(v) for k, v in overrides.items() if v is not None})
    return Config(**values)


def set_config(
    data_dir: str | os.PathLike | None = None,
    runs_dir: str | os.PathLike | None = None,
    timezone: str | None = None,
    save: bool = True,
) -> Config:
    """Update defaults and (by default) save them to Drive for future sessions."""
    current = {**asdict(default_config()), **_read_file()}
    updates = {"data_dir": data_dir, "runs_dir": runs_dir, "timezone": timezone}
    current.update({k: str(v) for k, v in updates.items() if v is not None})
    if timezone is not None:
        _validate_timezone(timezone)
    if save:
        if _colab.is_colab():
            _colab.mount()
        path = config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2) + "\n")
    return get_config()


def reset_config() -> None:
    """Delete the saved config file so built-in defaults apply again."""
    path = config_path()
    if path.is_file():
        path.unlink()


def data_dir(*parts: str, create: bool = True) -> Path:
    """Path inside the data folder (``/content/data`` in Colab by default)."""
    path = get_config().data_path.joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_timezone(name: str) -> None:
    from zoneinfo import ZoneInfo

    ZoneInfo(name)
