"""Copy an env file (API keys, settings) from Google Drive into a project.

Typical use in a notebook, right after the project is cloned::

    cu.env.sync_env_file("keys/cs4337.env", dest="/content/CodeCS4337Fall2026/.env")

The file is looked up relative to ``MyDrive`` (an absolute path is used as given).
Drive is mounted first when needed; if it cannot be mounted, or the file is not there,
:class:`EnvFileError` says exactly what is missing instead of letting the run continue
without its keys.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from .. import _colab
from ._secrets import DRIVE_ENV_FILE, load_secrets


class EnvFileError(FileNotFoundError):
    """The env file could not be copied: Drive is not mounted, or the file does not exist."""


def drive_env_path(source: str | os.PathLike | None = None) -> Path:
    """Where ``source`` is on Drive: relative paths are inside ``MyDrive``.

    Without ``source`` this is ``MyDrive/.colabutilscs4337/.env``, the file
    :func:`~colabutilscs4337.env.load_secrets` already looks in.
    """
    path = Path(source) if source is not None else DRIVE_ENV_FILE
    path = path.expanduser()
    return path if path.is_absolute() else _colab.drive_root() / path


def sync_env_file(
    source: str | os.PathLike | None = None,
    dest: str | os.PathLike = ".env",
    *,
    mount: bool = True,
    append: bool = False,
    load: bool = False,
    verbose: bool = True,
) -> Path:
    """Copy an env file from Google Drive to ``dest`` (and optionally load it).

    Args:
        source: The file on Drive: relative to ``MyDrive``, or absolute. Defaults to
            ``MyDrive/.colabutilscs4337/.env``.
        dest: Where to write the copy, normally the project's ``.env``.
        mount: Mount Drive first when running in Colab and it is not mounted yet.
        append: Add the Drive file after what ``dest`` already contains (for example
            the repo's ``.envcolab`` settings) instead of replacing it. Later lines win
            when a name appears twice.
        load: Also put the values into ``os.environ`` (names already set are kept).
        verbose: Print where the file came from. Values are never printed.

    Returns:
        The path of the written file. It is made readable by its owner only.

    Raises:
        EnvFileError: Drive is not mounted (and could not be), or ``source`` is not
            on Drive.
    """
    path = drive_env_path(source)
    if _colab.is_colab() and not _colab.is_drive_mounted():
        if mount:
            _colab.mount()
        if not _colab.is_drive_mounted():
            fix = (
                "Run `from google.colab import drive; drive.mount('/content/drive')` first."
                if _colab.is_notebook()
                else _colab.SSH_MOUNT_HINT
            )
            raise EnvFileError(f"Google Drive is not mounted, so {path} cannot be read. {fix}")
    if not path.is_file():
        how = (
            "Upload it to Google Drive at that path (a file of NAME=value lines)"
            if _colab.is_notebook() or not _colab.is_colab()
            else "Create it on Drive from the session, a file of NAME=value lines, e.g. "
            f'`echo "WANDB_API_KEY=your-key" >> {path}` (the Drive mount is shared with '
            "the session)"
        )
        raise EnvFileError(
            f"Env file not found: {path}. {how}, or pass the right location as `source`."
        )

    dest = Path(dest).expanduser()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if append and dest.is_file():
        existing = dest.read_text()
        sep = "" if existing.endswith("\n") or not existing else "\n"
        dest.write_text(existing + sep + path.read_text())
    else:
        shutil.copyfile(path, dest)
    dest.chmod(0o600)
    if verbose:
        print(f"Copied env file {path} -> {dest}")
    if load:
        load_secrets(env_file=dest, verbose=False)
    return dest
