"""One call that replaces a notebook's setup cells: mount Drive, clone, env file, packages."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from .. import _colab
from ..env._drive_env import sync_env_file

COLAB_BASE_DIR = "/content"


def repo_name_from_url(url: str) -> str:
    """``https://github.com/org/Repo.git`` -> ``Repo``."""
    return url.rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def _git(*args: str, cwd: Path | None = None) -> None:
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"`git {' '.join(args)}` failed:\n{result.stderr.strip()}")
    if result.stdout.strip():
        print(result.stdout.strip())


def clone_or_pull(url: str, base_dir: str | os.PathLike = COLAB_BASE_DIR) -> Path:
    """Clone ``url`` into ``base_dir``, or fast-forward the clone that is already there."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    repo = base / repo_name_from_url(url)
    if (repo / ".git").is_dir():
        _git("pull", "--ff-only", cwd=repo)
    else:
        _git("clone", url, str(repo), cwd=base)
    return repo


def setup_colab(
    repo_url: str,
    *,
    env_template: str | None = ".envcolab",
    env_source: str | os.PathLike | None = None,
    requirements: str | None = "requirements.txt",
    base_dir: str | os.PathLike = COLAB_BASE_DIR,
    local_repo: str | os.PathLike | None = None,
    mount: bool = True,
    chdir: bool = True,
    verbose: bool = True,
) -> Path:
    """Get a course repo ready to run in Colab, and return its folder.

    In Colab this does, in order:

    1. Mounts Google Drive (``mount=True``), where the runs are saved.
    2. Clones ``repo_url`` into ``base_dir`` (``/content``), or pulls it if already there.
    3. Writes the repo's ``.env``: a copy of ``env_template`` (``.envcolab``, the settings
       that put data on the local disk and results on Drive), then, when ``env_source``
       is given, the env file at that Drive path added on top (your API keys). A missing
       Drive file raises :class:`~colabutilscs4337.env.EnvFileError`.
    4. Installs the packages of ``requirements`` that are not installed yet, never
       changing Colab's own.
    5. Changes into the repo and puts it on ``sys.path`` so projects can be imported.

    Outside Colab nothing is cloned or copied: the repo is ``local_repo`` (default: the
    parent of the current folder, where a notebook inside a project folder runs) and only
    ``sys.path`` is updated.

    Example::

        REPO = cu.project.setup_colab(
            "https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git",
            env_source="keys/cs4337.env",
        )

    Returns:
        The repo folder.
    """
    if not _colab.is_colab():
        repo = Path(local_repo).expanduser().resolve() if local_repo else Path.cwd().parent
        _add_to_path(repo)
        return repo

    if mount:
        _colab.mount()
    repo = clone_or_pull(repo_url, base_dir)

    env_file = repo / ".env"
    if env_template and (repo / env_template).is_file():
        shutil.copyfile(repo / env_template, env_file)
        if verbose:
            print(f"Copied {env_template} -> .env")
    if env_source is not None:
        sync_env_file(
            env_source, dest=env_file, mount=False, append=env_file.is_file(), verbose=verbose
        )

    if requirements and (repo / requirements).is_file():
        from .._install import install_requirements

        install_requirements(repo / requirements, verbose=verbose)

    if chdir:
        os.chdir(repo)
    _add_to_path(repo)
    return repo


def _add_to_path(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
