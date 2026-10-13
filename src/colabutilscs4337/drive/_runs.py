"""Run folders: ``<runs_dir>/<notebook>_<YYYYmmdd-HHMMSS>_<slug>/``."""

from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .. import _colab
from ._config import get_config

TIMESTAMP_FORMAT = "%Y%m%d-%H%M%S"
RUN_INFO_FILENAME = "run.json"

_ADJECTIVES = (
    "amber",
    "bold",
    "brave",
    "bright",
    "calm",
    "clever",
    "cosmic",
    "crisp",
    "eager",
    "fancy",
    "gentle",
    "golden",
    "happy",
    "jolly",
    "keen",
    "lively",
    "lucky",
    "mellow",
    "nimble",
    "proud",
    "quick",
    "quiet",
    "rapid",
    "shiny",
    "silent",
    "smooth",
    "snowy",
    "solar",
    "steady",
    "sunny",
    "swift",
    "tidy",
    "vivid",
    "witty",
    "zesty",
)
_NOUNS = (
    "badger",
    "beaver",
    "bison",
    "cobra",
    "condor",
    "coyote",
    "falcon",
    "ferret",
    "gecko",
    "heron",
    "ibis",
    "jaguar",
    "koala",
    "lemur",
    "lynx",
    "marmot",
    "moose",
    "narwhal",
    "ocelot",
    "otter",
    "owl",
    "panda",
    "pelican",
    "puffin",
    "quokka",
    "raven",
    "salmon",
    "sparrow",
    "tapir",
    "tiger",
    "toucan",
    "walrus",
    "wombat",
    "yak",
    "zebra",
)


def slugify(text: str, max_length: int = 60) -> str:
    """Filesystem-safe slug: ``"Lab 3 - CNNs (v2)"`` -> ``"Lab-3-CNNs-v2"``."""
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", text.strip())
    text = re.sub(r"-{2,}", "-", text).strip("-._")
    return text[:max_length].rstrip("-._") or "run"


def random_slug(rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    return f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}"


@dataclass(frozen=True)
class Run:
    """A run folder. Usable anywhere a path is: ``open(run / "log.txt", "w")``."""

    path: Path
    notebook: str
    timestamp: str
    slug: str

    def __fspath__(self) -> str:
        return str(self.path)

    def __truediv__(self, other: str | os.PathLike) -> Path:
        return self.path / other

    def __str__(self) -> str:
        return str(self.path)

    @property
    def name(self) -> str:
        return self.path.name

    def subdir(self, *parts: str) -> Path:
        path = self.path.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


def _now(timezone: str) -> datetime:
    try:
        return datetime.now(ZoneInfo(timezone))
    except Exception:
        return datetime.now(ZoneInfo("UTC"))


def run_folder_name(notebook: str, timestamp: str, slug: str) -> str:
    return f"{slugify(notebook)}_{timestamp}_{slugify(slug)}"


def new_run(
    slug: str | None = None,
    *,
    notebook: str | None = None,
    runs_dir: str | os.PathLike | None = None,
    mount: bool = True,
    metadata: dict | None = None,
) -> Run:
    """Create and return a new run folder on Drive.

    The folder is ``<runs_dir>/<notebook>_<YYYYmmdd-HHMMSS>_<slug>``. ``slug`` labels
    this run (for example ``"resnet34-lr1e-3"``); a random one like ``"brave-otter"``
    is generated when omitted. A ``run.json`` with runtime details is written inside.
    """
    if mount and runs_dir is None:
        _colab.mount()
    cfg = get_config(runs_dir=str(runs_dir) if runs_dir is not None else None)
    notebook = notebook or _colab.notebook_name()
    slug = slugify(slug) if slug else random_slug()
    timestamp = _now(cfg.timezone).strftime(TIMESTAMP_FORMAT)

    base = cfg.runs_path / run_folder_name(notebook, timestamp, slug)
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    path.mkdir(parents=True)

    run = Run(path=path, notebook=notebook, timestamp=timestamp, slug=slug)
    _write_run_info(run, cfg.timezone, metadata)
    return run


def _write_run_info(run: Run, timezone: str, metadata: dict | None) -> None:
    from ..env import runtime_info

    info = {
        "notebook": run.notebook,
        "slug": run.slug,
        "timestamp": run.timestamp,
        "timezone": timezone,
        "runtime": runtime_info(),
        "metadata": metadata or {},
    }
    (run.path / RUN_INFO_FILENAME).write_text(json.dumps(info, indent=2, default=str) + "\n")


def list_runs(notebook: str | None = None, runs_dir: str | os.PathLike | None = None) -> list[Path]:
    """Run folders, oldest first. Filter to one notebook with ``notebook=``."""
    root = Path(runs_dir) if runs_dir is not None else get_config().runs_path
    if not root.is_dir():
        return []
    prefix = f"{slugify(notebook)}_" if notebook else ""
    runs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    return sorted(runs, key=lambda p: (_timestamp_of(p), p.name))


def latest_run(
    notebook: str | None = None, runs_dir: str | os.PathLike | None = None
) -> Path | None:
    runs = list_runs(notebook, runs_dir)
    return runs[-1] if runs else None


def _timestamp_of(path: Path) -> str:
    match = re.search(r"_(\d{8}-\d{6})_", path.name)
    return match.group(1) if match else ""
