"""Where a project's configs, data, and runs are, and how to find a run again.

A *project* is a folder (for example ``LitWBTrainingBasicConvnet``) inside a *repo* that
holds a shared ``.env``, ``data/`` and ``runs/``. Every project of the course repo used
to carry its own copy of this code in ``utils/paths.py`` and ``models/loading.py``;
:class:`Project` is that code once::

    from pathlib import Path
    import colabutilscs4337 as cu

    PROJECT = cu.project.Project(Path(__file__).resolve().parent)
    PROJECT.data_dir()                       # <repo>/<DATA_DIR from .env>
    PROJECT.resolve_config("config01")       # <project>/configs/config01.json
    PROJECT.find_checkpoint("config01")      # newest run's best checkpoint

Layout of the results: ``<OUTPUT_DIR>/<project name>/<config name>/<timestamp>/``.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from ..env._secrets import read_env_file

RUN_ID_FORMAT = "%Y-%m-%d_%H-%M-%S"
"""Format of the timestamp that names a run folder, e.g. ``2026-10-02_12-29-09``."""
ENV_FILENAME = ".env"
DEFAULT_DATA_DIR = "data"
DEFAULT_OUTPUT_DIR = "runs"


@dataclass(frozen=True)
class RunDir:
    """A freshly created run folder and its timestamp."""

    run_id: str
    path: Path

    def __fspath__(self) -> str:
        return str(self.path)

    def __truediv__(self, other: str | os.PathLike) -> Path:
        return self.path / other


class Project:
    """Paths and run lookup for one project folder of a repo.

    Args:
        project_dir: The project folder, e.g. ``Path(__file__).resolve().parent`` from
            the project's ``main.py``.
        repo_dir: The repo root holding ``.env``, ``data/`` and ``runs/``. Defaults to
            the project's parent folder.
        checkpoint_suffix: Extension of the checkpoints :meth:`find_checkpoint` looks for.
    """

    def __init__(
        self,
        project_dir: str | os.PathLike,
        repo_dir: str | os.PathLike | None = None,
        *,
        checkpoint_suffix: str = ".ckpt",
    ) -> None:
        self.dir = Path(project_dir).expanduser().resolve()
        self.name = self.dir.name
        self.repo_dir = Path(repo_dir).expanduser().resolve() if repo_dir else self.dir.parent
        self.configs_dir = self.dir / "configs"
        self.env_file = self.repo_dir / ENV_FILENAME
        self.checkpoint_suffix = checkpoint_suffix

    def __repr__(self) -> str:
        return f"Project({self.name!r}, repo_dir={str(self.repo_dir)!r})"

    # ---- environment and paths -------------------------------------------------------

    def load_env(self) -> bool:
        """Put the repo's ``.env`` into ``os.environ`` (names already set are kept).

        Replaces ``dotenv.load_dotenv(REPO_DIR / ".env")``. Returns whether the file exists.
        """
        if not self.env_file.is_file():
            return False
        for key, value in read_env_file(self.env_file).items():
            os.environ.setdefault(key, value)
        return True

    def resolve_repo_path(self, path: str | os.PathLike) -> Path:
        """``path`` against the repo root unless it is absolute, so data and results land in
        the same place whatever the current working directory is."""
        path = Path(path).expanduser()
        return path if path.is_absolute() else self.repo_dir / path

    def data_dir(self, *parts: str) -> Path:
        """``DATA_DIR`` from ``.env`` (default ``data``), as an absolute path, plus ``parts``."""
        self.load_env()
        base = self.resolve_repo_path(os.getenv("DATA_DIR") or DEFAULT_DATA_DIR)
        return base.joinpath(*parts)

    def output_dir(self) -> Path:
        """``OUTPUT_DIR`` from ``.env`` (default ``runs``), as an absolute path."""
        self.load_env()
        return self.resolve_repo_path(os.getenv("OUTPUT_DIR") or DEFAULT_OUTPUT_DIR)

    def runs_dir(self) -> Path:
        """This project's results folder, ``<OUTPUT_DIR>/<project name>``."""
        return self.output_dir() / self.name

    # ---- configs ---------------------------------------------------------------------

    def resolve_config(self, config: str | os.PathLike) -> Path:
        """Locate a JSON config: as given, then relative to the project, then in ``configs/``,
        each with and without ``.json``. ``"config01"``, ``"config01.json"`` and
        ``"configs/config01.json"`` all work.

        Raises:
            FileNotFoundError: Nothing matched; the message lists the available configs.
        """
        config = Path(config).expanduser()
        for base in (config, self.dir / config, self.configs_dir / config):
            for candidate in (base, base.with_suffix(".json")):
                if candidate.is_file():
                    return candidate.resolve()
        available = sorted(p.name for p in self.configs_dir.glob("*.json"))
        raise FileNotFoundError(f"Config {str(config)!r} not found. Available: {available}")

    def read_config(self, config: str | os.PathLike) -> tuple[Path, dict[str, Any]]:
        """:meth:`resolve_config` and parse it: ``(path, config dict)``."""
        path = self.resolve_config(config)
        return path, json.loads(path.read_text())

    # ---- runs ------------------------------------------------------------------------

    def new_run_dir(self, config_name: str, run_id: str | None = None) -> RunDir:
        """Create ``<OUTPUT_DIR>/<project>/<config_name>/<timestamp>/``.

        ``config_name`` is the config file's stem, e.g. ``"config01"``.
        """
        run_id = run_id or datetime.now().astimezone().strftime(RUN_ID_FORMAT)
        path = self.runs_dir() / config_name / run_id
        path.mkdir(parents=True, exist_ok=True)
        return RunDir(run_id, path)

    def save_run_files(
        self,
        run_dir: str | os.PathLike,
        config_path: str | os.PathLike,
        hparams: dict[str, Any],
        *,
        resumed_from: str | os.PathLike | None = None,
    ) -> None:
        """Write ``config.json`` (a copy of the config) and ``hparams.json`` into ``run_dir``."""
        run_dir = Path(run_dir)
        shutil.copy(config_path, run_dir / "config.json")
        hparams = dict(hparams)
        if resumed_from:
            hparams["resumed_from"] = str(resumed_from)
        (run_dir / "hparams.json").write_text(json.dumps(hparams, indent=2, default=str))

    def resolve_run(self, run: str | os.PathLike) -> Path:
        """Find ``run`` as given, in the repo root, or in ``<OUTPUT_DIR>/<project name>``.

        Raises:
            FileNotFoundError: Not found; the message lists the runs saved for this project
                and reminds you to mount Google Drive when there are none.
        """
        path = Path(run).expanduser()
        outputs = self.runs_dir()
        for candidate in (path, self.repo_dir / path, outputs / path):
            if candidate.exists():
                return candidate.resolve()
        if outputs.is_dir():
            found = sorted(d.name for d in outputs.iterdir() if d.is_dir())
            hint = f"Runs saved for this project: {found or 'none yet'}"
        else:
            hint = (
                f"{outputs} does not exist: no run of {self.name} was saved there yet. Train "
                "first; in Colab, mount Google Drive before training and before loading"
            )
        raise FileNotFoundError(
            f"{str(run)!r} not found here, in {self.repo_dir}, or in {outputs}. {hint}"
        )

    def find_checkpoint(self, run: str | os.PathLike, which: str = "best") -> Path:
        """Locate a checkpoint saved by a training run.

        ``run`` can be a checkpoint file, a run folder
        (``<OUTPUT_DIR>/<project>/config01/<timestamp>``), or a config's folder
        (``.../config01``), which picks its newest run that has a checkpoint. Relative paths
        are tried from the current folder, the repo root, and ``<OUTPUT_DIR>/<project>``, so
        ``"config01"`` and ``"config01/2026-10-02_12-29-09"`` both work.

        Args:
            run: Checkpoint file, run folder, or config folder.
            which: ``"best"`` or ``"last"``.

        Raises:
            ValueError: ``which`` is not ``"best"`` or ``"last"``.
            FileNotFoundError: No matching checkpoint exists.
        """
        if which not in ("best", "last"):
            raise ValueError(f"which must be 'best' or 'last', got {which!r}")
        path = self.resolve_run(run)
        if path.is_file():
            return path

        pattern = f"{which}*{self.checkpoint_suffix}"
        if not any(path.glob(pattern)):
            runs = sorted(d for d in path.iterdir() if d.is_dir() and any(d.glob(pattern)))
            if runs:
                path = runs[-1]
        checkpoints = list(path.glob(pattern))
        if not checkpoints:
            raise FileNotFoundError(f"No {pattern} checkpoint in {path} or its run folders")
        return max(checkpoints, key=lambda p: p.stat().st_mtime)

    def find_resume_checkpoint(self, run: str | os.PathLike) -> Path:
        """The checkpoint to continue training from: the run's last epoch.

        ``run`` is looked up as in :meth:`find_checkpoint`; a run stopped before it saved a
        ``last`` checkpoint continues from its best one.
        """
        path = self.resolve_run(run)
        if path.is_file():
            return path
        run_dir = self.find_checkpoint(path, "best").parent
        last = list(run_dir.glob(f"last*{self.checkpoint_suffix}"))
        if last:
            return max(last, key=lambda p: p.stat().st_mtime)
        return self.find_checkpoint(run_dir, "best")

    def run_config(self, checkpoint: str | os.PathLike) -> dict[str, Any]:
        """The config of the run that saved ``checkpoint``.

        ``config.json`` is copied into the run folder when the run finishes. For a run that
        was interrupted, the project's config of the same name is used, with a warning.
        """
        checkpoint = Path(checkpoint)
        saved = checkpoint.parent / "config.json"
        if saved.is_file():
            return json.loads(saved.read_text())
        config_path = self.resolve_config(checkpoint.parent.parent.name)
        warnings.warn(
            f"{saved} not found (the run did not finish); using {config_path}", stacklevel=2
        )
        return json.loads(config_path.read_text())


# ---- runs_summary.csv ----------------------------------------------------------------


def append_run_summary(summary_path: str | os.PathLike, row: dict[str, Any]) -> None:
    """Append one run's key metrics to a CSV shared by all runs of a config.

    The header is taken from the keys of ``row``. When ``row`` has columns the file does
    not have yet (a new setting was added to the config), the file is rewritten with the
    new columns added, left empty for the earlier runs.
    """
    summary_path = Path(summary_path)
    old_fieldnames: list[str] = []
    old_rows: list[dict[str, Any]] = []
    if summary_path.exists():
        with summary_path.open(newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)
            old_fieldnames = list(reader.fieldnames or [])

    if old_fieldnames and set(row) <= set(old_fieldnames):
        with summary_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=old_fieldnames).writerow(row)
        return

    fieldnames = old_fieldnames + [key for key in row if key not in old_fieldnames]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([*old_rows, row])


def newest_summaries(project_runs: str | os.PathLike) -> list[dict[str, str]]:
    """The last ``runs_summary.csv`` row of every config folder, with a ``config`` column.

    Rows are sorted by config name; configs without a summary are skipped.
    """
    rows = []
    for summary in sorted(Path(project_runs).glob("*/runs_summary.csv")):
        with summary.open(newline="") as f:
            runs = list(csv.DictReader(f))
        if runs:
            rows.append({"config": summary.parent.name, **runs[-1]})
    return rows


def compare_runs(
    project_runs: str | os.PathLike,
    metric: str,
    *,
    baseline_column: str = "strategy",
    baseline_value: str = "from scratch",
) -> list[dict[str, Any]]:
    """:func:`newest_summaries` plus each row's gain over a baseline config.

    The baseline is the first row whose ``baseline_column`` equals ``baseline_value``.
    Every row gets ``gain_vs_baseline``: its ``metric`` minus the baseline's (``None`` when
    there is no baseline or the metric is missing).
    """
    rows: list[dict[str, Any]] = list(newest_summaries(project_runs))
    baseline = next((r for r in rows if r.get(baseline_column) == baseline_value), None)
    for row in rows:
        row["gain_vs_baseline"] = (
            float(row[metric]) - float(baseline[metric])
            if baseline and row.get(metric) and baseline.get(metric)
            else None
        )
    return rows
