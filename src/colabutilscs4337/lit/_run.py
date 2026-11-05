"""The start of every project's ``main()``: parse the CLI, read the config, set up the run."""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..project import Project
from ._callbacks import checkpoint_callbacks


def parse_cli(description: str | None = None) -> tuple[str, str | None]:
    """Read ``--config`` (required) and ``--resume-from`` from the command line.

    Returns:
        ``(config, resume_from)``.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="e.g. config01.json")
    parser.add_argument(
        "--resume-from", help="continue an earlier run, e.g. config01 (its newest run)"
    )
    args = parser.parse_args()
    return args.config, args.resume_from


@dataclass
class TrainingRun:
    """Everything a project's ``main()`` needs from the start of a run.

    Attributes:
        project: The :class:`~colabutilscs4337.project.Project` it belongs to.
        config_path: The resolved config file.
        config: The parsed config.
        run_id: The run's timestamp, ``2026-10-02_12-29-09``.
        run_dir: ``<OUTPUT_DIR>/<project>/<config name>/<run_id>``, already created.
        epochs: Epochs to pass to ``Trainer(max_epochs=...)``. When resuming, the config's
            epochs *added to* the epochs the run already completed.
        resume_checkpoint: The checkpoint to pass to ``trainer.fit(ckpt_path=...)``, or
            ``None`` for a fresh run.
        csv_logger: A ``CSVLogger`` writing ``metrics.csv`` into ``run_dir``.
        checkpoint: The best-checkpoint callback (read ``best_model_score`` after training).
        last_checkpoint: The every-epoch ``last`` checkpoint callback.
    """

    project: Project
    config_path: Path
    config: dict[str, Any]
    run_id: str
    run_dir: Path
    epochs: int
    resume_checkpoint: Path | None
    csv_logger: CSVLogger
    checkpoint: ModelCheckpoint
    last_checkpoint: ModelCheckpoint

    @property
    def config_name(self) -> str:
        """The config file name without ``.json``, e.g. ``"config01"``."""
        return self.config_path.stem

    def save(self, hparams: dict[str, Any]) -> None:
        """Write ``config.json`` and ``hparams.json`` (with ``resumed_from`` when resuming)."""
        self.project.save_run_files(
            self.run_dir, self.config_path, hparams, resumed_from=self.resume_checkpoint
        )

    def append_summary(self, row: dict[str, Any]) -> None:
        """Add a row to ``runs_summary.csv`` next to this config's run folders."""
        from ..project import append_run_summary

        append_run_summary(self.run_dir.parent / "runs_summary.csv", row)


def start_run(
    project: Project,
    config_path: str | os.PathLike | None = None,
    resume_from: str | os.PathLike | None = None,
    *,
    description: str | None = None,
    epochs_key: tuple[str, str] = ("training", "epochs"),
    monitor: str = "val_acc",
    mode: str = "max",
    metric_label: str = "valacc",
) -> TrainingRun:
    """Do the setup every Lightning ``main()`` repeats.

    1. Loads the repo's ``.env``.
    2. When ``config_path`` is ``None``, reads ``--config`` / ``--resume-from`` from the
       command line.
    3. Finds and parses the config.
    4. When resuming, locates the run's last checkpoint and adds the epochs it completed to
       the config's ``epochs``.
    5. Creates the run folder, a ``CSVLogger`` in it, and the best/last checkpoint callbacks.

    Seeding, the data module, the model and the extra loggers stay in the project.

    Example::

        run = cu.lit.start_run(PROJECT, config_path, resume_from, description=__doc__)
        trainer = L.Trainer(
            max_epochs=run.epochs,
            logger=[run.csv_logger],
            callbacks=[run.checkpoint, run.last_checkpoint],
        )
        trainer.fit(model, datamodule=data, ckpt_path=run.resume_checkpoint)

    Args:
        project: The project folder's :class:`~colabutilscs4337.project.Project`.
        config_path: Config name or path; read from the command line when ``None``.
        resume_from: ``"config01"`` (its newest run), a run folder, or a checkpoint.
        description: Shown by ``--help``; its first line is used.
        epochs_key: Where the number of epochs is in the config.
        monitor, mode, metric_label: The metric the best checkpoint tracks, as in
            :func:`~colabutilscs4337.lit.checkpoint_callbacks`.
    """
    project.load_env()
    if config_path is None:
        first_line = description.splitlines()[0] if description else None
        config_path, resume_from = parse_cli(first_line)
    path, config = project.read_config(config_path)

    section, key = epochs_key
    epochs = config[section][key]
    resume_checkpoint = None
    if resume_from is not None:
        resume_checkpoint = project.find_resume_checkpoint(resume_from)
        state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        done = state["epoch"] + 1
        epochs += done
        print(f"Resuming {resume_checkpoint} after {done} epochs, up to {epochs} epochs")
        # Expected: the new run saves its checkpoints in its own folder.
        warnings.filterwarnings("ignore", message=".*dirpath has changed.*")

    run = project.new_run_dir(path.stem)
    csv_logger = CSVLogger(save_dir=project.runs_dir(), name=path.stem, version=run.run_id)
    checkpoint, last_checkpoint = checkpoint_callbacks(
        run.path, monitor=monitor, mode=mode, metric_label=metric_label
    )
    return TrainingRun(
        project=project,
        config_path=path,
        config=config,
        run_id=run.run_id,
        run_dir=run.path,
        epochs=epochs,
        resume_checkpoint=resume_checkpoint,
        csv_logger=csv_logger,
        checkpoint=checkpoint,
        last_checkpoint=last_checkpoint,
    )
