"""Experiment trackers that degrade gracefully: W&B (online, else offline) and LitLogger."""

from __future__ import annotations

import contextlib
import os
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

LITLOGGER_KEYS = ("LIGHTNING_USER_ID", "LIGHTNING_API_KEY")

try:  # the exception's module path changes between wandb releases
    from wandb.sdk.mailbox.mailbox import MailboxClosedError
except ImportError:  # pragma: no cover - depends on the installed wandb

    class MailboxClosedError(Exception):  # type: ignore[no-redef]
        """Placeholder when wandb does not define it."""


def _connect(start: Callable[[bool], Any], save_dir: Path, offline: bool) -> Any:
    """Run ``start(offline)``: online if there is a key and a login, offline otherwise."""
    import wandb

    def start_once(offline: bool) -> Any:
        try:
            return start(offline)
        except MailboxClosedError:
            # W&B's background process died (e.g. after interrupting a notebook cell) and
            # every later run in this Python session would fail; drop the dead connection so
            # the next run starts a new process.
            with contextlib.suppress(Exception):
                wandb.teardown()
            return start(offline)

    sync_hint = (
        "logging to W&B offline. Upload the run later with: "
        f"wandb sync {save_dir / 'wandb'}/offline-run-*"
    )
    if offline:
        return start_once(True)
    if not os.getenv("WANDB_API_KEY"):
        warnings.warn(f"WANDB_API_KEY is not set (add it to .env): {sync_hint}", stacklevel=3)
        return start_once(True)
    try:
        return start_once(False)
    except (wandb.errors.Error, OSError) as error:
        wandb.finish()
        warnings.warn(f"Could not log in to W&B ({error}): {sync_hint}", stacklevel=3)
        return start_once(True)


def connect_wandb(
    project: str,
    name: str,
    group: str,
    save_dir: str | os.PathLike,
    config: dict,
    offline: bool = False,
    **init_kwargs: Any,
):
    """Start a W&B run for Lightning, online when possible and offline otherwise.

    Args:
        project: W&B project name.
        name: Run name shown in W&B.
        group: Run group; W&B can show or average the runs of a group together.
        save_dir: Folder where W&B writes its local files (``save_dir/wandb``).
        config: Settings saved with the run, shown as columns in W&B.
        offline: Skip the online attempt, e.g. after an earlier run of the same search
            already failed to log in.
        **init_kwargs: More arguments for ``wandb.init``, e.g. ``job_type``.

    Returns:
        A ``WandbLogger`` whose run has already started; ``logger.experiment.offline``
        tells which mode it is in.
    """
    from lightning.pytorch.loggers import WandbLogger

    save_dir = Path(save_dir)

    def start(offline: bool) -> WandbLogger:
        logger = WandbLogger(
            project=project,
            name=name,
            group=group,
            save_dir=save_dir,
            config=config,
            # Passed to wandb.init: unlike offline=True, which only sets WANDB_MODE, this
            # still works after a failed online attempt.
            mode="offline" if offline else "online",
            **init_kwargs,
        )
        _ = logger.experiment
        return logger

    return _connect(start, save_dir, offline)


def connect_wandb_run(
    project: str,
    name: str,
    group: str,
    save_dir: str | os.PathLike,
    config: dict,
    offline: bool = False,
    **init_kwargs: Any,
):
    """Like :func:`connect_wandb`, for code without Lightning (e.g. Ultralytics YOLO).

    Returns:
        The started ``wandb`` run, online or offline (``run.offline`` tells which).
    """
    import wandb

    save_dir = Path(save_dir)

    def start(offline: bool):
        return wandb.init(
            project=project,
            name=name,
            group=group,
            dir=save_dir,
            config=config,
            mode="offline" if offline else "online",
            **init_kwargs,
        )

    return _connect(start, save_dir, offline)


def connect_litlogger(run_dir: str | os.PathLike, name: str):
    """Create a ``LitLogger`` and connect now, so a failure doesn't stop training later.

    Credentials come only from ``LIGHTNING_USER_ID`` and ``LIGHTNING_API_KEY``. Without them
    the Lightning SDK would fall back to a browser login, which hangs in Colab and over SSH,
    so LitLogger is skipped.

    Args:
        run_dir: Run folder; LitLogger keeps its local files in ``run_dir/litlogger``.
        name: Experiment name on lightning.ai (a timestamp is added to it).

    Returns:
        The connected logger, or ``None`` if it is not configured or can't connect.
    """
    from lightning.pytorch.loggers import LitLogger

    missing = [key for key in LITLOGGER_KEYS if not os.getenv(key)]
    if missing:
        warnings.warn(f"LitLogger disabled: add {' and '.join(missing)} to .env", stacklevel=2)
        return None
    try:
        # save_logs=True would record the terminal by relaunching the script from sys.argv,
        # which breaks `python -m` and notebooks.
        logger = LitLogger(root_dir=Path(run_dir) / "litlogger", name=name, save_logs=False)
        _ = logger.experiment
        return logger
    except (RuntimeError, OSError) as err:
        warnings.warn(f"LitLogger disabled, logging locally only: {err}", stacklevel=2)
        return None
