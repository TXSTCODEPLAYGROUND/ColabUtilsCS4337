"""PyTorch Lightning plumbing shared by the course projects.

Needs ``lightning`` (``cu.install("lit")``); W&B helpers also need ``wandb``
(``cu.install("tracking")``). Importing this module imports Lightning.

* :func:`start_run`: CLI, config, resume, run folder, CSV logger, checkpoint callbacks.
* :func:`connect_wandb`, :func:`connect_wandb_run`, :func:`connect_litlogger`: trackers
  that fall back to offline or local logging instead of failing.
* :class:`ResumableEarlyStopping`, :func:`checkpoint_callbacks`.
"""

from ._callbacks import ResumableEarlyStopping, checkpoint_callbacks
from ._run import TrainingRun, parse_cli, start_run
from ._tracking import connect_litlogger, connect_wandb, connect_wandb_run

__all__ = [
    "ResumableEarlyStopping",
    "TrainingRun",
    "checkpoint_callbacks",
    "connect_litlogger",
    "connect_wandb",
    "connect_wandb_run",
    "parse_cli",
    "start_run",
]
