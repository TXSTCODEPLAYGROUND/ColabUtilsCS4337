"""Lightning callbacks shared by the course projects."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStoppingReason


class ResumableEarlyStopping(EarlyStopping):
    """Stop training when the monitored metric stops improving, resumable.

    The same as Lightning's ``EarlyStopping``, except when a run is resumed. The checkpoint
    stores the callback's state, and Lightning's version would restore all of it: a run
    that stopped early would stop again after one epoch, and a new ``patience`` in the
    config would be ignored. Here only the best score is restored, so the resumed run must
    still beat the earlier run's best; the epochs without improvement are counted from zero,
    with the config's ``patience``.
    """

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore the best score from a checkpoint, and nothing else."""
        self.best_score = state_dict["best_score"]
        self.wait_count = 0
        self.stopped_epoch = 0
        self.stopping_reason = EarlyStoppingReason.NOT_STOPPED
        self.stopping_reason_message = None


def checkpoint_callbacks(
    run_dir: str | os.PathLike,
    monitor: str = "val_acc",
    mode: str = "max",
    metric_label: str = "valacc",
) -> tuple[ModelCheckpoint, ModelCheckpoint]:
    """The best-checkpoint and last-checkpoint callbacks every project uses.

    The best one is named ``best_epoch07_valacc0.8912.ckpt`` (``metric_label`` names the
    metric in the file name). The ``last`` one is rewritten every epoch with no version
    counter, so a stopped run can be resumed from its last epoch (``save_last=True`` would
    only copy the best checkpoint).

    Returns:
        ``(best, last)``; pass both to ``Trainer(callbacks=[...])``.
    """
    run_dir = Path(run_dir)
    best = ModelCheckpoint(
        dirpath=run_dir,
        filename="best_epoch{epoch:02d}_" + metric_label + "{" + monitor + ":.4f}",
        auto_insert_metric_name=False,
        monitor=monitor,
        mode=mode,
    )
    last = ModelCheckpoint(dirpath=run_dir, filename="last", enable_version_counter=False)
    return best, last
