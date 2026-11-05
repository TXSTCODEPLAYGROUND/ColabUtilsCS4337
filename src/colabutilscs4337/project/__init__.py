"""Course-repo plumbing: setup in Colab, project paths, configs, and finding runs again.

* :func:`setup_colab`: mount Drive, clone the repo, write ``.env``, install packages.
* :class:`Project`: ``.env`` settings, data and output folders, config lookup, run folders,
  checkpoint lookup for loading and resuming.
* :func:`append_run_summary`, :func:`compare_runs`: the ``runs_summary.csv`` files.

Nothing here imports torch or Lightning.
"""

from ._project import (
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    RUN_ID_FORMAT,
    Project,
    RunDir,
    append_run_summary,
    compare_runs,
    newest_summaries,
)
from ._setup import clone_or_pull, repo_name_from_url, setup_colab

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_OUTPUT_DIR",
    "RUN_ID_FORMAT",
    "Project",
    "RunDir",
    "append_run_summary",
    "clone_or_pull",
    "compare_runs",
    "newest_summaries",
    "repo_name_from_url",
    "setup_colab",
]
