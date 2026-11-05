Course projects: setup, paths, and runs
=======================================

The course repo holds many training projects (``LitTrainingBasicConvnet``,
``LitWBTransferLearningResnet``, ...). They share a ``.env``, a ``data/`` folder and a
``runs/`` folder at the repo root. These helpers replace the setup code every project
and notebook used to copy.

One setup cell
--------------

.. code-block:: python

   import colabutilscs4337 as cu

   REPO = cu.project.setup_colab(
       "https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git",
       env_source="keys/cs4337.env",   # a file on your Drive (relative to MyDrive)
   )

In Colab this mounts Drive, clones (or pulls) the repo, writes ``.env`` from the repo's
``.envcolab`` plus your Drive file, installs the packages in ``requirements.txt`` that are
missing, changes into the repo and puts it on ``sys.path``. Outside Colab it only returns the
repo folder. If ``env_source`` is not on Drive, it raises
:class:`~colabutilscs4337.env.EnvFileError` instead of running without your keys.

To copy an env file yourself:

.. code-block:: python

   cu.env.sync_env_file("keys/cs4337.env", dest="/content/CodeCS4337Fall2026/.env", load=True)

Keys from Colab Secrets can be saved to ``.env`` too (for runs from Colab's terminal):

.. code-block:: python

   cu.env.load_secrets("WANDB_API_KEY", persist_to=REPO / ".env")

In a Colab CLI (SSH) session
----------------------------

A session opened with ``colab ssh`` is Colab but not a notebook: Colab Secrets cannot be read
and ``drive.mount`` does not work. Mount Drive from your **local** terminal
(``colab drivemount -s <session>``, then ``colab ssh -s <session>`` again), keep your keys in an
env file on Drive, and call ``setup_colab`` with ``mount=False``:

.. code-block:: bash

   pip install -q colabutilscs4337
   python - <<'EOF'
   import colabutilscs4337 as cu

   cu.project.setup_colab(
       "https://github.com/TXSTCODEPLAYGROUND/CodeCS4337Fall2026.git",
       env_source="/content/drive/MyDrive/cs4337.env",
       mount=False,
   )
   EOF

If Drive is not mounted, or the env file is not there, the error says which command or file
is missing (:class:`~colabutilscs4337.env.EnvFileError`).

Paths, configs, and finding runs
--------------------------------

.. code-block:: python

   from pathlib import Path
   import colabutilscs4337 as cu

   PROJECT = cu.project.Project(Path(__file__).resolve().parent)

   PROJECT.data_dir()                    # <repo>/<DATA_DIR from .env>
   PROJECT.resolve_config("config01")    # <project>/configs/config01.json
   PROJECT.find_checkpoint("config01")   # newest run's best checkpoint
   PROJECT.find_resume_checkpoint("config01")

Folder layout
-------------

The layout is fixed; only the base folders are configurable.

.. code-block:: text

   <repo>/                             the folder with the shared .env
   ├── .env                            DATA_DIR, OUTPUT_DIR, API keys
   ├── data/                           DATA_DIR (default "data")
   ├── runs/                           OUTPUT_DIR (default "runs")
   │   └── <project>/                  e.g. LitWBTrainingBasicConvnet
   │       └── <config>/               the config file's name, e.g. config01
   │           ├── runs_summary.csv    one row per run of this config
   │           └── <timestamp>/        2026-10-02_12-29-09, one per run
   │               ├── config.json     copy of the config
   │               ├── hparams.json    settings the run used (and resumed_from)
   │               ├── metrics.csv     per-epoch metrics
   │               ├── best_epoch07_valacc0.8912.ckpt
   │               ├── last.ckpt       rewritten every epoch, used to resume
   │               └── wandb/          W&B's local files
   └── <project>/
       └── configs/
           ├── config01.json
           └── config02.json

* ``DATA_DIR`` and ``OUTPUT_DIR`` come from ``.env`` (or the environment). A relative
  value is resolved against the repo root, so results land in the same place whatever the
  current folder is; an absolute value is used as is. In Colab ``.envcolab`` puts
  ``DATA_DIR`` on the local disk and ``OUTPUT_DIR`` on Drive.
* ``Project(project_dir, repo_dir=...)`` changes where ``.env`` is looked up (default: the
  project's parent folder). ``checkpoint_suffix=".pt"`` changes the checkpoint extension
  that ``find_checkpoint`` looks for (default ``.ckpt``).
* ``find_checkpoint("config01")`` takes the newest run of that config. A run folder
  (``"config01/2026-10-02_12-29-09"``), a checkpoint file and an absolute path also work.
  The lookup depends on the nesting above.

Configs
-------

* Configs are JSON files in ``<project>/configs/``.
* ``resolve_config`` accepts ``"config01"``, ``"config01.json"``, ``"configs/config01.json"``,
  or a path. The error for an unknown name lists the configs that exist.
* The config's file name (without ``.json``) becomes the run folder's middle level, so two
  configs never share a ``runs_summary.csv``.
* ``cu.lit.start_run`` reads the number of epochs from ``config["training"]["epochs"]``
  (change it with ``epochs_key=``). Everything else in a config belongs to the project.
* A finished run copies its config to ``config.json``. Loading a model later reads that copy
  (``Project.run_config``), so changing ``configs/config01.json`` afterwards does not break
  old runs. A run that did not finish has no copy; the project's config of the same name is
  used, with a warning.

A ``main()`` with Lightning
---------------------------

.. code-block:: python

   run = cu.lit.start_run(PROJECT, config_path, resume_from, description=__doc__)
   wandb_logger = cu.lit.connect_wandb(
       run.config["wandb"]["project"], f"{run.config_name}-{run.run_id}",
       run.config_name, run.run_dir, run.config,
   )
   trainer = L.Trainer(
       max_epochs=run.epochs,
       logger=[run.csv_logger, wandb_logger],
       callbacks=[run.checkpoint, run.last_checkpoint],
   )
   trainer.fit(model, datamodule=data, ckpt_path=run.resume_checkpoint)
   run.save({"model": dict(model.hparams)})
   run.append_summary({"run_id": run.run_id, "test_acc": f"{acc:.4f}"})

Weights & Biases logs online when ``WANDB_API_KEY`` is set and the login works, and offline
(with a ``wandb sync`` hint) otherwise.
