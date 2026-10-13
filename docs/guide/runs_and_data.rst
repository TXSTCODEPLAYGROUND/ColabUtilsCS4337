Runs and data folders
=====================

Colab deletes everything in ``/content`` when the runtime stops. Results you want to
keep (checkpoints, logs, plots) go to **Google Drive**; datasets go on the runtime's
**local disk**, which is much faster to read.

A run folder per experiment
---------------------------

.. code-block:: python

   run = cu.drive.new_run("resnet18-lr1e-3")
   print(run)
   # /content/drive/MyDrive/runs/Lab-3-CNNs_20261012-221500_resnet18-lr1e-3

:func:`~colabutilscs4337.drive.new_run` mounts Drive if needed (Colab asks you to
sign in once) and creates
``MyDrive/runs/<notebook name>_<YYYYmmdd-HHMMSS>_<slug>/``:

* **notebook name**: detected automatically, so runs from different notebooks never
  mix.
* **timestamp**: when the run started (UTC unless you set a time zone, below).
* **slug**: your label for the run. Leave it out to get a random one such as
  ``brave-otter``.

Each run folder starts with a ``run.json`` that records the notebook, time, Python,
GPU, and package versions, so you can tell later exactly what produced a result.

Use the run like a folder path:

.. code-block:: python

   torch.save(model.state_dict(), run / "model.pth")
   with open(run / "notes.txt", "w") as f:
       f.write("first try")
   ckpt_dir = run.subdir("checkpoints")    # creates run/checkpoints/

   learn.path = run.path                    # fastai: learn.save()/export() go here

Find runs later:

.. code-block:: python

   cu.drive.list_runs()                  # all runs, oldest first
   cu.drive.list_runs("Lab 3 - CNNs")    # runs of one notebook
   cu.drive.latest_run()                 # the newest one

The data folder
---------------

.. code-block:: python

   data_dir = cu.drive.data_dir()          # /content/data
   pets_dir = cu.drive.data_dir("pets")    # /content/data/pets (created)

Settings that persist
---------------------

.. code-block:: python

   cu.drive.set_config(timezone="America/Chicago")
   cu.drive.set_config(data_dir="/content/data", runs_dir="/content/drive/MyDrive/cs4337/runs")
   cu.drive.get_config()
   cu.drive.reset_config()                 # back to the defaults

Settings are saved in ``MyDrive/.colabutilscs4337/config.json`` and apply to every
notebook. When a setting is given in more than one place, the first of these wins:

#. A function argument, e.g. ``new_run(runs_dir=...)``.
#. An environment variable: ``COLABUTILS_DATA_DIR``, ``COLABUTILS_RUNS_DIR``,
   ``COLABUTILS_TZ``, ``COLABUTILS_NOTEBOOK_NAME``.
#. The saved config file.
#. The defaults: ``/content/data``, ``MyDrive/runs``, UTC.

The notebook name
-----------------

Colab has no official way to ask for the notebook's name, so
:func:`~colabutilscs4337.env.notebook_name` tries a few. If they all fail you get
``untitled``; set the name yourself with:

.. code-block:: python

   import os
   os.environ["COLABUTILS_NOTEBOOK_NAME"] = "Lab 3 - CNNs"
