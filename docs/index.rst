colabutilscs4337
================

Google Colab utilities for **CS4337** (computer vision with PyTorch and fastai).
One import gives you short, safe helpers for the chores around every assignment:

* **Setup**: install extra tools without breaking Colab's preinstalled PyTorch and
  fastai, load API keys from Colab Secrets, check the runtime.
* **Runs and data**: mount Google Drive and get a fresh, timestamped run folder per
  experiment.
* **Look at your data**: image grids and one-row-per-class views of any dataset.
* **Look inside your model**: ``torchinfo`` summaries, the Netron graph viewer,
  loading fastai and PyTorch checkpoints.
* **Explain predictions**: Grad-CAM heatmaps, layer activations, feature maps.

.. code-block:: python

   %pip install -q --upgrade colabutilscs4337
   import colabutilscs4337 as cu

   cu.install("all")                        # torchinfo, netron, grad-cam (safely)
   cu.env.load_secrets("WANDB_API_KEY")     # from the Colab Secrets panel
   run = cu.drive.new_run("baseline")       # MyDrive/runs/<notebook>_<time>_baseline/
   cu.viz.show_by_class(train_ds)           # one row of examples per class
   cam = cu.viz.gradcam(model, x)           # where is the model looking?
   cu.viz.show_cam(x, cam)

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   guide/secrets
   guide/runs_and_data
   guide/data_visualization
   guide/model_inspection
   guide/cnn_visualization
   guide/compatibility

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/install
   api/env
   api/drive
   api/tools
   api/models
   api/viz

.. toctree::
   :maxdepth: 1
   :caption: Project

   development
   changelog

.. toctree::
   :caption: Links

   PyPI package <https://pypi.org/project/colabutilscs4337/>
   GitHub repository <https://github.com/TXSTCODEPLAYGROUND/ColabUtilsCS4337>
   Issue tracker <https://github.com/TXSTCODEPLAYGROUND/ColabUtilsCS4337/issues>
   Colab runtime versions <https://github.com/googlecolab/backend-info>
