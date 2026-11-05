``cu.lit``: PyTorch Lightning plumbing
======================================

Needs ``lightning`` (``cu.install("lit")``) and, for the W&B helpers, ``wandb``
(``cu.install("tracking")``).

.. automodule:: colabutilscs4337.lit
   :no-members:

Starting a run
--------------

.. autofunction:: colabutilscs4337.lit.start_run
.. autoclass:: colabutilscs4337.lit.TrainingRun
   :members:
.. autofunction:: colabutilscs4337.lit.parse_cli

Callbacks
---------

.. autoclass:: colabutilscs4337.lit.ResumableEarlyStopping
.. autofunction:: colabutilscs4337.lit.checkpoint_callbacks

Trackers
--------

.. autofunction:: colabutilscs4337.lit.connect_wandb
.. autofunction:: colabutilscs4337.lit.connect_wandb_run
.. autofunction:: colabutilscs4337.lit.connect_litlogger
