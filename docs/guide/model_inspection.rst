Inspecting and loading models
=============================

Layer-by-layer summary
----------------------

.. code-block:: python

   cu.tools.summary(model)                              # input (1, 3, 224, 224)
   cu.tools.summary(model, input_size=(1, 1, 28, 28))   # MNIST-sized input
   cu.tools.summary(learn)                              # a fastai Learner works too

Shows every layer's output shape and parameter count (uses ``torchinfo``; run
``cu.install("tools")`` once).

Interactive graph with Netron
-----------------------------

.. code-block:: python

   cu.tools.netron(model)                               # drawn inside the cell output
   cu.tools.netron(model, input_size=(1, 1, 28, 28))
   cu.tools.netron("/content/drive/MyDrive/model.onnx") # any file Netron can open
   cu.tools.stop_netron()

In-memory models are exported first (TorchScript by default; ``format="export"``
writes a ``torch.export`` file; ``format="onnx"`` needs ``cu.install("onnx")``).

Loading saved models
--------------------

:func:`~colabutilscs4337.models.load_model` figures out the file type and always
returns a ready-to-use ``torch.nn.Module`` in eval mode, on the GPU when there is one.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - You saved it with
     - Load it with
   * - ``learn.export("export.pkl")`` (fastai)
     - ``cu.models.load_model("export.pkl")``
   * - ``learn.save("stage1")`` (fastai)
     - ``cu.models.load_model("models/stage1.pth", arch=learn)``
   * - ``torch.save(model.state_dict(), "w.pth")``
     - ``cu.models.load_model("w.pth", arch=torchvision.models.resnet18(num_classes=10))``
   * - ``torch.save(model, "full.pt")``
     - ``cu.models.load_model("full.pt")``
   * - TorchScript or ``torch.export`` (``.pt2``)
     - ``cu.models.load_model(path)``

``arch`` is the network the weights belong to: a model instance, a fastai
``Learner``, or a function that builds one (``arch=MyNet``).

.. code-block:: python

   learn = cu.models.load_model("export.pkl", learner=True)   # the whole fastai Learner
   model = cu.models.load_model("w.pth", arch=MyNet, device="cpu")

.. warning::

   fastai ``.pkl`` exports and full PyTorch models are Python *pickles*: loading one
   can run code. Only load files you or your course staff created. Plain weights
   (state dicts) are always loaded safely.
