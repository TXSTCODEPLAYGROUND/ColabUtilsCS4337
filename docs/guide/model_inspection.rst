Inspecting and loading models
=============================

Layer-by-layer summary
----------------------

.. code-block:: python

   cu.tools.summary(model)                              # input (1, 3, 224, 224)
   cu.tools.summary(model, input_size=(1, 1, 28, 28))   # MNIST-sized input
   cu.tools.summary(learn)                              # a fastai Learner works too

Shows every layer's output shape and parameter count (uses ``torchinfo``, installed
with the package).

Input and output shape of one block
-----------------------------------

When you need the exact shape going *into* a block (to build a new head, or to
feed the block on its own), ask the model instead of reading it off the table:

.. code-block:: python

   backbone = torchvision.models.efficientnet_b0().features
   cu.tools.layer_shapes(backbone, backbone[2][0], input_size=(1, 3, 160, 160))
   # 2.0 (MBConv): (1, 16, 80, 80) -> (1, 24, 40, 40)

   s = cu.tools.layer_shapes(backbone, "2.0", (1, 3, 160, 160), verbose=False)
   s.input, s.output      # first input / first output
   s.inputs               # every positional input, e.g. [(1, 16, 80, 80)]
   s.kwargs               # tensors passed by keyword, e.g. {"mask": (1, 80, 80)}

   cu.tools.layer_shapes(model, ["layer1", "layer2.0", "fc"])   # several at once
   cu.tools.layer_shapes(learn, "0.4", x=xb)                     # fastai, real batch

The layer can be the module itself or its dotted name (as listed by
:func:`~colabutilscs4337.viz.layer_names`). The model runs once in eval mode
without gradients, so BatchNorm statistics are not changed. A block that runs more
than once per pass (shared weights) reports its first call.

This replaces the usual hand-written hook:

.. code-block:: python

   inputs = []
   hook = block.register_forward_pre_hook(lambda m, args: inputs.append(tuple(args[0].shape)))
   with torch.no_grad():
       backbone.eval()(torch.zeros(1, 3, 160, 160))
   hook.remove()

Interactive graph with Netron
-----------------------------

.. code-block:: python

   cu.tools.netron(model, input_size=(1, 1, 28, 28))    # drawn inside the cell output
   cu.tools.netron(learn)                               # fastai Learner, (1, 3, 224, 224)
   cu.tools.netron("/content/drive/MyDrive/model.onnx", height=600)  # a saved file
   cu.tools.stop_netron()

This one call replaces the usual Colab recipe of ``model.eval()``, a dummy input on
the right device, ``torch.onnx.export``, ``netron.start`` on a free port, and
``google.colab.output.serve_kernel_port_as_iframe``. Running it again for the same
model replaces the old viewer, so there are no "port in use" errors.

In-memory models are exported to ONNX by default, with the graph's input and output
named ``input`` and ``output``. ``format="torchscript"`` and ``format="export"``
(``torch.export``) are also available. To keep the exported file, for example in
your run folder:

.. code-block:: python

   path = cu.tools.export_for_netron(model, run / "model.onnx", input_size=(1, 1, 28, 28))
   cu.tools.netron(path)

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
