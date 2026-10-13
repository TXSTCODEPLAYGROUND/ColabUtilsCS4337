Quick start
===========

A typical assignment notebook, top to bottom. Every step is one line.

1. Set up
---------

.. code-block:: python

   %pip install -q --upgrade colabutilscs4337
   import colabutilscs4337 as cu

   cu.env.check()                         # versions vs. the tested Colab runtime
   cu.env.load_secrets("WANDB_API_KEY")   # only if you use W&B (see :doc:`guide/secrets`)

2. Get the data and look at it
------------------------------

.. code-block:: python

   import torch, torchvision
   from torchvision import transforms

   data_dir = cu.drive.data_dir()          # /content/data: fast local disk
   train_ds = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True,
                                                transform=transforms.ToTensor())
   train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

   cu.viz.show_by_class(train_ds)          # one row per class
   cu.viz.show_images(train_loader)        # one batch, titled with class names

3. Build and inspect the model
------------------------------

.. code-block:: python

   model = torchvision.models.resnet18(num_classes=10)
   model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)   # grayscale input

   cu.tools.summary(model, input_size=(1, 1, 28, 28))
   cu.tools.netron(model, input_size=(1, 1, 28, 28))           # interactive graph

4. Train, saving into a run folder on Drive
-------------------------------------------

.. code-block:: python

   run = cu.drive.new_run("resnet18-lr1e-3")   # mounts Drive, creates the folder
   # ... your training loop ...
   torch.save(model.state_dict(), run / "model.pth")

5. Check the predictions and explain them
-----------------------------------------

.. code-block:: python

   model = cu.models.load_model(run / "model.pth", arch=model)   # e.g. in a new session
   x, y = next(iter(train_loader))
   preds = model(x.to(next(model.parameters()).device)).argmax(1).cpu()
   cu.viz.show_images(x, y, train_ds.classes, preds=preds)     # wrong ones in red

   cam = cu.viz.gradcam(model, x[:1])
   cu.viz.show_cam(x[0], cam)
