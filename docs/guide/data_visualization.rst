Looking at image data
=====================

Always look at your data before training. These functions take **whatever you have**:
a ``Dataset``, a ``DataLoader``, an ``(x, y)`` batch, a tensor plus labels, or a list
of PIL images. Class names are read from ``dataset.classes`` when it exists.

The examples use FashionMNIST:

.. code-block:: python

   import torch, torchvision
   from torchvision import transforms
   import colabutilscs4337 as cu

   train_ds = torchvision.datasets.FashionMNIST("data", train=True, download=True,
                                                transform=transforms.ToTensor())
   train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

One row per class
-----------------

.. code-block:: python

   cu.viz.show_by_class(train_ds)                    # 8 random examples of each class
   cu.viz.show_by_class(train_ds, n_per_class=12)

   x, y = next(iter(train_loader))
   labels_map = {0: "T-Shirt", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
                 5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}
   cu.viz.show_by_class(x, y, labels_map, n_per_class=5)

The class names go on the left. A class with no examples in the batch gets an empty
row, which is a quick way to spot an unbalanced batch.

A grid of samples
-----------------

.. code-block:: python

   cu.viz.show_images(train_ds)                       # 32 random images
   cu.viz.show_images(train_loader, n=16)             # first batch of a loader
   cu.viz.show_images(x, y, train_ds.classes, ncols=4)
   cu.viz.show_images(train_ds, seed=0)               # the same sample every time

Predictions vs. labels
----------------------

Pass ``preds=`` and every title becomes ``prediction / truth``, green when correct
and red when wrong:

.. code-block:: python

   model.eval()
   with torch.no_grad():
       preds = model(x).argmax(dim=1)
   cu.viz.show_images(x, y, train_ds.classes, preds=preds)

Images as a PIL strip
---------------------

:func:`~colabutilscs4337.viz.tensor_to_image` joins images into one
``PIL.Image``, handy for saving or logging:

.. code-block:: python

   cu.viz.tensor_to_image(x[:8])                # side by side (a row)
   cu.viz.tensor_to_image(x[:8], dim=1)         # top to bottom (a column)
   cu.viz.tensor_to_image(x[:8], pad=2)         # with white gaps
   cu.viz.tensor_to_image(img_a, img_b)         # several tensors at once
   cu.viz.tensor_to_image(x[:8]).save(run / "samples.png")

Normalized images
-----------------

If your transform includes ``Normalize`` (for example the ImageNet statistics used
with pretrained models), images are rescaled to look right automatically. To undo the
normalization exactly, pass the same statistics:

.. code-block:: python

   cu.viz.show_images(x, y, classes, mean=cu.viz.IMAGENET_MEAN, std=cu.viz.IMAGENET_STD)

fastai
------

With fastai ``DataLoaders``, ``cu.viz.show_images(dls)`` simply calls fastai's own
``dls.show_batch()``. For a single image prepared the way the model sees it, use
:func:`~colabutilscs4337.viz.learner_input`.
