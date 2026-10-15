Explaining a CNN: Grad-CAM and activations
==========================================

Grad-CAM
--------

Grad-CAM highlights the parts of the image that pushed the model toward a class.

.. code-block:: python

   cam = cu.viz.gradcam(model, x)             # x: (1, 3, H, W), as fed to the model
   cu.viz.show_cam(x, cam)                    # input and heatmap overlay side by side

   cam.class_idx, cam.scores                  # which class was explained, its score
   cam.layer                                  # which layer was used, e.g. "layer4"

* **Which class?** The predicted one by default; ask for another with
  ``class_idx=3``. For a batch, pass one index per image.
* **Which layer?** The last layer that still has a spatial map, found automatically
  (``layer4`` for torchvision ResNets, the body for fastai ``vision_learner``).
  Choose another with ``target_layer="layer3"``.
* **Normalized input?** Pass ``mean=cu.viz.IMAGENET_MEAN, std=cu.viz.IMAGENET_STD``
  to ``show_cam`` so the image underneath looks natural.
* **Reading the colors.** A color bar next to the overlay maps colors to importance
  from 0 (ignored) to 1 (most important); the map is scaled per image. Hide it with
  ``show_color_bar=False``.

With fastai:

.. code-block:: python

   x = cu.viz.learner_input(learn, "cat.jpg")       # same transforms as training
   cam = cu.viz.gradcam(learn, x)
   cu.viz.show_cam(x, cam, mean=cu.viz.IMAGENET_MEAN, std=cu.viz.IMAGENET_STD)

Other CAM methods
-----------------

After ``cu.install("cam")``, choose a method from
`pytorch-grad-cam <https://github.com/jacobgil/pytorch-grad-cam>`_:

.. code-block:: python

   cu.viz.gradcam(model, x, method="gradcam++")
   # also: "xgradcam", "hirescam", "layercam", "scorecam", "eigencam", "ablationcam"

Layer activations
-----------------

.. code-block:: python

   cu.viz.layer_names(model, torch.nn.Conv2d)              # what can I look at?
   acts = cu.viz.activations(model, x, layers=["layer1", "layer3"])
   acts["layer1"].shape                                    # (1, 64, 56, 56)
   cu.viz.show_feature_maps(acts["layer1"], n=16)          # first 16 channels

Early layers respond to edges and colours; later layers to textures and parts.

None of these functions change your model: training mode, BatchNorm statistics, and
parameter gradients are left exactly as they were.
