``cu.viz``: data and CNN visualization
======================================

.. automodule:: colabutilscs4337.viz
   :no-members:

Image data
----------

.. autofunction:: colabutilscs4337.viz.show_by_class
.. autofunction:: colabutilscs4337.viz.show_images
.. autofunction:: colabutilscs4337.viz.tensor_to_image
.. autofunction:: colabutilscs4337.viz.to_image

Grad-CAM
--------

.. autofunction:: colabutilscs4337.viz.gradcam
.. autoclass:: colabutilscs4337.viz.CAMResult
   :members: cam
.. autofunction:: colabutilscs4337.viz.show_cam
.. autofunction:: colabutilscs4337.viz.visualize_gradcam
.. autofunction:: colabutilscs4337.viz.get_classification_topk
.. autofunction:: colabutilscs4337.viz.find_categories_like
.. autofunction:: colabutilscs4337.viz.overlay
.. autofunction:: colabutilscs4337.viz.suggest_target_layer
.. autofunction:: colabutilscs4337.viz.learner_input

Activations
-----------

.. autofunction:: colabutilscs4337.viz.activations
.. autofunction:: colabutilscs4337.viz.show_feature_maps
.. autofunction:: colabutilscs4337.viz.layer_names
.. autofunction:: colabutilscs4337.viz.resolve_layer

Constants
---------

.. autodata:: colabutilscs4337.viz.IMAGENET_MEAN
.. autodata:: colabutilscs4337.viz.IMAGENET_STD
