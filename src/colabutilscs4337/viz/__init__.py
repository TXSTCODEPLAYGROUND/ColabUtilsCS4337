"""Visualize image data and CNNs: sample grids, Grad-CAM, activations, feature maps."""

from typing import Any

from ._cam import BACKEND_METHODS, CAMResult, gradcam, suggest_target_layer
from ._classify import find_categories_like, get_classification_topk
from ._data import show_by_class, show_images, tensor_to_image
from ._hooks import activations, layer_names, resolve_layer
from ._plot import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    overlay,
    show_cam,
    show_feature_maps,
    to_image,
    visualize_gradcam,
)


def learner_input(learn: Any, item: Any):
    """Turn a raw item (PIL image, file path) into a model-ready batch via Learner transforms."""
    batch = learn.dls.test_dl([item]).one_batch()
    return batch[0] if isinstance(batch, (tuple, list)) else batch


__all__ = [
    "BACKEND_METHODS",
    "CAMResult",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "activations",
    "find_categories_like",
    "get_classification_topk",
    "gradcam",
    "layer_names",
    "learner_input",
    "overlay",
    "resolve_layer",
    "show_by_class",
    "show_cam",
    "show_feature_maps",
    "show_images",
    "suggest_target_layer",
    "tensor_to_image",
    "to_image",
    "visualize_gradcam",
]
