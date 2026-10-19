"""Classification probabilities and category-name search."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from .._compat import require
from ..tools._summary import unwrap_model
from ._hooks import _device_of, eval_mode, plain_tensor


def model_input(image: Any, preprocess: Any = None):
    """Turn a PIL image or tensor into a ``(1, C, H, W)`` batch.

    ``preprocess`` (a torchvision ``Compose``, for example) runs only on non-tensors,
    so a tensor you already prepared is left alone.
    """
    torch = require("torch")
    if not torch.is_tensor(image):
        if preprocess is None:
            raise ValueError("Pass preprocess= to turn a PIL image into a model input.")
        image = preprocess(image)
    tensor = plain_tensor(image) if torch.is_tensor(image) else torch.as_tensor(image)
    if tensor.ndim == 3:
        tensor = tensor[None]
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected an image of shape (C, H, W) or (1, C, H, W), got {tuple(tensor.shape)}"
        )
    return tensor


def class_probabilities(model: Any, image: Any, preprocess: Any = None):
    """Softmax probabilities for one image, shape ``(num_classes,)`` on CPU."""
    torch = require("torch")
    module = unwrap_model(model)
    x = model_input(image, preprocess).to(_device_of(module))
    with torch.no_grad(), eval_mode(module):
        logits = module(plain_tensor(x))
    return torch.softmax(logits, dim=1)[0].detach().cpu()


def _label(categories: Any, index: int) -> str:
    if categories is None:
        return str(index)
    try:
        return str(categories[index])
    except (IndexError, KeyError, TypeError):
        return str(index)


def get_classification_topk(
    model: Any,
    image: Any,
    categories: Any = None,
    k: int | None = 10,
    *,
    preprocess: Any = None,
):
    """Top classes for one image, as a DataFrame sorted by probability.

    Parameters
    ----------
    model
        An ``nn.Module`` or a fastai ``Learner``.
    image
        A PIL image, or a tensor ``(C, H, W)`` / ``(1, C, H, W)`` already prepared
        for the model.
    categories
        Class names: a list (``categories[i]`` is class ``i``) or a dict
        ``{index: name}``. Missing names fall back to the index.
    k
        How many rows to return. ``None`` returns every class.
    preprocess
        A transform (for example a ``transforms.Compose``) applied to a PIL image.
        Ignored when ``image`` is already a tensor.

    Returns
    -------
    pandas.DataFrame
        Columns ``Index`` (class index), ``label``, and ``probability``, highest
        probability first.

    Examples
    --------
    >>> df = cu.viz.get_classification_topk(model, img, categories, k=10, preprocess=preprocess)
    >>> df.head()
    """
    pd = require("pandas", purpose="the classification table")
    if k is not None and k < 1:
        raise ValueError(f"k must be at least 1, or None for every class; got {k}")
    prob = class_probabilities(model, image, preprocess)
    order = prob.argsort(descending=True)
    if k is not None:
        order = order[:k]
    rows = [
        {"Index": int(i), "label": _label(categories, int(i)), "probability": float(prob[i])}
        for i in order.tolist()
    ]
    return pd.DataFrame(rows)


def find_categories_like(categories: Any, class_like: str) -> list[tuple[str, int]]:
    """Category names matching a regular expression, as ``(name, index)`` pairs.

    ``categories`` is a list (position is the class index) or a dict
    ``{index: name}``. ``class_like`` is a regular expression: ``r"\\bcat\\b"``
    matches the word "cat" and not "catch".

    Examples
    --------
    >>> cu.viz.find_categories_like(categories, r"\\bcat\\b")
    [("tabby cat", 281), ("tiger cat", 282)]
    """
    pattern = re.compile(class_like)
    if isinstance(categories, Mapping):
        items = ((int(idx), str(name)) for idx, name in categories.items())
    else:
        items = ((idx, str(name)) for idx, name in enumerate(categories))
    return [(name, idx) for idx, name in items if pattern.search(name)]
