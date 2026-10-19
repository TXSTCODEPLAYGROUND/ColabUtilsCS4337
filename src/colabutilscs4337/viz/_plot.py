"""Plotting helpers for CAMs and feature maps (matplotlib, imported lazily)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from .._compat import require

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_image(
    image: Any,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    """Convert an image to a float ``HxWx3`` array in ``[0, 1]``.

    Accepts a PIL image, an ``HxWx3`` array, or a ``(3,H,W)``/``(1,3,H,W)`` tensor.
    Pass ``mean``/``std`` (for example :data:`IMAGENET_MEAN`/:data:`IMAGENET_STD`) to
    undo normalization; otherwise values outside ``[0, 1]`` are min-max rescaled.
    """
    np = require("numpy")
    if hasattr(image, "detach"):
        t = image.detach().cpu().float()
        if t.ndim == 4:
            t = t[0]
        if mean is not None and std is not None:
            t = t * t.new_tensor(std)[:, None, None] + t.new_tensor(mean)[:, None, None]
        arr = t.permute(1, 2, 0).numpy() if t.ndim == 3 else t.numpy()
    else:
        arr = np.asarray(image)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.min() < 0 or arr.max() > 1:
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-8)
    return arr


def show_once(fig):
    """Draw ``fig`` once in a notebook and return it (for ``fig.savefig(...)``).

    With the inline backend a returned figure is drawn twice: once when the cell
    finishes and once more as the cell's value. Here the figure is drawn right away
    and closed, and its automatic display is skipped until the cell ends.
    """
    plt = require("matplotlib.pyplot")
    if "inline" not in plt.get_backend().lower():
        return fig
    try:
        from IPython import get_ipython
        from IPython.display import display
    except ImportError:
        return fig
    ip = get_ipython()
    if ip is None:
        return fig
    display(fig)
    plt.close(fig)
    fig._ipython_display_ = lambda: None

    def restore(*_):
        fig.__dict__.pop("_ipython_display_", None)
        ip.events.unregister("post_run_cell", restore)

    ip.events.register("post_run_cell", restore)
    return fig


def _resize(heatmap, size: tuple[int, int]):
    torch = require("torch")
    t = torch.as_tensor(heatmap, dtype=torch.float32)
    if tuple(t.shape[-2:]) == size:
        return t.numpy()
    t = torch.nn.functional.interpolate(
        t[None, None], size=size, mode="bilinear", align_corners=False
    )
    return t[0, 0].numpy()


def overlay(image: Any, cam: Any, alpha: float = 0.5, cmap: str = "jet", **image_kwargs):
    """Blend a heatmap onto an image; returns an ``HxWx3`` float array."""
    plt = require("matplotlib.pyplot")
    img = to_image(image, **image_kwargs)
    heat = _resize(getattr(cam, "cam", cam), img.shape[:2])
    colored = plt.get_cmap(cmap)(heat)[..., :3]
    return (1 - alpha) * img + alpha * colored


def show_cam(
    image: Any,
    cam: Any,
    *,
    alpha: float = 0.5,
    cmap: str = "jet",
    title: str | None = None,
    ax: Any = None,
    side_by_side: bool = True,
    show_color_bar: bool = True,
    **image_kwargs,
):
    """Show the image and its CAM overlay. ``cam`` is a :class:`CAMResult` or ``(H, W)`` heatmap.

    ``show_color_bar`` adds a 0-1 importance scale next to the overlay (CAMs are
    normalized to 0-1 per image). Pass ``ax`` to draw only the overlay into your own axes.

    Examples
    --------
    >>> cu.viz.show_cam(img, cam)                          # input | overlay + color bar
    >>> cu.viz.show_cam(img, cam, show_color_bar=False, side_by_side=False)
    """
    plt = require("matplotlib.pyplot")
    if title is None and hasattr(cam, "class_idx"):
        title = f"{cam.method} on {cam.layer}: class {cam.class_idx[0]} (score {cam.scores[0]:.2f})"
    blended = overlay(image, cam, alpha=alpha, cmap=cmap, **image_kwargs)

    def color_bar(target):
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
        target.figure.colorbar(mappable, ax=target, fraction=0.046, pad=0.04, label="importance")

    if ax is not None:
        ax.imshow(blended)
        ax.set_title(title or "")
        ax.axis("off")
        if show_color_bar:
            color_bar(ax)
        return ax
    ncols = 2 if side_by_side else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5), squeeze=False)
    if side_by_side:
        axes[0, 0].imshow(to_image(image, **image_kwargs))
        axes[0, 0].set_title("input")
        axes[0, 0].axis("off")
    axes[0, -1].imshow(blended)
    axes[0, -1].set_title(title or "")
    axes[0, -1].axis("off")
    if show_color_bar:
        color_bar(axes[0, -1])
    fig.tight_layout()
    return show_once(fig)


def _as_index_list(target_indices: Any) -> list[int]:
    torch = require("torch")
    if torch.is_tensor(target_indices):
        values = target_indices.detach().cpu().flatten().tolist()
    elif isinstance(target_indices, int):
        values = [target_indices]
    else:
        values = list(target_indices)
    if not values:
        raise ValueError("target_indices is empty.")
    return [int(v) for v in values]


def visualize_gradcam(
    model: Any,
    image: Any,
    target_indices: Any,
    target_layers: Any,
    *,
    preprocess: Any = None,
    categories: Any = None,
    method: str = "gradcam",
    alpha: float = 0.5,
    cmap: str = "jet",
    show_color_bar: bool = True,
    size: float = 3.2,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
):
    """Grad-CAM for several classes and layers: one row per class, one column per layer.

    A single class or a single layer is drawn as one row. Each title is the layer
    name, the category name, and that class's probability. The color bar is a 0-1
    importance scale, the same as :func:`show_cam`.

    Parameters
    ----------
    model
        An ``nn.Module`` or a fastai ``Learner``.
    image
        A PIL image, or a tensor ``(C, H, W)`` / ``(1, C, H, W)``.
    target_indices
        Class indices to explain (a list or a 1-D tensor).
    target_layers
        Modules or dotted names, for example ``[model.layer1[-1], model.layer4[-1]]``.
    preprocess
        Transform applied to a PIL image. Ignored when ``image`` is a tensor.
    categories
        Class names, a list or ``{index: name}``, used in the titles.
    method
        ``"gradcam"`` (built in) or a pytorch-grad-cam method such as ``"gradcam++"``
        (needs ``cu.install("cam")``).
    mean, std
        Undo normalization for the picture under the heatmap. When omitted, ImageNet
        statistics are used if the tensor is outside ``[0, 1]``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> cu.viz.visualize_gradcam(
    ...     model, img, target_indices=[281, 282],
    ...     target_layers=[model.layer1[-1], model.layer4[-1]],
    ...     preprocess=preprocess, categories=categories,
    ... )
    """
    from ._cam import gradcam
    from ._classify import _label, class_probabilities, model_input
    from ._hooks import resolve_layer

    plt = require("matplotlib.pyplot")
    indices = _as_index_list(target_indices)
    if isinstance(target_layers, (list, tuple)):
        layers = list(target_layers)
    else:
        layers = [target_layers]
    if not layers:
        raise ValueError("target_layers is empty.")

    batch = model_input(image, preprocess)
    probs = class_probabilities(model, batch)
    if len(indices) > 1:
        batch = batch.expand(len(indices), *batch.shape[1:]).contiguous()
    shown = batch[0]
    if mean is None and std is None and (float(shown.min()) < 0 or float(shown.max()) > 1):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    base = to_image(shown, mean, std)

    names = [resolve_layer(model, layer)[0] for layer in layers]
    heatmaps = []
    for layer in layers:
        result = gradcam(model, batch, target_layer=layer, class_idx=indices, method=method)
        heatmaps.append(result.cams)

    one_row = len(indices) == 1 or len(layers) == 1
    cells = [(r, c) for r in range(len(indices)) for c in range(len(layers))]
    if one_row:
        nrows, ncols = 1, len(cells)
        grid = [cells]
    else:
        nrows, ncols = len(indices), len(layers)
        grid = [cells[r * ncols : (r + 1) * ncols] for r in range(nrows)]

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(size * ncols, size * nrows + 0.45 * nrows),
        squeeze=False,
        layout="constrained",
    )
    for r, row in enumerate(grid):
        for c, (cls_i, layer_i) in enumerate(row):
            ax = axes[r, c]
            ax.imshow(overlay(base, heatmaps[layer_i][cls_i], alpha=alpha, cmap=cmap))
            label = _label(categories, indices[cls_i])
            prob = float(probs[indices[cls_i]])
            ax.set_title(f"{names[layer_i]}\n{label}  {prob:.3f}", fontsize=8)
            ax.axis("off")
    if show_color_bar:
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
        fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.04, label="importance")
    return show_once(fig)


def show_feature_maps(
    act: Any,
    n: int = 16,
    *,
    ncols: int = 8,
    sample: int = 0,
    cmap: str = "viridis",
    title: str | None = None,
):
    """Grid of the first ``n`` channels of an activation ``(B, C, H, W)`` or ``(C, H, W)``."""
    plt = require("matplotlib.pyplot")
    t = act.detach().cpu().float() if hasattr(act, "detach") else require("torch").as_tensor(act)
    if t.ndim == 4:
        t = t[sample]
    if t.ndim != 3:
        raise ValueError(f"Expected a (C, H, W) feature map, got shape {tuple(t.shape)}")
    n = min(n, t.shape[0])
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.6 * ncols, 1.6 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(t[i].numpy(), cmap=cmap)
            ax.set_title(str(i), fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return show_once(fig)
