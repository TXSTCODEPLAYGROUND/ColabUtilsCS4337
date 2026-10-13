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
    **image_kwargs,
):
    """Show the image and its CAM overlay. ``cam`` is a :class:`CAMResult` or ``(H, W)`` heatmap."""
    plt = require("matplotlib.pyplot")
    if title is None and hasattr(cam, "class_idx"):
        title = f"{cam.method} on {cam.layer}: class {cam.class_idx[0]} (score {cam.scores[0]:.2f})"
    blended = overlay(image, cam, alpha=alpha, cmap=cmap, **image_kwargs)

    if ax is not None:
        ax.imshow(blended)
        ax.set_title(title or "")
        ax.axis("off")
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
    fig.tight_layout()
    return fig


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
    return fig
