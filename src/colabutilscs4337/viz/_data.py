"""Look at image data: grids of samples, one row per class, tensors to PIL images.

Every function accepts whatever you have at hand::

    cu.viz.show_images(train_ds)                    # a torchvision / custom Dataset
    cu.viz.show_images(train_loader)                # a DataLoader (shows one batch)
    cu.viz.show_images(x, y, classes)               # a batch tensor plus labels
    cu.viz.show_by_class(train_ds, n_per_class=8)   # one row per class
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from typing import Any

from .._compat import require
from ._plot import show_once

Classes = Sequence[str] | Mapping[int, str] | None


def _as_chw(img: Any, mean=None, std=None):
    """One image (PIL, numpy HxW/HxWxC, tensor HxW/CxHxW) -> float tensor CxHxW in [0, 1]."""
    torch = require("torch")
    if not torch.is_tensor(img):
        np = require("numpy")
        arr = np.asarray(img)
        t = torch.from_numpy(arr.copy())
        if t.ndim == 3 and t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
            t = t.permute(2, 0, 1)
    else:
        t = img.detach().cpu()
        t = t.as_subclass(torch.Tensor) if type(t) is not torch.Tensor else t
    if t.ndim == 2:
        t = t[None]
    if t.ndim != 3:
        raise ValueError(f"Expected a single image, got shape {tuple(t.shape)}")
    t = t.float() / 255.0 if t.dtype == torch.uint8 else t.float()
    if mean is not None and std is not None:
        c = t.shape[0]
        t = t * torch.tensor(std[:c])[:, None, None] + torch.tensor(mean[:c])[:, None, None]
    if t.min() < 0 or t.max() > 1:
        t = (t - t.min()) / (t.max() - t.min()).clamp_min(1e-8)
    return t


def _match_shapes(images: list) -> list:
    """Make images stackable: same channel count (gray -> RGB if mixed) and same H x W."""
    torch = require("torch")
    if not images:
        return images
    channels = max(t.shape[0] for t in images)
    size = tuple(images[0].shape[-2:])
    out = []
    for t in images:
        if t.shape[0] != channels:
            t = t[:1].expand(channels, -1, -1)
        if tuple(t.shape[-2:]) != size:
            t = torch.nn.functional.interpolate(
                t[None], size=size, mode="bilinear", align_corners=False
            )[0]
        out.append(t)
    return out


def tensor_to_image(*imgtensor: Any, dim: int = 2, pad: int = 0, mean=None, std=None):
    """Turn one or more image tensors into a single ``PIL.Image`` strip.

    Parameters
    ----------
    *imgtensor : torch.Tensor
        One or more tensors, each ``C x H x W`` or a batch ``B x C x H x W``. Values in
        ``[0, 1]``, ``uint8`` in ``[0, 255]``, or normalized (pass ``mean``/``std``, or
        values are min-max rescaled).
    dim : int
        How images are joined: ``2`` (default) puts them **side by side** (a row),
        ``1`` **stacks** them top to bottom (a column).
    pad : int
        White pixels between images.
    mean, std : sequence of float, optional
        Undo ``Normalize(mean, std)``, e.g. ``cu.viz.IMAGENET_MEAN``/``IMAGENET_STD``.

    Returns
    -------
    PIL.Image.Image
        Grayscale (``"L"``) for 1-channel input, otherwise RGB.

    Examples
    --------
    >>> cu.viz.tensor_to_image(x[:8])            # first 8 images of a batch, in a row
    >>> cu.viz.tensor_to_image(x[:8], dim=1)     # ... as a column
    >>> cu.viz.tensor_to_image(img_a, img_b)     # two images side by side
    """
    torch = require("torch")
    pil = require("PIL.Image")
    if dim not in (1, 2):
        raise ValueError("dim must be 2 (side by side) or 1 (stacked top to bottom)")
    images = []
    for t in imgtensor:
        t = torch.as_tensor(t) if not torch.is_tensor(t) else t
        batch = t if t.ndim == 4 else t[None]
        images.extend(_as_chw(img, mean, std) for img in batch)
    images = _match_shapes(images)
    if pad and len(images) > 1:
        c, h, w = images[0].shape
        gap = torch.ones(c, h, pad) if dim == 2 else torch.ones(c, pad, w)
        spaced = []
        for i, img in enumerate(images):
            spaced += [img, gap] if i < len(images) - 1 else [img]
        images = spaced
    strip = torch.cat(images, dim=dim)
    arr = (strip.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).numpy()
    return pil.fromarray(arr[..., 0] if arr.shape[-1] == 1 else arr[..., :3])


def _labels_to_list(y: Any) -> list[int] | None:
    if y is None:
        return None
    torch = require("torch")
    t = torch.as_tensor(y)
    if t.ndim == 2 and t.shape[1] > 1:  # one-hot / scores
        t = t.argmax(dim=1)
    return [int(v) for v in t.flatten().tolist()]


def _class_name(classes: Classes, idx: int) -> str:
    if classes is None:
        return str(idx)
    try:
        return str(classes[idx])
    except (IndexError, KeyError):
        return str(idx)


def _split_item(item: Any) -> tuple[Any, int | None]:
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        return item[0], int(require("torch").as_tensor(item[1]).flatten()[0])
    return item, None


def _is_loader(source: Any) -> bool:
    return hasattr(source, "dataset") and hasattr(source, "batch_size")


def _samples(source: Any, labels: Any, shuffle: bool, seed: int | None):
    """Normalize any input into ``(iterator of (image, label or None), classes or None)``.

    Lazy, so callers can stop early without loading a whole dataset.
    """
    torch = require("torch")
    classes = getattr(source, "classes", None)

    is_xy_batch = (
        isinstance(source, (tuple, list))
        and len(source) == 2
        and torch.is_tensor(source[0])
        and source[0].ndim == 4
    )
    if is_xy_batch:
        source, labels = source

    if torch.is_tensor(source):
        batch = source if source.ndim == 4 else source[None]
        ys = _labels_to_list(labels) or [None] * len(batch)
        return zip(batch, ys, strict=False), classes

    if _is_loader(source):
        classes = getattr(source.dataset, "classes", classes)

        def from_loader():
            for batch in source:
                x, y = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (batch, None)
                yield from zip(x, _labels_to_list(y) or [None] * len(x), strict=False)

        return from_loader(), classes

    if hasattr(source, "__getitem__") and hasattr(source, "__len__"):  # Dataset or list
        indices = list(range(len(source)))
        if shuffle:
            random.Random(seed).shuffle(indices)
        given = _labels_to_list(labels)

        def from_dataset():
            for i in indices:
                img, y = _split_item(source[i])
                yield img, (given[i] if given is not None else y)

        return from_dataset(), classes

    raise TypeError(f"Don't know how to show a {type(source).__name__}")


def _take(it, n: int) -> tuple[list, list[int] | None]:
    imgs, ys = [], []
    for img, y in it:
        imgs.append(img)
        ys.append(y)
        if len(imgs) >= n:
            break
    return imgs, (ys if any(y is not None for y in ys) else None)


def show_images(
    images: Any,
    labels: Any = None,
    classes: Classes = None,
    *,
    preds: Any = None,
    n: int = 32,
    ncols: int = 8,
    size: float = 1.6,
    shuffle: bool = True,
    seed: int | None = None,
    mean=None,
    std=None,
    title: str | None = None,
):
    """Show images in a grid, with class names (and predictions) as titles.

    Parameters
    ----------
    images
        A batch tensor ``B x C x H x W``, an ``(x, y)`` batch, a ``Dataset`` (samples
        ``n`` random items), a ``DataLoader`` (shows its first batch), a list of PIL
        images/arrays/tensors, or a fastai ``DataLoaders`` (uses its ``show_batch``).
    labels
        Class indices, shape ``(B,)`` or ``(B, 1)``. Not needed for datasets/loaders.
    classes
        Class names: a list (``train_ds.classes``) or a dict ``{0: "T-Shirt", ...}``.
        Taken from ``dataset.classes`` automatically when available.
    preds
        Predicted class indices. Titles become ``pred / true`` (red when wrong).
    n, ncols, size
        Number of images, images per row, and inches per image.
    shuffle, seed
        Random sample from a Dataset (``seed`` makes it repeatable).
    mean, std
        Undo normalization, e.g. ``mean=cu.viz.IMAGENET_MEAN, std=cu.viz.IMAGENET_STD``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> train_ds = torchvision.datasets.FashionMNIST("data", download=True,
    ...                                              transform=transforms.ToTensor())
    >>> cu.viz.show_images(train_ds)                          # 32 random samples
    >>> x, y = next(iter(train_loader))
    >>> cu.viz.show_images(x, y, train_ds.classes, n=16)
    >>> cu.viz.show_images(x, y, train_ds.classes, preds=model(x).argmax(1))
    """
    if hasattr(images, "show_batch") and hasattr(images, "one_batch"):
        return images.show_batch(max_n=n)
    plt = require("matplotlib.pyplot")
    it, auto_classes = _samples(images, labels, shuffle and not _is_loader(images), seed)
    imgs, ys = _take(it, n)
    classes = classes if classes is not None else auto_classes
    ps = _labels_to_list(preds)
    tensors = [_as_chw(img, mean, std) for img in imgs]

    count = len(tensors)
    ncols = max(1, min(ncols, count))
    nrows = max(1, -(-count // ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(size * ncols, size * nrows + 0.3 * nrows), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= count:
            continue
        t = tensors[i]
        if t.shape[0] == 1:
            ax.imshow(t[0].numpy(), cmap="gray", vmin=0, vmax=1)
        else:
            ax.imshow(t[:3].permute(1, 2, 0).numpy())
        true = _class_name(classes, ys[i]) if ys and ys[i] is not None else None
        if ps is not None and i < len(ps):
            pred = _class_name(classes, ps[i])
            wrong = ys is not None and ys[i] is not None and ps[i] != ys[i]
            ax.set_title(
                f"{pred} / {true}" if true else pred, fontsize=8, color="red" if wrong else "green"
            )
        elif true is not None:
            ax.set_title(true, fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return show_once(fig)


def show_by_class(
    images: Any,
    labels: Any = None,
    classes: Classes = None,
    *,
    n_per_class: int = 8,
    size: float = 1.0,
    max_scan: int = 5000,
    seed: int | None = None,
    mean=None,
    std=None,
    pad: int = 2,
    title: str | None = None,
):
    """One row per class with ``n_per_class`` examples each; class names on the left.

    Parameters
    ----------
    images
        A batch tensor ``B x C x H x W`` with ``labels``, an ``(x, y)`` batch, a
        ``DataLoader``, or a ``Dataset`` (up to ``max_scan`` random items are scanned
        to fill every row).
    labels
        Class indices ``(B,)`` or ``(B, 1)`` when ``images`` is a tensor.
    classes
        Class names (list or dict). Defaults to ``dataset.classes`` or the label numbers.
        Classes with no examples get an empty row.
    n_per_class, size
        Examples per row and inches per image.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> cu.viz.show_by_class(train_ds)                     # every class, 8 examples each
    >>> cu.viz.show_by_class(x, y, labels_map, n_per_class=5)
    """
    torch = require("torch")
    plt = require("matplotlib.pyplot")

    it, auto_classes = _samples(images, labels, shuffle=not torch.is_tensor(images), seed=seed)
    classes = classes if classes is not None else auto_classes

    groups: dict[int, list] = {}
    if isinstance(classes, Mapping):
        groups = {c: [] for c in sorted(classes)}
    elif classes is not None:
        groups = {c: [] for c in range(len(classes))}
    known = bool(groups)

    for scanned, (img, y) in enumerate(it):
        if y is None:
            raise ValueError("show_by_class needs labels: pass `labels=` or a labeled dataset.")
        if not known:
            groups.setdefault(y, [])
        if y in groups and len(groups[y]) < n_per_class:
            groups[y].append(_as_chw(img, mean, std))
        if known and all(len(g) >= n_per_class for g in groups.values()):
            break
        if scanned + 1 >= max_scan:
            break
    if not known:
        groups = {c: groups.get(c, []) for c in range(max(groups, default=-1) + 1)}
    class_ids = list(groups)

    sample = next((g[0] for g in groups.values() if g), None)
    if sample is None:
        raise ValueError("No images found for any class.")
    blank = torch.ones_like(sample)

    rows = len(class_ids)
    fig, axes = plt.subplots(
        rows, 1, figsize=(size * n_per_class + 1.5, size * rows), squeeze=False
    )
    for ax, c in zip(axes[:, 0], class_ids, strict=True):
        row = groups[c] or [blank]
        strip = tensor_to_image(torch.stack(_match_shapes([sample, *row])[1:]), pad=pad)
        ax.imshow(strip, cmap="gray" if strip.mode == "L" else None, vmin=0, vmax=255)
        ax.set_anchor("W")
        ax.set_ylabel(_class_name(classes, c), rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return show_once(fig)
