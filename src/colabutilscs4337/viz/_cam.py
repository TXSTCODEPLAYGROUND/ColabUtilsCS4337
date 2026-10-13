"""Class activation maps: a small Grad-CAM implementation plus pytorch-grad-cam methods."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .._compat import require
from ..tools._summary import unwrap_model
from ._hooks import LayerRef, _device_of, eval_mode, plain_tensor, resolve_layer

# method name -> pytorch_grad_cam class name
BACKEND_METHODS = {
    "gradcam++": "GradCAMPlusPlus",
    "xgradcam": "XGradCAM",
    "hirescam": "HiResCAM",
    "layercam": "LayerCAM",
    "scorecam": "ScoreCAM",
    "eigencam": "EigenCAM",
    "ablationcam": "AblationCAM",
}

_POOLING = (
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "AdaptiveConcatPool2d",
    "AvgPool2d",
    "MaxPool2d",
)


def suggest_target_layer(model: Any, x: Any | None = None) -> str:
    """Name of a good Grad-CAM layer: the last module with a spatial (H, W > 1) output.

    With an example input ``x`` this is found by tracing a forward pass (works for
    torchvision ResNets -> ``"layer4"``, fastai ``vision_learner`` -> ``"0"``, timm,
    and custom CNNs). Without ``x`` the last ``nn.Conv2d`` is returned.
    """
    torch = require("torch")
    module = unwrap_model(model)
    if x is None:
        convs = [n for n, m in module.named_modules() if isinstance(m, torch.nn.Conv2d)]
        if not convs:
            raise ValueError("Model has no Conv2d layers; pass target_layer explicitly.")
        return convs[-1]

    order: list[str] = []

    def make_hook(name: str):
        def hook(mod, _inp, out):
            if (
                torch.is_tensor(out)
                and out.ndim == 4
                and out.shape[-1] * out.shape[-2] > 1
                and type(mod).__name__ not in _POOLING
            ):
                order.append(name)

        return hook

    handles = [
        sub.register_forward_hook(make_hook(name)) for name, sub in module.named_modules() if name
    ]
    try:
        with torch.no_grad(), eval_mode(module):
            module(plain_tensor(x)[:1].to(_device_of(module)))
    finally:
        for handle in handles:
            handle.remove()
    if not order:
        raise ValueError("No layer produced a spatial feature map; pass target_layer explicitly.")
    return order[-1]


@dataclass
class CAMResult:
    """Grad-CAM output. ``cams`` is ``(B, H, W)`` in ``[0, 1]`` at the input resolution."""

    cams: Any
    class_idx: list[int]
    scores: list[float]
    layer: str
    method: str

    @property
    def cam(self):
        """The first (often only) heatmap, shape ``(H, W)``."""
        return self.cams[0]


def _class_indices(logits, class_idx: int | Sequence[int] | None) -> list[int]:
    batch = logits.shape[0]
    if class_idx is None:
        return logits.argmax(dim=1).tolist()
    if isinstance(class_idx, int):
        return [class_idx] * batch
    idx = list(class_idx)
    if len(idx) != batch:
        raise ValueError(f"class_idx has {len(idx)} entries for a batch of {batch}")
    return idx


def _normalize(cams):
    flat = cams.flatten(1)
    lo = flat.min(dim=1).values[:, None, None]
    hi = flat.max(dim=1).values[:, None, None]
    return (cams - lo) / (hi - lo).clamp_min(1e-8)


def _gradcam(module, x, layer_name: str, class_idx):
    torch = require("torch")
    F = torch.nn.functional
    _, layer = resolve_layer(module, layer_name)
    store: dict[str, Any] = {}

    def forward_hook(_mod, _inp, out):
        store["acts"] = out

    handle = layer.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad(), eval_mode(module):
            inp = x.to(_device_of(module)).clone().requires_grad_(True)
            logits = module(inp)
            idx = _class_indices(logits, class_idx)
            picked = logits.gather(1, torch.tensor(idx, device=logits.device)[:, None])
            # autograd.grad (not .backward) leaves the model's parameter .grad untouched.
            (grads,) = torch.autograd.grad(picked.sum(), store["acts"])
    finally:
        handle.remove()

    acts, grads = store["acts"].detach(), grads.detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cams = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cams = F.interpolate(cams, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return _normalize(cams).cpu(), idx, picked.detach().flatten().cpu().tolist()


def _backend_cam(module, x, layer_name: str, class_idx, method: str):
    torch = require("torch")
    pgc = require("pytorch_grad_cam", extra="cam", purpose=f"method={method!r}")
    targets_mod = require("pytorch_grad_cam.utils.model_targets", extra="cam")
    _, layer = resolve_layer(module, layer_name)
    cam_cls = getattr(pgc, BACKEND_METHODS[method])

    x = x.to(_device_of(module))
    with eval_mode(module):
        with torch.no_grad():
            logits = module(x)
        idx = _class_indices(logits, class_idx)
        picked = logits.gather(1, torch.tensor(idx, device=logits.device)[:, None])
        scores = picked.flatten().tolist()
        targets = [targets_mod.ClassifierOutputTarget(i) for i in idx]
        with cam_cls(model=module, target_layers=[layer]) as cam:
            grayscale = cam(input_tensor=x, targets=targets)
    return torch.from_numpy(grayscale).float(), idx, scores


def gradcam(
    model: Any,
    x: Any,
    target_layer: LayerRef | None = None,
    class_idx: int | Sequence[int] | None = None,
    *,
    method: str = "gradcam",
) -> CAMResult:
    """Class activation map for a batch ``x`` of shape ``(B, 3, H, W)``.

    ``class_idx`` defaults to the predicted class. ``target_layer`` is a dotted name
    (``"layer4"``) or module; it defaults to :func:`suggest_target_layer`. ``method``
    is ``"gradcam"`` (built in) or one of ``gradcam++``, ``xgradcam``, ``hirescam``,
    ``layercam``, ``scorecam``, ``eigencam``, ``ablationcam`` (needs ``cu.install("cam")``).
    """
    module = unwrap_model(model)
    x = plain_tensor(x)
    if x.ndim == 3:
        x = x[None]
    if target_layer is None:
        layer_name = suggest_target_layer(module, x)
    else:
        layer_name, _ = resolve_layer(module, target_layer)

    method = method.lower()
    if method == "gradcam":
        cams, idx, scores = _gradcam(module, x, layer_name, class_idx)
    elif method in BACKEND_METHODS:
        cams, idx, scores = _backend_cam(module, x, layer_name, class_idx, method)
    else:
        raise ValueError(f"Unknown method {method!r}; choose gradcam or {sorted(BACKEND_METHODS)}")
    return CAMResult(cams=cams, class_idx=idx, scores=scores, layer=layer_name, method=method)
