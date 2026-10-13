"""Load fastai and PyTorch models into a plain ``torch.nn.Module``."""

from ._load import detect_format, extract_state_dict, load_model

__all__ = ["detect_format", "extract_state_dict", "load_model"]
