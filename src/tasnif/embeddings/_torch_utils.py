from __future__ import annotations


def resolve_device(device: str = "auto") -> str:
    """Resolve ``"auto"`` to ``cuda`` / ``mps`` / ``cpu`` based on availability."""
    if device != "auto":
        return device

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
