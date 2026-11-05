"""Shared device and threading policy for training and inference."""

import torch


def configure_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {device}")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return device


def configure_threads(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
