"""Shared device and threading policy for training and inference."""

import torch


def configure_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def configure_threads(threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
