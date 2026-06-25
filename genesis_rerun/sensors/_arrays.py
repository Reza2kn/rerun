"""Small helpers for converting Genesis tensors and batched arrays."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    """Convert Torch/Genesis/Numpy values to a detached CPU ndarray."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def select_env(value: Any, env_index: int) -> np.ndarray:
    """Select one environment from a value that may be batched by Genesis."""

    arr = as_numpy(value)
    if arr.ndim > 1 and arr.shape[0] > env_index:
        return arr[env_index]
    return arr
