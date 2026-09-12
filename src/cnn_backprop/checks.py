"""Finite-difference helpers used to validate analytical gradients."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Array = np.ndarray


def numerical_gradient(
    function: Callable[[], float],
    values: Array,
    *,
    epsilon: float = 1e-5,
) -> Array:
    gradient = np.zeros_like(values, dtype=np.float64)
    iterator = np.nditer(values, flags=["multi_index"], op_flags=["readwrite"])
    while not iterator.finished:
        index = iterator.multi_index
        original = values[index]
        values[index] = original + epsilon
        positive = function()
        values[index] = original - epsilon
        negative = function()
        values[index] = original
        gradient[index] = (positive - negative) / (2 * epsilon)
        iterator.iternext()
    return gradient


def relative_error(analytical: Array, numerical: Array) -> float:
    denominator = np.maximum(1e-12, np.abs(analytical) + np.abs(numerical))
    return float(np.max(np.abs(analytical - numerical) / denominator))
