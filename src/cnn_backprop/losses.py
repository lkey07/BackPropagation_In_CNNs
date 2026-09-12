"""Numerically stable classification functions."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def softmax(logits: Array) -> Array:
    if logits.ndim != 2:
        raise ValueError("softmax expects a (batch, classes) matrix.")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def softmax_cross_entropy(logits: Array, targets: Array) -> tuple[float, Array]:
    """Return mean cross-entropy and its gradient with respect to logits."""

    probabilities = softmax(logits)
    batch_size, class_count = logits.shape

    if targets.ndim == 1:
        if targets.shape[0] != batch_size:
            raise ValueError("Target count must match batch size.")
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Class-index targets must use an integer dtype.")
        if np.any((targets < 0) | (targets >= class_count)):
            raise ValueError("Target index is outside the class range.")
        target_matrix = np.zeros_like(probabilities)
        target_matrix[np.arange(batch_size), targets] = 1.0
    elif targets.shape == logits.shape:
        target_matrix = targets
    else:
        raise ValueError("Targets must be class indices or a one-hot matrix.")

    log_probabilities = logits - logits.max(axis=1, keepdims=True)
    log_probabilities -= np.log(np.exp(log_probabilities).sum(axis=1, keepdims=True))
    loss = -float(np.sum(target_matrix * log_probabilities) / batch_size)
    gradient = (probabilities - target_matrix) / batch_size
    return loss, gradient


def accuracy(logits: Array, targets: Array) -> float:
    labels = targets.argmax(axis=1) if targets.ndim == 2 else targets
    return float(np.mean(logits.argmax(axis=1) == labels))
