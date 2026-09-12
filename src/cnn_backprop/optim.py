"""Small optimizers that keep parameter updates separate from backpropagation."""

from __future__ import annotations

import numpy as np

from .model import Sequential


class SGD:
    def __init__(
        self,
        learning_rate: float = 0.01,
        *,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must be in [0, 1).")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._velocity: dict[str, np.ndarray] = {}

    def step(self, model: Sequential) -> None:
        for name, parameter, gradient in model.parameters_and_gradients():
            effective_gradient = gradient + self.weight_decay * parameter
            velocity = self._velocity.setdefault(name, np.zeros_like(parameter))
            velocity *= self.momentum
            velocity += effective_gradient
            parameter -= self.learning_rate * velocity
