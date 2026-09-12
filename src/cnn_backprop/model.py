"""Composable model utilities for the educational CNN."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from .layers import Conv2D, Dense, Flatten, Layer, MaxPool2D, ReLU

Array = np.ndarray


class Sequential:
    def __init__(self, layers: Sequence[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential requires at least one layer.")
        self.layers = list(layers)

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        output = inputs
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, gradient: Array) -> Array:
        output_gradient = gradient
        for layer in reversed(self.layers):
            output_gradient = layer.backward(output_gradient)
        return output_gradient

    def parameters_and_gradients(self) -> Iterator[tuple[str, Array, Array]]:
        for index, layer in enumerate(self.layers):
            for name, parameter, gradient in layer.parameters_and_gradients():
                yield f"{index}.{layer.__class__.__name__}.{name}", parameter, gradient

    @property
    def parameter_count(self) -> int:
        return sum(parameter.size for _, parameter, _ in self.parameters_and_gradients())


def build_mnist_cnn(*, seed: int = 42) -> Sequential:
    """Construct a compact CNN for 28x28 grayscale images."""

    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Conv2D(1, 8, 3, padding=1, rng=rng),
            ReLU(),
            MaxPool2D(2),
            Conv2D(8, 16, 3, padding=1, rng=rng),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Dense(16 * 7 * 7, 64, rng=rng),
            ReLU(),
            Dense(64, 10, rng=rng),
        ]
    )
