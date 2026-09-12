"""Compare Conv2D analytical gradients with finite differences."""

from __future__ import annotations

import numpy as np

from cnn_backprop.checks import numerical_gradient, relative_error
from cnn_backprop.layers import Conv2D


def main() -> None:
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(2, 2, 4, 4))
    layer = Conv2D(2, 2, 3, padding=1, rng=rng)
    upstream = rng.normal(size=layer.forward(inputs).shape)

    layer.forward(inputs)
    analytical_input = layer.backward(upstream).copy()
    analytical_weight = layer.dweight.copy()
    analytical_bias = layer.dbias.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * upstream))

    numerical_input = numerical_gradient(objective, inputs)
    numerical_weight = numerical_gradient(objective, layer.weight)
    numerical_bias = numerical_gradient(objective, layer.bias)

    errors = {
        "input": relative_error(analytical_input, numerical_input),
        "weight": relative_error(analytical_weight, numerical_weight),
        "bias": relative_error(analytical_bias, numerical_bias),
    }
    for name, error in errors.items():
        print(f"{name:>6} relative error: {error:.3e}")

    if max(errors.values()) >= 1e-7:
        raise SystemExit("Gradient check failed.")
    print("PASS: analytical Conv2D gradients match finite differences.")


if __name__ == "__main__":
    main()
