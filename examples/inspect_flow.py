"""Print activation shapes and gradient norms for one complete CNN pass."""

from __future__ import annotations

import numpy as np

from cnn_backprop import build_mnist_cnn, softmax_cross_entropy


def main() -> None:
    rng = np.random.default_rng(11)
    model = build_mnist_cnn()
    activation = rng.normal(size=(2, 1, 28, 28))

    print("FORWARD")
    for index, layer in enumerate(model.layers):
        input_shape = activation.shape
        activation = layer.forward(activation)
        print(
            f"{index:02d} {layer.__class__.__name__:<10} "
            f"{str(input_shape):>18} -> {activation.shape}"
        )

    loss, gradient = softmax_cross_entropy(activation, np.array([2, 7]))
    print(f"\nloss={loss:.6f}\n\nBACKWARD")
    for index, layer in reversed(list(enumerate(model.layers))):
        output_shape = gradient.shape
        gradient = layer.backward(gradient)
        print(
            f"{index:02d} {layer.__class__.__name__:<10} "
            f"{str(output_shape):>18} -> {gradient.shape} "
            f"|grad|={np.linalg.norm(gradient):.3e}"
        )


if __name__ == "__main__":
    main()
