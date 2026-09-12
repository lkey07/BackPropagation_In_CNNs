import numpy as np

from cnn_backprop.checks import numerical_gradient, relative_error
from cnn_backprop.layers import Conv2D, Dense, Flatten, MaxPool2D, ReLU


def test_conv2d_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(1)
    inputs = rng.normal(size=(1, 2, 4, 4))
    layer = Conv2D(2, 2, 3, padding=1, rng=rng)
    upstream = rng.normal(size=layer.forward(inputs).shape)

    layer.forward(inputs)
    analytical_input = layer.backward(upstream).copy()
    analytical_weight = layer.dweight.copy()
    analytical_bias = layer.dbias.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * upstream))

    assert relative_error(analytical_input, numerical_gradient(objective, inputs)) < 1e-7
    assert relative_error(analytical_weight, numerical_gradient(objective, layer.weight)) < 1e-7
    assert relative_error(analytical_bias, numerical_gradient(objective, layer.bias)) < 1e-7


def test_dense_gradients_match_finite_differences() -> None:
    rng = np.random.default_rng(2)
    inputs = rng.normal(size=(3, 4))
    layer = Dense(4, 2, rng=rng)
    upstream = rng.normal(size=(3, 2))

    layer.forward(inputs)
    analytical_input = layer.backward(upstream).copy()
    analytical_weight = layer.dweight.copy()

    def objective() -> float:
        return float(np.sum(layer.forward(inputs) * upstream))

    assert relative_error(analytical_input, numerical_gradient(objective, inputs)) < 1e-7
    assert relative_error(analytical_weight, numerical_gradient(objective, layer.weight)) < 1e-7


def test_max_pool_routes_gradient_to_single_winner() -> None:
    inputs = np.array([[[[1.0, 3.0], [2.0, 0.0]]]])
    layer = MaxPool2D(2)
    output = layer.forward(inputs)
    gradient = layer.backward(np.ones_like(output))

    np.testing.assert_array_equal(output, [[[[3.0]]]])
    np.testing.assert_array_equal(gradient, [[[[0.0, 1.0], [0.0, 0.0]]]])


def test_relu_and_flatten_restore_expected_shapes() -> None:
    inputs = np.array([[[[-1.0, 2.0], [3.0, -4.0]]]])
    relu = ReLU()
    flatten = Flatten()
    flattened = flatten.forward(relu.forward(inputs))
    gradient = relu.backward(flatten.backward(np.ones_like(flattened)))

    assert flattened.shape == (1, 4)
    np.testing.assert_array_equal(gradient, [[[[0.0, 1.0], [1.0, 0.0]]]])
