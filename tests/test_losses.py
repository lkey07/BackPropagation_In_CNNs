import numpy as np

from cnn_backprop.checks import numerical_gradient, relative_error
from cnn_backprop.losses import softmax, softmax_cross_entropy


def test_softmax_is_stable_and_normalized() -> None:
    probabilities = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_softmax_cross_entropy_gradient() -> None:
    logits = np.array([[0.4, -0.2, 1.1], [0.1, 0.8, -0.5]], dtype=np.float64)
    targets = np.array([2, 1])
    _, analytical = softmax_cross_entropy(logits, targets)
    numerical = numerical_gradient(lambda: softmax_cross_entropy(logits, targets)[0], logits)
    assert relative_error(analytical, numerical) < 1e-7
