import numpy as np

from cnn_backprop import SGD, Dense, ReLU, Sequential
from cnn_backprop.losses import softmax_cross_entropy


def test_optimizer_updates_parameters_and_loss_decreases() -> None:
    rng = np.random.default_rng(5)
    inputs = rng.normal(size=(12, 3))
    targets = (inputs[:, 0] > 0).astype(int)
    model = Sequential([Dense(3, 6, rng=rng), ReLU(), Dense(6, 2, rng=rng)])
    optimizer = SGD(learning_rate=0.1)

    initial_loss = softmax_cross_entropy(model.forward(inputs), targets)[0]
    for _ in range(40):
        logits = model.forward(inputs)
        _, gradient = softmax_cross_entropy(logits, targets)
        model.backward(gradient)
        optimizer.step(model)
    final_loss = softmax_cross_entropy(model.forward(inputs), targets)[0]

    assert final_loss < initial_loss * 0.5
