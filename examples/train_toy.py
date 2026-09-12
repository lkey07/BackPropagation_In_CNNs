"""Train the NumPy CNN on a tiny, dependency-free shape dataset."""

from __future__ import annotations

import numpy as np

from cnn_backprop import SGD, Conv2D, Dense, Flatten, MaxPool2D, ReLU, Sequential
from cnn_backprop.losses import accuracy, softmax_cross_entropy


def make_dataset(samples: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = rng.normal(0, 0.12, size=(samples, 1, 8, 8))
    labels = np.arange(samples) % 2
    for index, label in enumerate(labels):
        offset = int(rng.integers(-1, 2))
        if label == 0:
            images[index, 0, :, 3 + offset : 5 + offset] += 1.0
        else:
            images[index, 0, 3 + offset : 5 + offset, :] += 1.0
    return np.clip(images, 0, 1), labels


def build_model(seed: int = 42) -> Sequential:
    rng = np.random.default_rng(seed)
    return Sequential(
        [
            Conv2D(1, 4, 3, padding=1, rng=rng),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Dense(4 * 4 * 4, 2, rng=rng),
        ]
    )


def main() -> None:
    train_images, train_labels = make_dataset(160, seed=1)
    test_images, test_labels = make_dataset(40, seed=2)
    model = build_model()
    optimizer = SGD(learning_rate=0.08, momentum=0.9)
    rng = np.random.default_rng(3)

    for epoch in range(1, 16):
        order = rng.permutation(len(train_images))
        losses = []
        for start in range(0, len(order), 16):
            indices = order[start : start + 16]
            logits = model.forward(train_images[indices])
            loss, gradient = softmax_cross_entropy(logits, train_labels[indices])
            model.backward(gradient)
            optimizer.step(model)
            losses.append(loss)

        if epoch in {1, 5, 10, 15}:
            test_logits = model.forward(test_images, training=False)
            print(
                f"epoch={epoch:02d} loss={np.mean(losses):.4f} "
                f"test_accuracy={accuracy(test_logits, test_labels):.3f}"
            )


if __name__ == "__main__":
    main()
