"""Neural-network layers with explicit NumPy forward and backward passes.

The implementation favors readability and testability over raw speed. Tensors use
the NCHW convention: ``(batch, channels, height, width)``.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

Array = np.ndarray


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError("Expected an int or a pair of integers.")
    return value


class Layer:
    """Minimal layer protocol used by :class:`Sequential`."""

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        raise NotImplementedError

    def backward(self, gradient: Array) -> Array:
        raise NotImplementedError

    def parameters_and_gradients(self) -> Iterator[tuple[str, Array, Array]]:
        return iter(())


class Conv2D(Layer):
    """A 2-D cross-correlation layer with explicit input/weight gradients."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        rng: np.random.Generator | None = None,
    ) -> None:
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Channel counts must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)

        if min(*self.kernel_size, *self.stride) <= 0 or min(*self.padding) < 0:
            raise ValueError("Kernel/stride must be positive and padding non-negative.")

        generator = rng or np.random.default_rng()
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight = generator.normal(
            0.0,
            np.sqrt(2.0 / fan_in),
            size=(out_channels, in_channels, *self.kernel_size),
        )
        self.bias = np.zeros(out_channels, dtype=np.float64)
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)

        self._input_shape: tuple[int, ...] | None = None
        self._padded_input: Array | None = None
        self._windows: Array | None = None

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        del training
        if inputs.ndim != 4:
            raise ValueError(f"Conv2D expects NCHW input, received shape {inputs.shape}.")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, received {inputs.shape[1]}."
            )

        pad_h, pad_w = self.padding
        padded = np.pad(
            inputs,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        if padded.shape[2] < kernel_h or padded.shape[3] < kernel_w:
            raise ValueError("Kernel cannot be larger than the padded input.")

        windows = sliding_window_view(padded, (kernel_h, kernel_w), axis=(2, 3))
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]
        output = np.einsum("nchwij,ocij->nohw", windows, self.weight, optimize=True)
        output += self.bias[None, :, None, None]

        self._input_shape = inputs.shape
        self._padded_input = padded
        self._windows = windows
        return output

    def backward(self, gradient: Array) -> Array:
        if self._windows is None or self._padded_input is None or self._input_shape is None:
            raise RuntimeError("Conv2D.backward() called before forward().")

        expected = (
            self._input_shape[0],
            self.out_channels,
            self._windows.shape[2],
            self._windows.shape[3],
        )
        if gradient.shape != expected:
            raise ValueError(f"Expected output gradient shape {expected}, got {gradient.shape}.")

        self.dweight[...] = np.einsum("nchwij,nohw->ocij", self._windows, gradient, optimize=True)
        self.dbias[...] = gradient.sum(axis=(0, 2, 3))

        padded_gradient = np.zeros_like(self._padded_input)
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h, output_w = gradient.shape[2:]

        for kernel_row in range(kernel_h):
            for kernel_col in range(kernel_w):
                contribution = np.einsum(
                    "nohw,oc->nchw",
                    gradient,
                    self.weight[:, :, kernel_row, kernel_col],
                    optimize=True,
                )
                padded_gradient[
                    :,
                    :,
                    kernel_row : kernel_row + stride_h * output_h : stride_h,
                    kernel_col : kernel_col + stride_w * output_w : stride_w,
                ] += contribution

        pad_h, pad_w = self.padding
        height_slice = slice(pad_h, -pad_h or None)
        width_slice = slice(pad_w, -pad_w or None)
        return padded_gradient[:, :, height_slice, width_slice]

    def parameters_and_gradients(self) -> Iterator[tuple[str, Array, Array]]:
        yield "weight", self.weight, self.dweight
        yield "bias", self.bias, self.dbias


class MaxPool2D(Layer):
    """Max pooling that routes each gradient to exactly one winning element."""

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 2,
        *,
        stride: int | tuple[int, int] | None = None,
    ) -> None:
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride if stride is not None else kernel_size)
        if min(*self.kernel_size, *self.stride) <= 0:
            raise ValueError("Kernel and stride must be positive.")
        self._input_shape: tuple[int, ...] | None = None
        self._argmax: Array | None = None

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        del training
        if inputs.ndim != 4:
            raise ValueError(f"MaxPool2D expects NCHW input, received shape {inputs.shape}.")

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        windows = sliding_window_view(inputs, (kernel_h, kernel_w), axis=(2, 3))
        windows = windows[:, :, ::stride_h, ::stride_w, :, :]
        flat_windows = windows.reshape(*windows.shape[:4], kernel_h * kernel_w)

        self._input_shape = inputs.shape
        self._argmax = flat_windows.argmax(axis=-1)
        return flat_windows.max(axis=-1)

    def backward(self, gradient: Array) -> Array:
        if self._input_shape is None or self._argmax is None:
            raise RuntimeError("MaxPool2D.backward() called before forward().")
        if gradient.shape != self._argmax.shape:
            raise ValueError(f"Expected gradient shape {self._argmax.shape}, got {gradient.shape}.")

        input_gradient = np.zeros(self._input_shape, dtype=gradient.dtype)
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride

        for row in range(gradient.shape[2]):
            for col in range(gradient.shape[3]):
                winners = self._argmax[:, :, row, col]
                for flat_index in range(kernel_h * kernel_w):
                    kernel_row, kernel_col = divmod(flat_index, kernel_w)
                    input_gradient[
                        :, :, row * stride_h + kernel_row, col * stride_w + kernel_col
                    ] += gradient[:, :, row, col] * (winners == flat_index)
        return input_gradient


class ReLU(Layer):
    def __init__(self) -> None:
        self._positive_mask: Array | None = None

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        del training
        self._positive_mask = inputs > 0
        return np.maximum(inputs, 0)

    def backward(self, gradient: Array) -> Array:
        if self._positive_mask is None:
            raise RuntimeError("ReLU.backward() called before forward().")
        return gradient * self._positive_mask


class Flatten(Layer):
    def __init__(self) -> None:
        self._input_shape: tuple[int, ...] | None = None

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        del training
        if inputs.ndim < 2:
            raise ValueError("Flatten expects a batch dimension.")
        self._input_shape = inputs.shape
        return inputs.reshape(inputs.shape[0], -1)

    def backward(self, gradient: Array) -> Array:
        if self._input_shape is None:
            raise RuntimeError("Flatten.backward() called before forward().")
        return gradient.reshape(self._input_shape)


class Dense(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Feature counts must be positive.")
        generator = rng or np.random.default_rng()
        self.weight = generator.normal(
            0.0, np.sqrt(2.0 / in_features), size=(in_features, out_features)
        )
        self.bias = np.zeros(out_features, dtype=np.float64)
        self.dweight = np.zeros_like(self.weight)
        self.dbias = np.zeros_like(self.bias)
        self._inputs: Array | None = None

    def forward(self, inputs: Array, *, training: bool = True) -> Array:
        del training
        if inputs.ndim != 2 or inputs.shape[1] != self.weight.shape[0]:
            raise ValueError(
                f"Dense expects shape (batch, {self.weight.shape[0]}), got {inputs.shape}."
            )
        self._inputs = inputs
        return inputs @ self.weight + self.bias

    def backward(self, gradient: Array) -> Array:
        if self._inputs is None:
            raise RuntimeError("Dense.backward() called before forward().")
        if gradient.shape != (self._inputs.shape[0], self.weight.shape[1]):
            raise ValueError("Output gradient has an incompatible shape.")

        self.dweight[...] = self._inputs.T @ gradient
        self.dbias[...] = gradient.sum(axis=0)
        return gradient @ self.weight.T

    def parameters_and_gradients(self) -> Iterator[tuple[str, Array, Array]]:
        yield "weight", self.weight, self.dweight
        yield "bias", self.bias, self.dbias
