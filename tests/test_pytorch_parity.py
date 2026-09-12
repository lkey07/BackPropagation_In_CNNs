import numpy as np
import pytest

from cnn_backprop.layers import Conv2D

torch = pytest.importorskip("torch")


def test_conv2d_matches_pytorch_forward_and_backward() -> None:
    rng = np.random.default_rng(17)
    inputs = rng.normal(size=(2, 2, 5, 5))
    layer = Conv2D(2, 3, 3, stride=2, padding=1, rng=rng)
    output = layer.forward(inputs)
    upstream = rng.normal(size=output.shape)
    input_gradient = layer.backward(upstream)

    torch_layer = torch.nn.Conv2d(2, 3, 3, stride=2, padding=1).double()
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(layer.weight))
        torch_layer.bias.copy_(torch.from_numpy(layer.bias))

    torch_inputs = torch.tensor(inputs, dtype=torch.float64, requires_grad=True)
    torch_output = torch_layer(torch_inputs)
    torch_output.backward(torch.from_numpy(upstream))

    np.testing.assert_allclose(output, torch_output.detach().numpy(), atol=1e-11)
    np.testing.assert_allclose(input_gradient, torch_inputs.grad.numpy(), atol=1e-11)
    np.testing.assert_allclose(layer.dweight, torch_layer.weight.grad.numpy(), atol=1e-11)
    np.testing.assert_allclose(layer.dbias, torch_layer.bias.grad.numpy(), atol=1e-11)
