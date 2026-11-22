import torch


def test_tensor_operations():
    x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    loss = (x ** 2).sum()
    loss.backward()

    expected = torch.tensor([2.0, 4.0, 6.0])

    assert x.grad is not None
    assert torch.allclose(x.grad, expected)
