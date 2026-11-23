
import pytest
import torch

from src.attention import CausalSelfAttention
from src.config import GPTConfig


def config(n_heads=8, n_kv_heads=2):
    return GPTConfig(
        n_embd=32,
        n_layers=2,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        context_length=16,
        dropout=0.0,
    )


def test_gqa_output_shape():
    layer = CausalSelfAttention(config())
    x = torch.randn(2, 6, 32)

    assert layer(x).shape == x.shape


def test_kv_group_mapping():
    layer = CausalSelfAttention(config())

    raw = torch.arange(2 * 3 * 4).reshape(1, 2, 3, 4)
    repeated = layer.expand_kv(raw)

    assert repeated.shape == (1, 8, 3, 4)

    for head in range(8):
        assert torch.equal(
            repeated[:, head], raw[:, head // 4]
        )


def test_projection_size_reduction():
    gqa = CausalSelfAttention(config())
    mha = CausalSelfAttention(config(n_kv_heads=8))

    def count(layer):
        return sum(p.numel() for p in layer.parameters())

    assert count(gqa) < count(mha)
    assert gqa.k_proj.out_features == 8
    assert mha.k_proj.out_features == 32


def test_causal_attention():
    torch.manual_seed(11)

    layer = CausalSelfAttention(config()).eval()
    x = torch.randn(1, 6, 32)

    altered = x.clone()
    altered[:, 4:] += 5.0

    with torch.no_grad():
        original_output = layer(x)
        altered_output = layer(altered)

    assert torch.allclose(
        original_output[:, :4],
        altered_output[:, :4],
        atol=1e-6,
    )


def test_qkv_gradients():
    layer = CausalSelfAttention(config())
    x = torch.randn(2, 5, 32)

    loss = layer(x).square().mean()
    loss.backward()

    for projection in (
        layer.q_proj, layer.k_proj, layer.v_proj
    ):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()


def test_mha_mode():
    layer = CausalSelfAttention(config(n_kv_heads=8))

    assert layer.group_size == 1
    assert layer(torch.randn(1, 5, 32)).shape == (1, 5, 32)


@pytest.mark.parametrize("kv_heads", [0, 3, 9])
def test_invalid_kv_heads(kv_heads):
    with pytest.raises(ValueError):
        config(n_kv_heads=kv_heads)
