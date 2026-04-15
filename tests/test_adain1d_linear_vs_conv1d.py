"""AdaIN1d: Conv1d path matches Linear reference math; checkpoint hook round-trip."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from kokoro.istftnet import AdaIN1d


def _reference_adain_forward(x, s, fc_linear: nn.Linear):
    """Same as AdaIN1d.forward but using Linear + view for style projection."""
    B, C, T = x.shape
    num_features = fc_linear.out_features // 2
    eps = 1e-5
    mean = x.mean(dim=2, keepdim=True)
    var = x.var(dim=2, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    h = fc_linear(s).view(B, 2 * num_features, 1)
    gamma, beta = torch.chunk(h, 2, dim=1)
    gamma_exp = gamma.expand(B, C, T)
    beta_exp = beta.expand(B, C, T)
    return (1.0 + gamma_exp) * x_norm + beta_exp


def test_adain_conv_matches_linear_reference():
    torch.manual_seed(0)
    B, C, T, style_dim = 2, 64, 32, 128
    x = torch.randn(B, C, T)
    s = torch.randn(B, style_dim)
    fc_lin = nn.Linear(style_dim, C * 2)
    out_ref = _reference_adain_forward(x, s, fc_lin)

    m = AdaIN1d(style_dim, C)
    with torch.no_grad():
        m.fc.weight.copy_(fc_lin.weight.unsqueeze(-1))
        m.fc.bias.copy_(fc_lin.bias)
    out = m(x, s)
    assert torch.allclose(out, out_ref, rtol=1e-5, atol=1e-6)


def test_load_state_dict_hook_3d_round_trip():
    m = AdaIN1d(16, 32)
    torch.manual_seed(1)
    for p in m.parameters():
        p.data.normal_(0, 0.1)
    sd = m.state_dict()
    m2 = AdaIN1d(16, 32)
    m2.load_state_dict(sd, strict=True)
    assert m2.fc.weight.shape == sd["fc.weight"].shape
    assert torch.allclose(m2.fc.weight, m.fc.weight)
    assert torch.allclose(m2.fc.bias, m.fc.bias)


def test_load_state_dict_hook_2d_linear_checkpoint():
    style_dim, C = 8, 16
    m = AdaIN1d(style_dim, C)
    lin_w = torch.randn(C * 2, style_dim)
    lin_b = torch.randn(C * 2)
    sd = {"fc.weight": lin_w, "fc.bias": lin_b}
    m.load_state_dict(sd, strict=True)
    assert m.fc.weight.shape == (C * 2, style_dim, 1)
    assert torch.allclose(m.fc.weight.squeeze(-1), lin_w)
    assert torch.allclose(m.fc.bias, lin_b)
