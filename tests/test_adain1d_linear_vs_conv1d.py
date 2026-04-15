"""AdaIN1d: forward math correctness, edge cases, and state_dict round-trip.

History: this file originally tested a Conv1d variant of AdaIN1d (nn.Conv1d
with kernel_size=1 replacing nn.Linear).  That experiment was reverted after
benchmarking showed no improvement — see README/Notes/performance-notes.md
"ANE optimization experiment" section.  The current AdaIN1d uses nn.Linear.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from kokoro.istftnet import AdaIN1d


def _reference_adain_forward(x, s, fc_linear: nn.Linear):
    """Reference AdaIN1d forward using explicit Linear + view math."""
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


def test_adain_forward_matches_reference():
    """AdaIN1d.forward() matches the reference Linear math exactly."""
    torch.manual_seed(0)
    B, C, T, style_dim = 2, 64, 32, 128
    x = torch.randn(B, C, T)
    s = torch.randn(B, style_dim)

    # Build reference with standalone Linear
    fc_lin = nn.Linear(style_dim, C * 2)
    out_ref = _reference_adain_forward(x, s, fc_lin)

    # Copy the same weights into AdaIN1d
    m = AdaIN1d(style_dim, C)
    with torch.no_grad():
        m.fc.weight.copy_(fc_lin.weight)
        m.fc.bias.copy_(fc_lin.bias)
    out = m(x, s)
    assert torch.allclose(out, out_ref, rtol=1e-5, atol=1e-6)


def test_state_dict_round_trip():
    """state_dict save/load round-trips with 2D Linear weight shape."""
    m = AdaIN1d(16, 32)
    torch.manual_seed(1)
    for p in m.parameters():
        p.data.normal_(0, 0.1)
    sd = m.state_dict()

    m2 = AdaIN1d(16, 32)
    m2.load_state_dict(sd, strict=True)

    # Linear weight shape: (out_features, in_features) = (64, 16)
    assert m2.fc.weight.shape == (64, 16), f"unexpected shape {m2.fc.weight.shape}"
    assert m2.fc.weight.shape == sd["fc.weight"].shape
    assert torch.allclose(m2.fc.weight, m.fc.weight)
    assert torch.allclose(m2.fc.bias, m.fc.bias)


def test_adain1d_forward_t1_finite():
    """T=1: var(dim=2, unbiased=False) == 0 by definition; output must be finite.

    With T=1 the normalization collapses to x_norm=0, so output equals beta
    everywhere — deterministic and finite, but different from T>1 behaviour.
    """
    torch.manual_seed(0)
    style_dim, C = 16, 8
    m = AdaIN1d(style_dim, C)
    x = torch.randn(1, C, 1)
    s = torch.randn(1, style_dim)
    y = m(x, s)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_adain1d_channel_mismatch_raises():
    """AdaIN1d raises AssertionError when C != num_features.

    The dead torch.cat padding branch was removed and replaced with an assert
    to guard against silent shape mismatches (see performance-notes.md).
    """
    torch.manual_seed(0)
    style_dim, C = 16, 8
    m = AdaIN1d(style_dim, C)
    # Feed x with wrong channel count
    x_wrong = torch.randn(1, C + 4, 10)
    s = torch.randn(1, style_dim)
    with pytest.raises(AssertionError, match="channel mismatch"):
        m(x_wrong, s)


def test_adain1d_in_resblock_state_dict():
    """AdaIN1d weights load correctly through AdainResBlk1d parent."""
    from kokoro.istftnet import AdainResBlk1d

    style_dim, dim = 8, 16
    blk = AdainResBlk1d(dim_in=dim, dim_out=dim, style_dim=style_dim)

    sd = blk.state_dict()
    blk2 = AdainResBlk1d(dim_in=dim, dim_out=dim, style_dim=style_dim)
    blk2.load_state_dict(sd, strict=True)

    # Linear weight: (out_features, in_features) = (dim*2, style_dim)
    assert blk2.norm1.fc.weight.shape == (dim * 2, style_dim)
    assert blk2.norm2.fc.weight.shape == (dim * 2, style_dim)
    assert torch.allclose(blk2.norm1.fc.weight, blk.norm1.fc.weight)
    assert torch.allclose(blk2.norm2.fc.weight, blk.norm2.fc.weight)
