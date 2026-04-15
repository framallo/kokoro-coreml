"""Fan-in smoke: AdainResBlk1d (decoder path) uses AdaIN1d with Linear fc."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from kokoro.istftnet import AdainResBlk1d


def test_adain_resblk1d_forward_finite():
    torch.manual_seed(0)
    B, C, T, style_dim = 1, 8, 16, 4
    blk = AdainResBlk1d(dim_in=C, dim_out=C, style_dim=style_dim)
    x = torch.randn(B, C, T)
    s = torch.randn(B, style_dim)
    y = blk(x, s)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
