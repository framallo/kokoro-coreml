"""Shape contracts for ``export_synth.wrappers`` DurationModel / SynthesizerModel."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture
def kmodel():
    """Fresh KModel per test (wrappers mutate submodules)."""
    from kokoro import KModel

    try:
        return KModel(disable_complex=True)
    except Exception as exc:
        pytest.skip(f"KModel load failed: {exc}")


def test_duration_model_returns_five_tensors_expected_shapes(kmodel):
    from export_synth.wrappers import DurationModel

    dm = DurationModel(kmodel)
    b, t = 1, 128
    input_ids = torch.zeros((b, t), dtype=torch.long)
    attention_mask = torch.ones((b, t), dtype=torch.long)
    ref_s = torch.randn(b, 256)
    speed = torch.ones(1)
    pred_dur, d, t_en, s, ref_s_out = dm(input_ids, ref_s, speed, attention_mask)

    assert pred_dur.shape == (b, t)
    assert d.ndim == 3 and d.shape[0] == b
    assert t_en.ndim == 3 and t_en.shape[0] == b
    assert s.shape == (b, 128)
    assert ref_s_out.shape == ref_s.shape


def test_synthesizer_model_forward_runs_and_returns_1d_audio(kmodel):
    from kokoro import KModel
    from export_synth.wrappers import DurationModel, SynthesizerModel

    dm = DurationModel(kmodel)
    b, t = 1, 128
    input_ids = torch.zeros((b, t), dtype=torch.long)
    attention_mask = torch.ones((b, t), dtype=torch.long)
    ref_s = torch.randn(b, 256)
    speed = torch.ones(1)
    pred_dur, d, t_en, s, ref_s_out = dm(input_ids, ref_s, speed, attention_mask)

    ttok = d.shape[-1]
    F = 72
    pred_aln_trg = torch.full((ttok, F), 1.0 / F, dtype=torch.float32)

    k2 = KModel(disable_complex=True)
    sm = SynthesizerModel(k2)
    audio = sm(d, t_en, s, ref_s_out, pred_aln_trg)
    assert audio.ndim == 1
    assert audio.numel() > 0

