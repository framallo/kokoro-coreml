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


def test_duration_model_reuses_already_masked_predictor_lstm(kmodel):
    from export_synth.wrappers import DurationModel, MaskedBidirectionalLSTM

    kmodel.predictor.lstm = MaskedBidirectionalLSTM(kmodel.predictor.lstm)

    dm = DurationModel(kmodel)

    assert dm.duration_lstm is kmodel.predictor.lstm


def test_duration_model_padded_input_matches_exact_valid_prefix(kmodel):
    from export_synth.wrappers import DurationModel

    dm = DurationModel(kmodel)
    dm.eval()
    # Bakeoff 3s token prefix. It is long enough to expose right-padding drift
    # through the shared bidirectional duration LSTM.
    valid_ids = torch.tensor(
        [
            0, 81, 83, 16, 53, 65, 156, 102, 53, 16, 44, 123,
            156, 39, 56, 16, 48, 156, 69, 53, 61, 16, 82, 156,
            138, 55, 58, 61, 16, 156, 31, 64, 83, 123, 16, 81,
            83, 16, 46, 156, 76, 92, 4, 0,
        ],
        dtype=torch.long,
    ).unsqueeze(0)
    padded_ids = torch.zeros((1, 64), dtype=torch.long)
    padded_ids[:, : valid_ids.shape[1]] = valid_ids
    exact_mask = torch.ones_like(valid_ids)
    padded_mask = torch.zeros_like(padded_ids)
    padded_mask[:, : valid_ids.shape[1]] = 1
    torch.manual_seed(1234)
    ref_s = torch.randn(1, 256)
    speed = torch.ones(1)

    with torch.no_grad():
        exact_pred, *_ = dm(valid_ids, ref_s, speed, exact_mask)
        padded_pred, *_ = dm(padded_ids, ref_s, speed, padded_mask)

    assert torch.equal(exact_pred, padded_pred[:, : valid_ids.shape[1]])


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
    if audio.ndim == 2:
        assert audio.shape[0] == 1
        audio = audio.squeeze(0)
    assert audio.ndim == 1
    assert audio.numel() > 0
