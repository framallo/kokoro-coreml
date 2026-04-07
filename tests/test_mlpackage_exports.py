"""Optional integration tests: load shipped CoreML packages and assert I/O contracts.

Skipped when ``coreml/kokoro_duration.mlpackage`` is absent or coremltools is not installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_DURATION_PKG = _ROOT / "coreml" / "kokoro_duration.mlpackage"
_DECODER_3S_PKG = _ROOT / "coreml" / "kokoro_decoder_only_3s.mlpackage"
_SYNTH_3S_PKG = _ROOT / "coreml" / "kokoro_synthesizer_3s.mlpackage"

ct = pytest.importorskip("coremltools", reason="coremltools not installed")


def _multiarray_shape(desc) -> tuple[int, ...]:
    """Best-effort static shape from Core ML feature type (empty if unknown/dynamic)."""
    try:
        return tuple(int(x) for x in desc.type.multiArrayType.shape)
    except Exception:
        return ()


@pytest.mark.skipif(not _DURATION_PKG.is_dir(), reason="coreml/kokoro_duration.mlpackage not in tree")
def test_kokoro_duration_mlpackage_loads_and_predict_shapes():
    """Load duration model, run predict, assert output keys and array shapes."""
    model = ct.models.MLModel(str(_DURATION_PKG))
    spec = model.get_spec()
    inputs = {i.name: i for i in spec.description.input}
    assert set(inputs) >= {"input_ids", "ref_s", "speed", "attention_mask"}

    test_inputs = {
        "input_ids": np.zeros((1, 128), dtype=np.int32),
        "attention_mask": np.ones((1, 128), dtype=np.int32),
        "ref_s": np.zeros((1, 256), dtype=np.float32),
        "speed": np.ones((1,), dtype=np.float32),
    }
    out = model.predict(test_inputs)
    assert isinstance(out, dict)
    assert "pred_dur" in out and "d" in out and "t_en" in out and "s" in out and "ref_s_out" in out
    out_specs = {o.name: o for o in spec.description.output}
    for name in ("pred_dur", "d", "t_en", "s", "ref_s_out"):
        arr = out[name]
        assert hasattr(arr, "shape")
        expected = _multiarray_shape(out_specs[name])
        if expected:
            assert tuple(arr.shape) == expected


@pytest.mark.skipif(not _SYNTH_3S_PKG.is_dir(), reason="coreml/kokoro_synthesizer_3s.mlpackage not in tree")
def test_kokoro_synthesizer_3s_mlpackage_loads_and_predict_shapes():
    """Load 3s synthesizer, run predict with zeros; shapes come from the model spec (no hardcoded trace_length)."""
    model = ct.models.MLModel(str(_SYNTH_3S_PKG))
    spec = model.get_spec()
    inputs = {i.name: i for i in spec.description.input}
    assert set(inputs) >= {"d", "t_en", "s", "ref_s", "pred_aln_trg"}

    test_inputs = {}
    for name in ("d", "t_en", "s", "ref_s", "pred_aln_trg"):
        shape = _multiarray_shape(inputs[name])
        assert shape, f"missing static shape for {name}"
        test_inputs[name] = np.zeros(shape, dtype=np.float32)

    out = model.predict(test_inputs)
    assert isinstance(out, dict)
    assert "waveform" in out
    out_specs = {o.name: o for o in spec.description.output}
    for name in ("waveform",):
        arr = out[name]
        assert hasattr(arr, "shape")
        expected = _multiarray_shape(out_specs[name])
        if expected:
            assert tuple(arr.shape) == expected


@pytest.mark.skipif(not _DECODER_3S_PKG.is_dir(), reason="coreml/kokoro_decoder_only_3s.mlpackage not in tree")
def test_kokoro_decoder_only_3s_mlpackage_loads_and_predict_shapes():
    """Load decoder-only 3s model, run predict with zeros, assert finite waveform and shapes."""
    model = ct.models.MLModel(str(_DECODER_3S_PKG))
    spec = model.get_spec()
    inputs = {i.name: i for i in spec.description.input}
    assert set(inputs) >= {"asr", "F0_pred", "N_pred", "ref_s"}

    test_inputs = {}
    for name in ("asr", "F0_pred", "N_pred", "ref_s"):
        shape = _multiarray_shape(inputs[name])
        assert shape, f"missing static shape for {name}"
        test_inputs[name] = np.zeros(shape, dtype=np.float32)

    out = model.predict(test_inputs)
    assert isinstance(out, dict)
    assert "waveform" in out
    waveform = np.asarray(out["waveform"])
    assert np.all(np.isfinite(waveform))
    out_specs = {o.name: o for o in spec.description.output}
    expected = _multiarray_shape(out_specs["waveform"])
    if expected:
        assert tuple(waveform.shape) == expected

