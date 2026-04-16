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
_DECODER_HAR_POST_3S_PKG = _ROOT / "coreml" / "kokoro_decoder_har_post_3s.mlpackage"
_DECODER_HAR_POST_10S_PKG = _ROOT / "coreml" / "kokoro_decoder_har_post_10s.mlpackage"
_SYNTH_3S_PKG = _ROOT / "coreml" / "kokoro_synthesizer_3s.mlpackage"

ct = pytest.importorskip("coremltools", reason="coremltools not installed")


def _multiarray_shape(desc) -> tuple[int, ...]:
    """Best-effort static shape from Core ML feature type (empty if unknown/dynamic)."""
    try:
        return tuple(int(x) for x in desc.type.multiArrayType.shape)
    except Exception:
        return ()


@pytest.mark.parametrize("bucket_sec", [3, 7, 10, 15, 30])
def test_decoder_har_post_bucket_shape_matches_advertised_duration(bucket_sec):
    """GeneratorFromHar package names must mean enough waveform capacity for that duration."""
    post_pkg = _ROOT / f"coreml/kokoro_decoder_har_post_{bucket_sec}s.mlpackage"
    pre_pkg = _ROOT / f"coreml/kokoro_decoder_pre_{bucket_sec}s.mlpackage"
    if not post_pkg.is_dir() or not pre_pkg.is_dir():
        pytest.skip(f"Core ML packages for {bucket_sec}s bucket not in tree")

    post_spec = ct.utils.load_spec(str(post_pkg))
    pre_spec = ct.utils.load_spec(str(pre_pkg))

    post_inputs = {i.name: _multiarray_shape(i) for i in post_spec.description.input}
    post_outputs = {o.name: _multiarray_shape(o) for o in post_spec.description.output}
    pre_outputs = {o.name: _multiarray_shape(o) for o in pre_spec.description.output}

    expected_har_time = bucket_sec * 24_000 * 2 // 5 + 1
    assert post_inputs["x_pre"][-1] == pre_outputs["x_pre"][-1]
    assert post_inputs["har"][-1] == expected_har_time
    assert post_outputs["waveform"][-1] >= bucket_sec * 24_000


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


@pytest.mark.skipif(not _DECODER_HAR_POST_3S_PKG.is_dir(), reason="coreml/kokoro_decoder_har_post_3s.mlpackage not in tree")
def test_kokoro_decoder_har_post_3s_mlpackage_loads_and_predict_shapes():
    """Post-hn-nsf tail: x_pre + ref_s + har -> waveform."""
    model = ct.models.MLModel(str(_DECODER_HAR_POST_3S_PKG))
    spec = model.get_spec()
    inputs = {i.name: i for i in spec.description.input}
    assert set(inputs) >= {"x_pre", "ref_s", "har"}

    test_inputs = {}
    for name in ("x_pre", "ref_s", "har"):
        shape = _multiarray_shape(inputs[name])
        assert shape, f"missing static shape for {name}"
        test_inputs[name] = np.zeros(shape, dtype=np.float32)

    out = model.predict(test_inputs)
    assert isinstance(out, dict)
    assert "waveform" in out
    waveform = np.asarray(out["waveform"])
    assert hasattr(waveform, "shape")
    out_specs = {o.name: o for o in spec.description.output}
    expected = _multiarray_shape(out_specs["waveform"])
    if expected:
        assert tuple(waveform.shape) == expected


@pytest.mark.skipif(not _DECODER_HAR_POST_10S_PKG.is_dir(), reason="coreml/kokoro_decoder_har_post_10s.mlpackage not in tree")
def test_kokoro_decoder_har_post_10s_mlpackage_loads_and_predict_shapes():
    """Same contract as 3s bucket; larger static shapes for long-form synthesis."""
    model = ct.models.MLModel(str(_DECODER_HAR_POST_10S_PKG))
    spec = model.get_spec()
    inputs = {i.name: i for i in spec.description.input}
    assert set(inputs) >= {"x_pre", "ref_s", "har"}

    test_inputs = {}
    for name in ("x_pre", "ref_s", "har"):
        shape = _multiarray_shape(inputs[name])
        assert shape, f"missing static shape for {name}"
        test_inputs[name] = np.zeros(shape, dtype=np.float32)

    out = model.predict(test_inputs)
    assert isinstance(out, dict)
    assert "waveform" in out
    waveform = np.asarray(out["waveform"])
    assert hasattr(waveform, "shape")
    out_specs = {o.name: o for o in spec.description.output}
    expected = _multiarray_shape(out_specs["waveform"])
    if expected:
        assert tuple(waveform.shape) == expected
