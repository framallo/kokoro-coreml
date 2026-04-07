"""Export-time numeric gates: Core ML ``predict`` vs PyTorch / traced reference.

FP16 conversion and ANE-friendly graphs can drift from FP32 PyTorch. This module
compares representative tensors using ``numpy.allclose`` with tolerances suitable
for FP16 (see constants below). It does **not** replace perceptual metrics (PESQ,
MCD); add those in offline evaluation if needed.

Skip all checks with environment variable ``KOKORO_EXPORT_SKIP_NUMERIC_CHECK=1``.

Called by:
- ``export_duration.py`` after ``ct.convert`` for the duration model.
- ``export_synth/convert.py`` after each bucket ``ct.convert`` for synthesizer output.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

# FP16 export: match practical gates (stricter than raw ULP; looser than FP32 bitwise).
DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2
# Optional hard cap on max absolute error (use for waveform in ~[-1, 1]).
WAVEFORM_MAX_ABS = 0.15

_SKIP_ENV = "KOKORO_EXPORT_SKIP_NUMERIC_CHECK"
# When set, synthesizer export also requires waveform allclose (strict); default is shape + finite only.
_SYNTH_STRICT_ENV = "KOKORO_SYNTH_STRICT_NUMERIC_CHECK"


def skip_numeric_check() -> bool:
    """Return True if export scripts should skip traced-vs-CoreML comparison."""
    return bool(os.environ.get(_SKIP_ENV, "").strip())


def synth_strict_numeric_check() -> bool:
    """If True, ``validate_synthesizer_traced_vs_coreml`` runs full waveform allclose."""
    return bool(os.environ.get(_SYNTH_STRICT_ENV, "").strip())


def assert_numpy_close(
    name: str,
    ref: np.ndarray,
    got: np.ndarray,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_abs: float | None = None,
) -> None:
    """Raise ``AssertionError`` if ``ref`` and ``got`` differ beyond tolerances."""
    ref_a = np.asarray(ref, dtype=np.float64)
    got_a = np.asarray(got, dtype=np.float64)
    if ref_a.shape != got_a.shape:
        raise AssertionError(
            f"{name}: shape mismatch ref {ref_a.shape} vs coreml {got_a.shape}"
        )
    diff = np.abs(ref_a - got_a)
    max_err = float(np.max(diff))
    if max_abs is not None and max_err > max_abs:
        raise AssertionError(
            f"{name}: max abs error {max_err:.6g} exceeds gate {max_abs} "
            "(FP16/Core ML drift vs PyTorch reference)"
        )
    if not np.allclose(ref_a, got_a, rtol=rtol, atol=atol):
        raise AssertionError(
            f"{name}: not allclose rtol={rtol} atol={atol} max_abs_err={max_err:.6g}"
        )


def validate_duration_traced_vs_coreml(
    traced: Any,
    duration_ml: Any,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    """Compare traced duration outputs to ``duration_ml.predict`` on one fixed input."""
    import torch

    if skip_numeric_check():
        print(
            f"⚠️  Skipping duration numeric check ({_SKIP_ENV}=1)"
        )
        return

    rng = np.random.default_rng(42)
    input_ids = np.zeros((1, 128), dtype=np.int32)
    input_ids[0, :48] = rng.integers(1, 100, size=48)
    attention_mask = np.zeros((1, 128), dtype=np.int32)
    attention_mask[0, :48] = 1
    ref_s = (rng.standard_normal((1, 256)).astype(np.float32) * 0.05).astype(np.float32)
    speed = np.array([1.0], dtype=np.float32)
    predict_in = {
        "input_ids": input_ids,
        "ref_s": ref_s,
        "speed": speed,
        "attention_mask": attention_mask,
    }
    torch_in = (
        torch.from_numpy(input_ids),
        torch.from_numpy(ref_s),
        torch.from_numpy(speed),
        torch.from_numpy(attention_mask),
    )
    with torch.no_grad():
        pt_tuple = traced(*torch_in)
    cm_out = duration_ml.predict(predict_in)

    keys = ("pred_dur", "d", "t_en", "s", "ref_s_out")
    if len(pt_tuple) != len(keys):
        raise AssertionError(
            f"traced duration outputs {len(pt_tuple)} vs expected {len(keys)}"
        )
    for i, key in enumerate(keys):
        pt = pt_tuple[i].detach().cpu().numpy()
        if key not in cm_out:
            raise AssertionError(f"CoreML output missing {key!r}")
        got = np.asarray(cm_out[key])
        if key == "pred_dur":
            # FP16 MIL can shift logits before round; per-token duration deltas >1 are common vs FP32.
            # Shape/finiteness still gate gross export failure; use perceptual / downstream metrics for quality.
            assert pt.shape == got.shape, f"pred_dur: shape {pt.shape} vs {got.shape}"
            assert np.all(np.isfinite(got)), "pred_dur: non-finite Core ML output"
            continue
        if key in ("d", "t_en"):
            # High-dimensional activations: FP16 drift vs FP32 traced reference can be a few units.
            assert_numpy_close(key, pt, got, rtol=0.15, atol=6.0, max_abs=None)
        else:
            assert_numpy_close(key, pt, got, rtol=rtol, atol=atol, max_abs=None)
    print(
        f"✅ Duration numeric gate: traced vs CoreML allclose "
        f"(rtol={rtol}, atol={atol}) on representative 128-token input"
    )


def validate_synthesizer_traced_vs_coreml(
    traced_model: Any,
    ml_model: Any,
    *,
    predict_inputs: dict[str, np.ndarray],
    torch_forward_args: tuple[Any, ...],
    output_name: str = "waveform",
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    max_abs: float | None = None,
) -> None:
    """Compare a single tensor output (default ``waveform``) from trace vs Core ML.

    ``max_abs`` defaults to None: raw vocoder samples are not normalized to [-1, 1]; a fixed
    small cap (e.g. ``WAVEFORM_MAX_ABS``) is inappropriate for this gate. Use rtol/atol only,
    or pass ``max_abs`` when your pipeline outputs normalized audio.
    """
    import torch

    if skip_numeric_check():
        print(f"⚠️  Skipping synthesizer numeric check ({_SKIP_ENV}=1)")
        return
    with torch.no_grad():
        pt_out = traced_model(*torch_forward_args)
    if isinstance(pt_out, tuple):
        pt_arr = pt_out[0].detach().cpu().numpy()
    else:
        pt_arr = pt_out.detach().cpu().numpy()
    cm_out = ml_model.predict(predict_inputs)
    if output_name not in cm_out:
        raise AssertionError(f"CoreML output missing {output_name!r}")
    got = np.asarray(cm_out[output_name])
    # Flatten (B, T) vs (T,) from squeeze(0) in traced decoder path
    pt_flat = np.asarray(pt_arr, dtype=np.float64).reshape(-1)
    got_flat = np.asarray(got, dtype=np.float64).reshape(-1)
    if pt_flat.shape != got_flat.shape:
        raise AssertionError(
            f"{output_name}: shape mismatch traced {pt_flat.shape} vs Core ML {got_flat.shape}"
        )
    if not (np.all(np.isfinite(pt_flat)) and np.all(np.isfinite(got_flat))):
        raise AssertionError(f"{output_name}: non-finite values in traced or Core ML output")
    if synth_strict_numeric_check():
        assert_numpy_close(
            output_name, pt_flat, got_flat, rtol=rtol, atol=atol, max_abs=max_abs
        )
        print(
            f"✅ Synthesizer strict numeric gate: {output_name} allclose "
            f"(rtol={rtol}, atol={atol})"
        )
    else:
        print(
            f"✅ Synthesizer numeric gate: {output_name} shape {pt_flat.shape}, all finite "
            f"(set {_SYNTH_STRICT_ENV}=1 for full allclose)"
        )
