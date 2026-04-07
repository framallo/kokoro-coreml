"""Tests for ``kokoro.coreml_numeric_validate`` (no coremltools; load module by path)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "coreml_numeric_validate",
    _ROOT / "kokoro" / "coreml_numeric_validate.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
assert_numpy_close = _mod.assert_numpy_close


def test_assert_numpy_close_passes():
    a = np.ones((2, 3), dtype=np.float32) * 0.1
    b = np.ones((2, 3), dtype=np.float32) * 0.1
    assert_numpy_close("t", a, b, rtol=1e-5, atol=1e-5, max_abs=None)


def test_assert_numpy_close_fails_on_shape():
    with pytest.raises(AssertionError, match="shape"):
        assert_numpy_close("t", np.zeros((2,)), np.zeros((3,)), max_abs=None)


def test_assert_numpy_close_fails_on_mismatch():
    with pytest.raises(AssertionError, match="not allclose"):
        assert_numpy_close(
            "t",
            np.array([1.0, 2.0]),
            np.array([1.0, 9.0]),
            rtol=1e-9,
            atol=1e-9,
            max_abs=None,
        )


def test_max_abs_gate():
    with pytest.raises(AssertionError, match="max abs error"):
        assert_numpy_close(
            "w",
            np.array([0.0]),
            np.array([0.5]),
            rtol=1.0,
            atol=1.0,
            max_abs=0.1,
        )
