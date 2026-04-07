"""Unit tests for ``kokoro.coreml_export_verify`` (regex heuristics; no coremltools load)."""

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "coreml_export_verify",
    _ROOT / "kokoro" / "coreml_export_verify.py",
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
text_suggests_cpu_fallback = _mod.text_suggests_cpu_fallback
merge_log_checks = _mod.merge_log_checks


def test_text_detects_cpu_fallback_phrase():
    assert (
        text_suggests_cpu_fallback(
            "SomeLayer will run on CPU because ANE does not support this op"
        )
        is not None
    )


def test_text_detects_espresso_bnns_hint():
    assert text_suggests_cpu_fallback("Espresso::BNNSEngine::convolution") is not None


def test_clean_log_no_match():
    assert text_suggests_cpu_fallback("Core ML conversion completed successfully.") is None


def test_merge_log_checks():
    assert "a" in merge_log_checks("a", "b")
    assert merge_log_checks("", "") == ""
