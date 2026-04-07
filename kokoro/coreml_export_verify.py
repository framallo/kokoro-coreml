"""Post-export Core ML checks: stderr + coremltools log scan for CPU-fallback hints.

These checks are **heuristic**. Apple does not expose per-op ANE placement in the
Python ``coremltools`` API; definitive proof still uses Xcode Instruments (Core ML
template, Neural Engine track) or ``powermetrics`` on device.

Set ``KOKORO_EXPORT_SKIP_ANE_CHECK=1`` to skip assertions (e.g. CI on non-Apple
hardware or when logs are known to be noisy).

Called by:
- ``export_duration.py`` after ``ct.convert`` and ``predict`` smoke.
- ``archive/export_vocoder.py`` after conversion and first ``predict`` (legacy).
- ``export_synth/convert.py`` after each bucket save (reload + predict).
"""

from __future__ import annotations

import io
import logging
import os
import re
import sys
from contextlib import contextmanager
from typing import IO, Any

# Skip all ANE-related assertions when set (non-empty).
_SKIP_ENV = "KOKORO_EXPORT_SKIP_ANE_CHECK"

# Conservative regexes: positive signals that ops or graphs fell back to CPU.
# Do not match generic "cpu" substrings (e.g. inside unrelated tokens).
_CPU_FALLBACK_PATTERNS: tuple[str, ...] = (
    r"fall\s*back\s+to\s+cpu",
    r"fallback\s+to\s+cpu",
    r"cpu[-\s]?only\s+execution",
    r"execut(?:e|ed|ing)\s+on\s+cpu\b",
    r"running\s+on\s+cpu\b",
    r"will\s+run\s+on\s+cpu\b",
    r"neural\s+engine.*not\s+supported",
    r"not\s+supported.*neural\s+engine",
    r"BNNSEngine::",
    r"Bnns\w*Engine",
    r"Espresso::BNNSEngine",
)

_COMPILED: list[re.Pattern[str]] = [re.compile(p, re.IGNORECASE) for p in _CPU_FALLBACK_PATTERNS]


def _skip_ane_checks() -> bool:
    return bool(os.environ.get(_SKIP_ENV, "").strip())


def should_run_ane_checks() -> bool:
    """Whether ANE log assertions are enabled (darwin, not skipped, NE may exist)."""
    if _skip_ane_checks():
        return False
    if sys.platform != "darwin":
        return False
    return True


def neural_engine_available(ct: Any) -> bool | None:
    """Return True if Core ML reports a Neural Engine device, False if not, None if unknown."""
    try:
        from coremltools.models.compute_device import MLNeuralEngineComputeDevice

        devs = ct.models.MLModel.get_available_compute_devices()
        return any(isinstance(d, MLNeuralEngineComputeDevice) for d in devs)
    except Exception:
        return None


def text_suggests_cpu_fallback(log_text: str) -> str | None:
    """Return a short reason string if ``log_text`` matches a CPU-fallback pattern, else None."""
    if not log_text or not log_text.strip():
        return None
    for rx in _COMPILED:
        m = rx.search(log_text)
        if m:
            return m.group(0)[:200]
    return None


def assert_no_cpu_fallback_in_logs(log_text: str, *, phase: str = "") -> None:
    """Raise ``RuntimeError`` if ``log_text`` contains CPU-fallback signals (Apple Silicon only).

    On hosts with no Neural Engine (e.g. Intel Mac), logs are not asserted so local
    export on legacy hardware does not fail spuriously.
    """
    if not should_run_ane_checks():
        return
    try:
        import coremltools as ct
    except ImportError:
        return
    ne = neural_engine_available(ct)
    if ne is False:
        return
    hit = text_suggests_cpu_fallback(log_text)
    if hit:
        label = f" ({phase})" if phase else ""
        raise RuntimeError(
            f"Core ML log{label} suggests CPU fallback (matched: {hit!r}). "
            f"Inspect export stderr or set {_SKIP_ENV}=1 to skip. "
            "For definitive ANE usage, profile in Instruments."
        )


class _StderrTee:
    """Write to the original stderr and an in-memory buffer."""

    def __init__(self, original: Any, extra: IO[str]) -> None:
        self._original = original
        self._extra = extra

    def write(self, data: str) -> int:
        self._original.write(data)
        return self._extra.write(data)

    def flush(self) -> None:
        self._original.flush()
        self._extra.flush()

    def isatty(self) -> bool:
        return False


@contextmanager
def capture_ane_logs() -> Any:
    """Tee stderr and attach DEBUG handlers for coremltools loggers into one buffer."""
    buf = io.StringIO()
    old_err = sys.stderr
    sys.stderr = _StderrTee(old_err, buf)
    handlers: list[tuple[logging.Logger, logging.Handler]] = []
    for name in (
        "coremltools",
        "coremltools.converters",
        "coremltools.converters.mil",
    ):
        lg = logging.getLogger(name)
        h = logging.StreamHandler(buf)
        h.setLevel(logging.DEBUG)
        lg.addHandler(h)
        handlers.append((lg, h))
        if lg.level > logging.DEBUG:
            lg.setLevel(logging.DEBUG)
    try:
        yield buf
    finally:
        sys.stderr = old_err
        for lg, h in handlers:
            lg.removeHandler(h)


def smoke_predict_assert_no_cpu_fallback(
    ct: Any,
    mlmodel: Any,
    inputs: dict[str, Any],
    *,
    phase: str = "predict",
) -> None:
    """Run ``predict`` once with stderr/logging capture; fail if logs suggest CPU fallback."""
    if not should_run_ane_checks():
        return
    ne = neural_engine_available(ct)
    if ne is False:
        print(
            "⚠️  ANE log check skipped: no Neural Engine reported for this host "
            "(Intel or unsupported configuration)."
        )
        return
    log_text = ""
    with capture_ane_logs() as buf:
        mlmodel.predict(inputs)
        log_text = buf.getvalue()
    assert_no_cpu_fallback_in_logs(log_text, phase=phase)


def merge_log_checks(*log_chunks: str) -> str:
    """Concatenate log fragments for a single ``assert_no_cpu_fallback_in_logs`` call."""
    return "\n".join(c for c in log_chunks if c)
