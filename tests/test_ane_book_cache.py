#!/usr/bin/env python3
"""Incrementality test for ane_book.py's per-chapter content cache.

Proves that, with `--cache-dir`, editing ONE chapter re-synthesizes only that
chapter on the next run while every other chapter is served from the cache —
without loading Kokoro/CoreML at all. The heavy bits (KPipeline G2P, the
kokoro-bench batch process, the per-chunk synth, and ffmpeg m4b assembly) are
stubbed; only the cache-key + reuse decision in `_build_book` runs for real.

Run:  python3 tests/test_ane_book_cache.py
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import wave
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent
_ANE = _ROOT / "scripts" / "ane_book.py"

# Import scripts/ane_book.py as a module without running main().
_spec = importlib.util.spec_from_file_location("ane_book", _ANE)
ane = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ane)


def _write_valid_wav(path: Path, seconds: float = 0.5) -> None:
    """Write a tiny but valid mono 16-bit WAV (>0.1s so it passes validation)."""
    sr = ane.SAMPLE_RATE
    n = int(sr * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x01\x00" * n)


# Count of real synthesis calls (cache misses).
SYNTH_CALLS: list[str] = []


def _fake_make_chunker(lang, repo_id, voice):
    """Return (chunk_text, vocab, hnsf) without importing kokoro."""
    def chunk_text(text, speed):
        # One non-empty chunk per chapter is enough for the cache logic.
        return [("g", "p", [0.0])]

    vocab = {"p": 1}
    hnsf = {"weights_sha256": "deadbeef-model-tag"}
    return chunk_text, vocab, hnsf


class _FakeBatch:
    """Stands in for BatchSynth; never launches kokoro-bench."""

    def __init__(self, *a, **k):
        pass

    def close(self):
        pass


def _fake_render_chunks_to_wav(batch, chunks, vocab, hnsf_hash, voice, speed,
                               out_wav, tmp, prefix, progress=None, chapter_idx=1):
    """Pretend to synthesize: record the call and write a valid WAV."""
    SYNTH_CALLS.append(prefix)
    _write_valid_wav(out_wav)
    if progress is not None:
        for k in range(len(chunks)):
            progress.tick(chapter_idx, k + 1, len(chunks))
    return len(chunks), 0.01


def _fake_assemble_m4b(chapter_wavs, chapter_titles, out_path, album_title,
                       artist, language="und"):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"FAKE-M4B")
    return [1.0 for _ in chapter_wavs]


def _install_stubs():
    ane._make_chunker = _fake_make_chunker
    ane.BatchSynth = _FakeBatch
    ane._render_chunks_to_wav = _fake_render_chunks_to_wav
    ane._assemble_m4b = _fake_assemble_m4b
    # _build_book ends in os._exit(0); turn that into a catchable exception so
    # the test can assert on what happened.
    def _no_exit(code):
        raise SystemExit(code)
    ane.os._exit = _no_exit


def _make_args(chapters_dir, out, cache_dir, work_dir):
    return argparse.Namespace(
        chapters_dir=str(chapters_dir),
        glob="capitulo-*.md",
        prepend=None,
        out=Path(out),
        title="Test Book",
        artist="Tester",
        voice="ef_dora",
        lang="e",
        speed=1.0,
        compute_units="staged",
        drop_title=False,
        cache_dir=str(cache_dir),
        work_dir=str(work_dir),
        models_dir=str(_ROOT / "coreml"),
        bench="/nonexistent/bench",
        repo_id="hexgrad/Kokoro-82M",
    )


def _run_build(args, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        ane._build_book(args, Path("/nonexistent/bench"), work_dir)
    except SystemExit:
        pass  # _build_book's os._exit(0), stubbed to raise


def main() -> int:
    _install_stubs()

    import tempfile
    tmproot = Path(tempfile.mkdtemp(prefix="ane_cache_test_"))
    chapters = tmproot / "book"
    chapters.mkdir()
    cache_dir = tmproot / "cache"
    out = tmproot / "out" / "book.m4b"

    files = []
    for i in range(1, 4):
        f = chapters / f"capitulo-{i:02d}.md"
        f.write_text(f"# Capítulo {i}: Title {i}\n\nThis is the body of chapter {i}.\n",
                     encoding="utf-8")
        files.append(f)

    # --- Run 1: cold cache, fresh work dir -> all 3 chapters synthesized. ---
    SYNTH_CALLS.clear()
    _run_build(_make_args(chapters, out, cache_dir, tmproot / "work1"),
               tmproot / "work1")
    run1 = list(SYNTH_CALLS)
    assert len(run1) == 3, f"run 1 should synth all 3 chapters, got {run1}"
    cached = sorted(cache_dir.glob("*.wav"))
    assert len(cached) == 3, f"cache should hold 3 WAVs after run 1, got {len(cached)}"

    # --- Edit ONE chapter (chapter 2). ---
    files[1].write_text("# Capítulo 2: Title 2\n\nThis body was EDITED on one line.\n",
                        encoding="utf-8")

    # --- Run 2: FRESH work dir (so only the content cache can serve clips),
    #     same cache dir -> only the edited chapter 2 re-synthesizes. ---
    SYNTH_CALLS.clear()
    _run_build(_make_args(chapters, out, cache_dir, tmproot / "work2"),
               tmproot / "work2")
    run2 = list(SYNTH_CALLS)
    assert len(run2) == 1, (
        f"run 2 should synth exactly the 1 edited chapter, got {run2} "
        f"(cache miss on unchanged chapters?)"
    )
    assert run2[0] == "ch002", f"the synthesized chapter should be ch002, got {run2}"

    # The cache now holds 4 WAVs (3 original + 1 for the edited chapter 2).
    cached2 = sorted(cache_dir.glob("*.wav"))
    assert len(cached2) == 4, f"cache should hold 4 WAVs after the edit, got {len(cached2)}"

    # --- Run 3: re-run unchanged -> zero synthesis (full cache hit). ---
    SYNTH_CALLS.clear()
    _run_build(_make_args(chapters, out, cache_dir, tmproot / "work3"),
               tmproot / "work3")
    run3 = list(SYNTH_CALLS)
    assert len(run3) == 0, f"run 3 (no edits) should be a full cache hit, got {run3}"

    print("PASS: cache incrementality")
    print(f"  run1 synth = {run1}  (3 cold)")
    print(f"  run2 synth = {run2}  (only edited chapter)")
    print(f"  run3 synth = {run3}  (full cache hit)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
