#!/usr/bin/env python3
"""ANE audiobook pipeline for Kokoro -- fast batch path + m4b assembly.

This is the production-fast sibling of ``scripts/ane_tts.py``. Where ``ane_tts``
spawns a fresh ``kokoro-bench`` subprocess per chunk (paying the ~10s CoreML
compile/load cold start every time), this script drives the Swift bench in its
persistent **batch / stdin** mode: models compile and load ONCE, then every
chunk renders in the same warm process.

Batch protocol (see swift/Sources/KokoroBenchmark/main.swift ``runBatch``):
  1. Launch ``kokoro-bench --batch --models-dir DIR --inputs-dir DIR
     --hnsf-weights FILE``.
  2. The process prints ``READY`` on stdout once weights are loaded.
  3. Write each chunk's input JSON as ``<inputs-dir>/<key>.json`` BEFORE sending
     its command (the Swift side reads the file by key).
  4. Send one JSON command per line on stdin:
       {"input_key": KEY, "seed": 42, "output": RESULT.json, "wav": OUT.wav}
  5. Read ``DONE`` on stdout after each command completes.
  6. Close stdin (EOF) to exit.

Two entry points:

  * ``synth`` -- render a single text file to one WAV (fast batch path).
  * ``book``  -- convert a directory of chapter markdown files into ONE
                 chaptered ``.m4b`` (strip markdown -> batch-render each chapter
                 to a WAV -> ffmpeg concat with chapter markers/titles).

Usage::

    # Single text -> WAV (fast):
    uv run python scripts/ane_book.py synth \
        --text-file chapter.txt --voice ef_dora --lang e --out chapter.wav

    # Whole book -> m4b (the reusable command):
    uv run python scripts/ane_book.py book \
        --chapters-dir /path/to/libro/es \
        --glob 'capitulo-*.md' \
        --voice ef_dora --lang e \
        --out /path/to/output/libro.m4b
"""
from __future__ import annotations

import argparse
import glob as globmod
import hashlib
import json
import os
import re
import shutil
import signal
import array
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# Duration-model enumerated token sizes -- pad input_ids to the nearest.
ENUM_SIZES = [32, 64, 128, 256, 320, 384, 512]

DEFAULT_BENCH = str(
    _ROOT / "swift" / ".build" / "arm64-apple-macosx" / "release" / "kokoro-bench"
)

SAMPLE_RATE = 24000  # kokoro-bench writes mono 16-bit @ 24kHz
SILENCE_SECONDS = 0.3          # gap between chunks within a chapter
CHAPTER_GAP_SECONDS = 0.8      # gap between chapters in the final book

# Per-chunk render watchdog. A render is normally < 30s even on a cold bucket
# compile; if a chunk takes longer than this the CoreML on-device AOT compiler
# has wedged (the ~99% CPU busy-spin in Espresso's graph segmenter). When that
# happens we kill the whole batch process group and restart it -- the partial
# compile is persisted in CoreML's on-disk e5bundlecache, so the retry of the
# same bucket loads fast instead of re-spinning.
CHUNK_TIMEOUT_S = 600.0
# How long to wait for the warm process to print READY (cold start compiles the
# small models once).
READY_TIMEOUT_S = 180.0
# Retries for a single chunk before giving up on the whole book.
CHUNK_MAX_RETRIES = 2

# --------------------------------------------------------------------------- #
# Child-process tracking -- so SIGINT/SIGTERM/atexit can reap EVERY child, the
# kokoro-bench batch process group especially. Nothing must survive a run.
# --------------------------------------------------------------------------- #
_LIVE_PROCS: "set[subprocess.Popen]" = set()
_LIVE_LOCK = threading.Lock()


def _register_proc(proc: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_PROCS.add(proc)


def _unregister_proc(proc: subprocess.Popen) -> None:
    with _LIVE_LOCK:
        _LIVE_PROCS.discard(proc)


def _kill_proc_tree(proc: subprocess.Popen, grace: float = 3.0) -> None:
    """Terminate a child and its whole process group, then SIGKILL the remnant.

    The batch child is launched with ``start_new_session=True`` so it leads its
    own process group; killing the group reaps any helper threads/subprocesses
    CoreML may have spawned. Falls back to killing just the pid if the group is
    gone.
    """
    if proc.poll() is not None:
        _unregister_proc(proc)
        return
    pgid = None
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError):
        pgid = None

    def _signal(sig):
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except (ProcessLookupError, PermissionError):
                pass
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, PermissionError):
            pass

    _signal(signal.SIGTERM)
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal(signal.SIGKILL)
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            pass
    _unregister_proc(proc)


def _kill_all_children() -> None:
    with _LIVE_LOCK:
        procs = list(_LIVE_PROCS)
    for p in procs:
        _kill_proc_tree(p)


_INTERRUPTED = threading.Event()


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        _INTERRUPTED.set()
        _kill_all_children()
        # Re-raise as KeyboardInterrupt so the try/finally unwinds and the
        # process exits with a non-zero status.
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):
            pass  # not in main thread / unsupported


# --------------------------------------------------------------------------- #
# Progress / ETA
# --------------------------------------------------------------------------- #
def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class Progress:
    """Chunk-level progress + ETA, printed to stderr as the book renders."""

    def __init__(self, total_chunks: int, total_chapters: int):
        self.total = max(1, total_chunks)
        self.total_chapters = total_chapters
        self.done = 0
        self.t0 = time.time()

    def tick(self, chapter_idx: int, chunk_in_chapter: int,
             chunks_in_chapter: int, label: str = "") -> None:
        self.done += 1
        elapsed = time.time() - self.t0
        rate = self.done / elapsed if elapsed > 0 else 0.0  # chunks/sec
        remaining = self.total - self.done
        eta = remaining / rate if rate > 0 else float("nan")
        pct = 100.0 * self.done / self.total
        msg = (
            f"[progress] chapter {chapter_idx}/{self.total_chapters}, "
            f"chunk {chunk_in_chapter}/{chunks_in_chapter} "
            f"({self.done}/{self.total} total), {pct:.0f}%, "
            f"ETA {_fmt_eta(eta)}{(' ' + label) if label else ''}"
        )
        if sys.stderr.isatty():
            # Terminal: overwrite the same line in place (no scroll spam).
            sys.stderr.write("\r  " + msg + "\033[K")
            sys.stderr.flush()
        elif self.done % 25 == 0 or self.done == self.total:
            # Log / pipe: throttle to avoid flooding the file.
            sys.stderr.write("  " + msg + "\n")
            sys.stderr.flush()

    def newline(self) -> None:
        """Commit the in-place progress line so the next output starts fresh."""
        if sys.stderr.isatty():
            sys.stderr.write("\n")
            sys.stderr.flush()


# --------------------------------------------------------------------------- #
# Model-side helpers (G2P only; no CoreML loaded in Python)
# --------------------------------------------------------------------------- #
def _load_vocab(repo_id: str) -> dict:
    from huggingface_hub import hf_hub_download

    cfg = json.loads(Path(hf_hub_download(repo_id, "config.json")).read_text())
    return cfg["vocab"]


def _hnsf_weights(repo_id: str) -> dict:
    from kokoro import KModel

    kmodel = KModel(repo_id=repo_id)
    gen = kmodel.decoder.generator
    linear_w = gen.m_source.l_linear.weight.detach().numpy().flatten().tolist()
    linear_b = float(gen.m_source.l_linear.bias.detach().numpy().flatten()[0])
    payload = json.dumps(
        {"linear_weights": linear_w, "linear_bias": linear_b},
        sort_keys=True,
        separators=(",", ":"),
    )
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {"linear_weights": linear_w, "linear_bias": linear_b, "weights_sha256": sha}


def _build_input_entry(key, text, phonemes, voice, speed, vocab, ref_s_list, hnsf_hash):
    """Build the BenchInput JSON the Swift bench decodes."""
    input_ids = [vocab[p] for p in phonemes if p in vocab]
    input_ids = [0] + input_ids + [0]  # BOS / EOS

    enum_T = next((s for s in ENUM_SIZES if s >= len(input_ids)), ENUM_SIZES[-1])
    padded_ids = input_ids[:enum_T] + [0] * max(0, enum_T - len(input_ids))
    attention_mask = [1] * min(len(input_ids), enum_T) + [0] * max(0, enum_T - len(input_ids))

    return {
        "key": key,
        "text": text,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "voice": voice,
        "speed": speed,
        "phonemes": phonemes,
        "input_ids": padded_ids,
        "attention_mask": attention_mask,
        "ref_s": ref_s_list,
        "canonical_duration_s": None,  # Swift derives bucket from its own prediction
        "num_tokens": len(input_ids),
        "hnsf_weights_sha256": hnsf_hash,
    }


# --------------------------------------------------------------------------- #
# WAV helpers
# --------------------------------------------------------------------------- #
def _read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        params = w.getparams()
        frames = w.readframes(w.getnframes())
    return params, frames


def _make_silence(num_frames: int, sampwidth: int, nchannels: int) -> bytes:
    """Generate inter-chunk/inter-chapter silence.

    NOTE: we deliberately do NOT emit a run of pure-zero (``\\x00``) PCM.
    ffmpeg's native AAC encoder collapses runs of exact-zero samples into
    degenerate 4-byte "digital silence" frames, and Apple's AudioToolbox
    decoder (Apple Books, BookPlayer, ...) stalls/underruns when it decodes
    several of those frames in a row -- the playback clock pauses until the
    next real frame. Audiobooks have a silent gap between every chunk, so this
    manifested as repeated mid-chapter halts.

    Instead we fill the gap with an inaudible dither floor (+/- 1 LSB, i.e.
    ~ -90 dBFS at 16-bit). This is below the threshold of hearing yet keeps
    every AAC frame well-formed, so players run straight through the gaps.
    """
    if num_frames <= 0:
        return b""
    total_samples = num_frames * nchannels
    if sampwidth == 2:
        # Alternate -1/+1 so the floor has no DC offset and stays inaudible.
        a = array.array("h", (1 if (i & 1) else -1 for i in range(total_samples)))
        if sys.byteorder == "big":
            a.byteswap()
        return a.tobytes()
    # Fallback for non-16-bit PCM: pure zero (rare; engine writes 16-bit).
    return b"\x00" * total_samples * sampwidth


def _floor_zeros_16(frames: bytes) -> bytes:
    """Replace every exact-zero 16-bit sample with an inaudible +/-1 dither.

    The TTS engine renders leading/trailing/inter-sentence silence as runs of
    pure-zero PCM. ffmpeg's native AAC encoder turns long zero runs into
    degenerate 4-byte "digital silence" frames; Apple's AudioToolbox decoder
    (Apple Books, BookPlayer) stalls when it decodes several in a row, which
    surfaced as repeated mid-chapter playback halts. Mapping 0 -> +/-1 LSB
    (~ -90 dBFS, inaudible, DC-free because we alternate sign) keeps every AAC
    frame well-formed so playback never pauses. This covers ALL silence in the
    final stream, not just the gaps we inject between chunks/chapters.
    """
    a = array.array("h")
    a.frombytes(frames)
    if sys.byteorder == "big":
        a.byteswap()
    sign = 1
    for i, s in enumerate(a):
        if s == 0:
            a[i] = sign
            sign = -sign
    if sys.byteorder == "big":
        a.byteswap()
    return a.tobytes()


def _concat_wavs(chunk_wavs, out_path: Path, gap_seconds: float):
    if not chunk_wavs:
        raise RuntimeError("no chunk WAVs to concatenate")
    params, _ = _read_wav(chunk_wavs[0])
    sw, nc, fr = params.sampwidth, params.nchannels, params.framerate
    silence = _make_silence(int(fr * gap_seconds), sw, nc)
    with wave.open(str(out_path), "wb") as out:
        out.setnchannels(nc)
        out.setsampwidth(sw)
        out.setframerate(fr)
        for i, cw in enumerate(chunk_wavs):
            _, frames = _read_wav(cw)
            if sw == 2:
                frames = _floor_zeros_16(frames)
            out.writeframes(frames)
            if i != len(chunk_wavs) - 1:
                out.writeframes(silence)


def _wav_duration_s(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


# --------------------------------------------------------------------------- #
# Per-chapter content-addressed cache
# --------------------------------------------------------------------------- #
# Bump when the rendered-WAV format changes (sample rate, silence dithering,
# chunking) so old cache entries are not silently reused after a pipeline change.
_CACHE_VERSION = "1"


def _chapter_cache_key(spoken_text: str, voice: str, lang: str, speed: float,
                       model_tag: str) -> str:
    """Content hash for one chapter's rendered audio.

    Hashing the *spoken text* (after the markdown->narration strip the renderer
    already applies) — together with the voice, language, speed, and a
    model/version tag — is what makes a one-line edit re-render only that
    chapter while reordering chapters changes nothing (the key is content, not
    position).
    """
    payload = "\x00".join([
        _CACHE_VERSION,
        str(model_tag),
        str(voice),
        str(lang),
        f"{float(speed):.4f}",
        spoken_text,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Markdown -> narration text
# --------------------------------------------------------------------------- #
_HEADING_RE = re.compile(r"^#\s+Cap[ií]tulo\s+(\d+)\s*[:\.]?\s*(.*)$", re.IGNORECASE)

# Number words for spoken chapter titles (1..30 is plenty for any book).
_NUM_WORDS_ES = {
    1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco", 6: "seis",
    7: "siete", 8: "ocho", 9: "nueve", 10: "diez", 11: "once", 12: "doce",
    13: "trece", 14: "catorce", 15: "quince", 16: "dieciséis",
    17: "diecisiete", 18: "dieciocho", 19: "diecinueve", 20: "veinte",
    21: "veintiuno", 22: "veintidós", 23: "veintitrés", 24: "veinticuatro",
    25: "veinticinco", 26: "veintiséis", 27: "veintisiete",
    28: "veintiocho", 29: "veintinueve", 30: "treinta",
}


def _num_to_words_es(num_str: str) -> str:
    try:
        return _NUM_WORDS_ES.get(int(num_str), num_str)
    except ValueError:
        return num_str


def _normalize_dialogue(line: str) -> str:
    """Turn one line of em-dash (raya) dialogue into TTS-friendly prose.

    Spanish dialogue uses the raya (—) three ways; each needs different
    handling so the line reads as flowing narration instead of choppy
    period-separated fragments:

      * line-leading raya ("—Una cosa...")  -> start of speech: drop it, it is
        NOT a sentence break.
      * a raya sitting right before punctuation, i.e. the *closing* raya after a
        speech tag ("...—dijo Javier—. El orden...")  -> drop the raya and keep
        the punctuation ("...dijo Javier. El orden...").
      * an interior raya that introduces a speech tag ("...no me cierra —dijo
        Javier")  -> a COMMA before the tag ("...no me cierra, dijo Javier").
    """
    # 1. Closing raya hugging punctuation: "Javier—." / "Javier—," / "Javier— ."
    #    -> drop the raya, keep the punctuation.
    line = re.sub(r"\s*—\s*([.,;:?!…])", r"\1", line)
    # 2. Leading raya = start of speech -> drop (do not make it a break).
    line = re.sub(r"^\s*—\s*", "", line)
    # 3. Any remaining interior raya introduces a tag / aside -> comma.
    line = re.sub(r"\s*—\s*", ", ", line)
    return line


def _cleanup_narration(body: str) -> str:
    """Final pass: strip stray tokens and collapse bad punctuation sequences."""
    # Stray pipeline tokens that sometimes leak into the text.
    body = re.sub(r"\[\s*break\s*\]", " ", body, flags=re.IGNORECASE)
    # Double punctuation produced by joining dialogue: ".," ". ," ".¿" ".¡" etc.
    body = re.sub(r"\.\s*,", ".", body)
    body = re.sub(r",\s*\.", ".", body)
    body = re.sub(r"([.;:?!])\s*,", r"\1", body)
    body = re.sub(r"\.\s*([¿¡])", r". \1", body)
    # ", ." -> "." and ",," -> ","
    body = re.sub(r",\s*,+", ",", body)
    # Collapse runs of spaces (but keep newlines).
    body = re.sub(r"[ \t]{2,}", " ", body)
    # Tidy space before punctuation.
    body = re.sub(r"\s+([.,;:?!])", r"\1", body)
    return body


def strip_markdown_chapter(md: str):
    """Return (spoken_title, body_text). Title is "Capitulo N. Title." for audio.

    Drops the markdown heading from the body and turns the rest into clean
    narration: removes emphasis markers, normalizes the em-dash dialogue, and
    collapses blank lines into sentence breaks.
    """
    lines = md.splitlines()
    spoken_title = None
    body_lines = []
    for ln in lines:
        m = _HEADING_RE.match(ln.strip())
        if m and spoken_title is None:
            num, title = m.group(1), m.group(2).strip()
            num_word = _num_to_words_es(num)
            spoken_title = (
                f"Capítulo {num_word}: {title}" if title else f"Capítulo {num_word}"
            )
            continue
        # any other markdown heading -> use the FIRST one as the spoken title
        # (e.g. an unnumbered "Derechos de autor" front-matter page); subsequent
        # headings are spoken inline.
        if ln.strip().startswith("#"):
            txt = ln.lstrip("#").strip()
            # drop trailing pandoc attribute block: "Heading {.unnumbered ...}"
            txt = re.sub(r"\s*\{[^}]*\}\s*$", "", txt).strip()
            if txt and spoken_title is None:
                spoken_title = txt
            elif txt:
                body_lines.append(_normalize_dialogue(txt))
            continue
        body_lines.append(_normalize_dialogue(ln))

    body = "\n".join(body_lines)
    # strip emphasis / inline markdown
    body = re.sub(r"\*\*(.+?)\*\*", r"\1", body)
    body = re.sub(r"\*(.+?)\*", r"\1", body)
    body = re.sub(r"_(.+?)_", r"\1", body)
    body = re.sub(r"`(.+?)`", r"\1", body)
    # collapse 3+ newlines, then blank-line paragraph breaks -> single newline
    body = re.sub(r"\n{2,}", "\n", body)
    body = _cleanup_narration(body)
    body = body.strip()
    return spoken_title, body


# --------------------------------------------------------------------------- #
# Batch driver
# --------------------------------------------------------------------------- #
class _ReadResult:
    """Sentinel for the watchdog-guarded readline."""
    __slots__ = ("line",)

    def __init__(self):
        self.line = None


class BatchSynth:
    """Persistent kokoro-bench --batch process. Models load ONCE.

    Reliability features over the naive driver:

      * The child runs in its OWN session/process group (``start_new_session``)
        so a wedged CoreML compile (the ~99% busy-spin in Espresso's graph
        segmenter) can be killed as a *group*, leaving nothing behind.
      * ``render`` is guarded by a per-chunk WATCHDOG. If the warm process does
        not emit ``DONE`` within ``CHUNK_TIMEOUT_S`` the compile has wedged: we
        kill the group, restart the process (the on-disk CoreML cache makes the
        retried bucket fast) and retry the chunk.
      * Swift stderr is tee'd to a log file (not /dev/null) so compile stalls
        are diagnosable.
    """

    def __init__(self, bench, models_dir, inputs_dir, hnsf_path, compute_units,
                 stderr_log: Path | None = None):
        self.bench = str(bench)
        self.models_dir = str(models_dir)
        self.inputs_dir = Path(inputs_dir)
        self.hnsf_path = str(hnsf_path)
        self.compute_units = compute_units
        self.stderr_log = stderr_log
        self._stderr_fh = None
        self.proc = None
        self._start()

    def _start(self):
        cmd = [
            self.bench,
            "--batch",
            "--models-dir", self.models_dir,
            "--inputs-dir", str(self.inputs_dir),
            "--hnsf-weights", self.hnsf_path,
            "--compute-units", self.compute_units,
        ]
        if self.stderr_log is not None:
            # append so a restart keeps the history
            self._stderr_fh = open(self.stderr_log, "a", buffering=1)
            stderr_dst = self._stderr_fh
        else:
            stderr_dst = subprocess.DEVNULL
        # PRIVATE TMPDIR: kokoro-bench compiles models with MLModel.compileModel,
        # which writes the compiled .mlmodelc into $TMPDIR. If two batches share
        # $TMPDIR (e.g. two concurrent runs), one's temp cleanup can delete the
        # other's compiled model mid-render ("model not found at URL .mlmodelc").
        # Giving each batch its own TMPDIR isolates the compiled-model cache.
        env = dict(os.environ)
        private_tmp = self.inputs_dir / "tmpdir"
        private_tmp.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(private_tmp)
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_dst,
            text=True,
            bufsize=1,
            start_new_session=True,  # own process group -> killable as a tree
            env=env,
        )
        _register_proc(self.proc)
        # Wait for READY, but never block forever on a cold compile.
        if not self._wait_for(("READY",), READY_TIMEOUT_S):
            self._teardown()
            raise RuntimeError("kokoro-bench --batch did not reach READY")

    def _readline_with_timeout(self, timeout: float):
        """Read one stdout line, returning None if the deadline passes.

        Runs the blocking readline in a daemon thread so the caller never hangs
        on a wedged child.
        """
        res = _ReadResult()

        def _reader():
            try:
                res.line = self.proc.stdout.readline()
            except Exception:
                res.line = None

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return None  # timed out -- caller will restart the process
        return res.line

    def _wait_for(self, tokens, timeout: float) -> bool:
        deadline = time.time() + timeout
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            line = self._readline_with_timeout(remaining)
            if line is None:
                return False  # timed out or EOF mid-read
            if line == "":
                return False  # EOF: process exited
            if line.strip() in tokens:
                return True

    def _restart(self):
        sys.stderr.write("  [watchdog] batch process wedged -- killing group and restarting\n")
        sys.stderr.flush()
        self._teardown()
        self._start()

    def render(self, key: str, entry: dict, wav_path: Path, result_path: Path):
        """Write the input JSON, send the command, block until DONE.

        Guarded by the watchdog + restart/retry so one pathological compile
        cannot wedge the whole book.
        """
        (self.inputs_dir / f"{key}.json").write_text(json.dumps(entry))
        cmd = {
            "input_key": key,
            "seed": 42,
            "output": str(result_path),
            "wav": str(wav_path),
        }
        last_err = ""
        for attempt in range(CHUNK_MAX_RETRIES + 1):
            if _INTERRUPTED.is_set():
                raise KeyboardInterrupt
            if wav_path.exists():
                wav_path.unlink()
            try:
                self.proc.stdin.write(json.dumps(cmd) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                self._restart()
                continue
            ok = self._wait_for(("DONE", "ERROR"), CHUNK_TIMEOUT_S)
            if not ok:
                # wedged compile or dead process -> restart and retry
                last_err = "watchdog timeout / process died"
                self._restart()
                continue
            if wav_path.exists():
                return  # success
            # DONE but no WAV: a real Swift-side input error -> surface, no retry
            err = ""
            if result_path.exists():
                try:
                    err = json.loads(result_path.read_text()).get("error", "")
                except Exception:
                    pass
            raise RuntimeError(f"batch render failed for {key}: {err}")
        raise RuntimeError(
            f"batch render for {key} failed after {CHUNK_MAX_RETRIES + 1} attempts: {last_err}"
        )

    def _teardown(self):
        if self.proc is not None:
            try:
                if self.proc.stdin and not self.proc.stdin.closed:
                    self.proc.stdin.close()
            except Exception:
                pass
            _kill_proc_tree(self.proc)
            self.proc = None
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None

    def close(self):
        """Graceful close on the happy path: EOF stdin, wait, then ensure dead."""
        if self.proc is None:
            return
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        # Whether it exited on EOF or not, guarantee the group is gone.
        _kill_proc_tree(self.proc)
        self.proc = None
        if self._stderr_fh is not None:
            try:
                self._stderr_fh.close()
            except Exception:
                pass
            self._stderr_fh = None


def _make_chunker(lang, repo_id, voice):
    """Load KPipeline ONCE; return (chunk_fn, vocab, hnsf, voice_pack)."""
    from kokoro.pipeline import KPipeline, voice_embedding_for_phoneme_string

    print(f"Loading KPipeline (lang={lang}) ...", flush=True)
    pipeline = KPipeline(lang_code=lang, model=False, repo_id=repo_id)
    vocab = _load_vocab(repo_id)
    hnsf = _hnsf_weights(repo_id)
    voice_pack = pipeline.load_voice(voice)

    # Phoneme sequences longer than this overrun the CoreML decoder's largest
    # duration bucket and crash kokoro-bench, so over-long chunks (e.g. a single
    # unbroken legal sentence on a copyright page) get re-split on commas.
    #
    # IMPORTANT (hang fix): the duration bucket is chosen by token count
    # (~= phoneme count + 2). The big buckets (padded_t256 and up, esp. t384 /
    # t512) trigger a pathological CoreML on-device AOT compile -- Espresso's
    # shortest-path graph segmenter busy-spins at ~99% CPU for minutes. The
    # small buckets (t32/t64/t128) compile in well under a second. Capping
    # chunks at ~120 phonemes keeps EVERY chunk in t32..t128, so the wedge never
    # happens. More chunks, but each renders fast and reliably -- and after the
    # one-time t128 compile, every subsequent chunk reuses the warm model.
    # (The watchdog + restart in BatchSynth remains the backstop.)
    MAX_PHONEMES = 120

    def _emit(graphemes, phonemes, out):
        if phonemes and phonemes.strip():
            ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)
            ref_s_list = ref_s.cpu().numpy().flatten().tolist()
            out.append((graphemes, phonemes, ref_s_list))

    def _ps_len(text_piece, speed):
        return sum(len(r.phonemes or "") for r in pipeline(text_piece, voice, speed))

    def _force_split(piece, speed):
        """Split a comma/period-free run by words until each part fits the
        bucket cap. Last-resort guard so NO chunk ever lands in a big bucket."""
        words = piece.split()
        sub, buf = [], ""
        for w in words:
            cand = (buf + " " + w).strip() if buf else w
            if buf and _ps_len(cand, speed) > MAX_PHONEMES:
                sub.append(buf)
                buf = w
            else:
                buf = cand
        if buf.strip():
            sub.append(buf)
        return sub

    def chunk_text(text, speed):
        out = []
        for result in pipeline(text, voice, speed):
            ps = result.phonemes
            if not (ps and ps.strip()):
                continue
            if len(ps) <= MAX_PHONEMES:
                _emit(result.graphemes, ps, out)
                continue
            # Too long: re-split the graphemes on sentence-enders AND
            # commas/semicolons/colons, then re-phonemize each piece so no chunk
            # overruns the small duration buckets. Splitting on '.' '!' '?' too
            # is critical -- a comma-free run of short sentences would otherwise
            # stay one over-long chunk and hit the slow-to-compile big buckets.
            parts = re.split(r"(?<=[.,;:!?…])\s+", (result.graphemes or "").strip())
            # Any single part that is STILL too long (no punctuation) gets a
            # hard word-level split.
            expanded = []
            for p in parts:
                if p and _ps_len(p, speed) > MAX_PHONEMES:
                    expanded.extend(_force_split(p, speed))
                elif p:
                    expanded.append(p)
            buf = ""
            for part in expanded:
                cand = (buf + " " + part).strip() if buf else part
                if buf and _ps_len(cand, speed) > MAX_PHONEMES:
                    for r in pipeline(buf, voice, speed):
                        _emit(r.graphemes, r.phonemes, out)
                    buf = part
                else:
                    buf = cand
            if buf.strip():
                for r in pipeline(buf, voice, speed):
                    _emit(r.graphemes, r.phonemes, out)
        return out

    return chunk_text, vocab, hnsf


def _render_chunks_to_wav(batch, chunks, vocab, hnsf_hash, voice, speed,
                          out_wav: Path, tmp: Path, prefix: str,
                          progress: "Progress | None" = None, chapter_idx: int = 1):
    """Render pre-computed `chunks` through the warm batch, concat to out_wav."""
    if not chunks:
        raise RuntimeError(f"no phonemizable chunks for {prefix}")
    chunk_wavs = []
    render_wall = 0.0
    n = len(chunks)
    for idx, (graphemes, phonemes, ref_s_list) in enumerate(chunks):
        key = f"{prefix}_{idx:04d}"
        entry = _build_input_entry(
            key, graphemes, phonemes, voice, speed, vocab, ref_s_list, hnsf_hash
        )
        wav_path = tmp / f"{key}.wav"
        result_path = tmp / f"{key}.result.json"
        t0 = time.time()
        batch.render(key, entry, wav_path, result_path)
        render_wall += time.time() - t0
        chunk_wavs.append(wav_path)
        if progress is not None:
            progress.tick(chapter_idx, idx + 1, n)
    _concat_wavs(chunk_wavs, out_wav, SILENCE_SECONDS)
    return n, render_wall


def _render_text_to_wav(batch, chunk_text, vocab, hnsf_hash, voice, speed, text,
                        out_wav: Path, tmp: Path, prefix: str,
                        progress: "Progress | None" = None, chapter_idx: int = 1):
    """Chunk `text` then render (used by the single-file synth path)."""
    chunks = chunk_text(text, speed)
    return _render_chunks_to_wav(
        batch, chunks, vocab, hnsf_hash, voice, speed, out_wav, tmp, prefix,
        progress, chapter_idx,
    )


# --------------------------------------------------------------------------- #
# m4b assembly
# --------------------------------------------------------------------------- #
def _format_ffchapter_time(seconds: float) -> int:
    return int(round(seconds * 1000.0))  # milliseconds, TIMEBASE 1/1000


def _build_ffmetadata(chapter_titles, chapter_durations) -> str:
    lines = [";FFMETADATA1"]
    start_ms = 0
    for title, dur in zip(chapter_titles, chapter_durations):
        end_ms = start_ms + _format_ffchapter_time(dur)
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title={title}",
        ]
        start_ms = end_ms
    return "\n".join(lines) + "\n"


def _assemble_m4b(chapter_wavs, chapter_titles, out_path: Path, album_title: str,
                  artist: str, language: str = "und"):
    durations = [_wav_duration_s(w) for w in chapter_wavs]
    tmp = Path(tempfile.mkdtemp(prefix="m4b_"))

    # 1. concat all chapter WAVs (with a gap between them) into one WAV
    full_wav = tmp / "full.wav"
    _concat_wavs(chapter_wavs, full_wav, CHAPTER_GAP_SECONDS)

    # chapter boundaries must account for the inter-chapter gaps we just added
    durations_with_gap = []
    for i, d in enumerate(durations):
        durations_with_gap.append(d + (CHAPTER_GAP_SECONDS if i != len(durations) - 1 else 0.0))

    meta = _build_ffmetadata(chapter_titles, durations_with_gap)
    meta_path = tmp / "chapters.txt"
    meta_path.write_text(meta, encoding="utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",       # regenerate clean, monotonic timestamps
        "-i", str(full_wav),
        "-i", str(meta_path),
        "-map_metadata", "1",
        "-metadata", f"title={album_title}",
        "-metadata", f"album={album_title}",
        "-metadata", f"artist={artist}",
        "-metadata", "genre=Audiobook",
        "-metadata", f"language={language}",
        "-metadata:s:a:0", f"language={language}",
        # One consistent, gap-free AAC stream: a fixed sample rate + channel
        # layout and async resampling so no segment boundary can introduce a
        # timestamp discontinuity that would make players pause mid-chapter.
        "-af", f"aresample=async=1:first_pts=0:osr={SAMPLE_RATE}",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",  # moov atom first: players don't stall on seek/resume
        "-f", "mp4",
        str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-3000:])
        raise RuntimeError("ffmpeg m4b assembly failed")
    return durations_with_gap


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_synth(args):
    bench = Path(args.bench)
    if not bench.exists():
        sys.exit(f"kokoro-bench not found: {bench}")
    text = args.text_file.read_text(encoding="utf-8").strip()
    if not text:
        sys.exit("text file is empty")

    chunk_text, vocab, hnsf = _make_chunker(args.lang, args.repo_id, args.voice)
    tmp = Path(tempfile.mkdtemp(prefix="ane_book_"))
    hnsf_path = tmp / "hnsf_weights.json"
    hnsf_path.write_text(json.dumps(hnsf))
    stderr_log = tmp / "kokoro-bench.stderr.log"

    batch = BatchSynth(bench, args.models_dir, tmp, hnsf_path, args.compute_units,
                       stderr_log=stderr_log)
    try:
        wall0 = time.time()
        n, render_wall = _render_text_to_wav(
            batch, chunk_text, vocab, hnsf["weights_sha256"], args.voice,
            args.speed, text, args.out, tmp, "synth"
        )
        total_wall = time.time() - wall0
    finally:
        batch.close()

    audio_s = _wav_duration_s(args.out)
    print(
        f"\nDONE: {args.out}\n"
        f"  chunks={n}  audio={audio_s:.2f}s  render_wall={render_wall:.2f}s  "
        f"total_wall={total_wall:.2f}s  speed={audio_s / render_wall:.1f}x realtime (render)"
    )


def _resolve_chapter_files(chapters_dir, glob, prepend):
    chap_files = sorted(globmod.glob(str(Path(chapters_dir) / glob)))
    if not chap_files:
        sys.exit(f"no chapters matched {glob} in {chapters_dir}")
    for pre in prepend or []:
        p = pre if Path(pre).is_absolute() else str(Path(chapters_dir) / pre)
        if not Path(p).exists():
            sys.exit(f"--prepend file not found: {p}")
        chap_files.insert(0, p)
    return chap_files


def _build_book(args, bench, tmp):
    """Render one book -> m4b. Returns a summary dict. Caller owns cleanup."""
    chap_files = _resolve_chapter_files(args.chapters_dir, args.glob, args.prepend)
    print(f"Found {len(chap_files)} chapter file(s).", flush=True)

    chunk_text, vocab, hnsf = _make_chunker(args.lang, args.repo_id, args.voice)
    hnsf_path = tmp / "hnsf_weights.json"
    hnsf_path.write_text(json.dumps(hnsf))
    stderr_log = tmp / "kokoro-bench.stderr.log"

    # Persistent per-chapter cache: rendered WAVs keyed by a hash of the spoken
    # text (+ voice/lang/speed/model). Survives across runs so a one-line edit
    # re-renders only the touched chapter; reordering reuses every clip.
    cache_dir = None
    if getattr(args, "cache_dir", None):
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    # Tag the cache by the actual model weights so a model swap invalidates it.
    model_tag = hnsf["weights_sha256"]

    # Pass 1: phonemize/chunk every chapter up front so progress + ETA have a
    # real denominator (total chunks) instead of guessing.
    print("Chunking chapters (G2P) ...", flush=True)
    chapters = []  # list of (title, chunks, spoken_text)
    total_chunks = 0
    for ci, cf in enumerate(chap_files, start=1):
        md = Path(cf).read_text(encoding="utf-8")
        spoken_title, body = strip_markdown_chapter(md)
        title = spoken_title or f"Capítulo {ci}"
        text = (spoken_title + "\n" + body) if (spoken_title and not args.drop_title) else body
        chunks = chunk_text(text, args.speed)
        if not chunks:
            raise RuntimeError(f"no phonemizable chunks for {Path(cf).name}")
        chapters.append((title, chunks, text))
        total_chunks += len(chunks)
    print(f"  {total_chunks} chunks across {len(chapters)} chapters.", flush=True)

    progress = Progress(total_chunks, len(chapters))
    batch = BatchSynth(bench, args.models_dir, tmp, hnsf_path, args.compute_units,
                       stderr_log=stderr_log)

    chapter_wavs = []
    chapter_titles = []
    cache_hits = []  # 1-based chapter indices served from the content cache
    total_render_wall = 0.0
    book_wall0 = time.time()
    try:
        for ci, (title, chunks, spoken_text) in enumerate(chapters, start=1):
            chap_wav = tmp / f"chapter_{ci:03d}.wav"
            cache_wav = None
            if cache_dir is not None:
                key = _chapter_cache_key(
                    spoken_text, args.voice, args.lang, args.speed, model_tag
                )
                cache_wav = cache_dir / f"{key}.wav"

            reused = False        # served without synthesis (work-dir or cache)
            cache_hit = False     # specifically a content-cache hit
            # 1. Content cache: a previously rendered WAV for this exact spoken
            #    text + voice/lang/speed/model. Copied into the work dir so the
            #    rest of the pipeline (assembly) is unchanged.
            if cache_wav is not None and cache_wav.exists():
                try:
                    if _wav_duration_s(cache_wav) > 0.1:
                        shutil.copyfile(cache_wav, chap_wav)
                        reused = True
                        cache_hit = True
                except Exception:
                    reused = False
            # 2. Resume: a valid chapter WAV from a previous (interrupted) run in
            #    this same work dir is reused so a wedge/kill never loses chapters.
            if not reused and chap_wav.exists():
                try:
                    if _wav_duration_s(chap_wav) > 0.1:
                        reused = True
                except Exception:
                    reused = False
            if reused:
                # advance the global progress bar past this chapter's chunks
                for k in range(len(chunks)):
                    progress.tick(ci, k + 1, len(chunks), label="(cached)")
                n, rw = len(chunks), 0.0
            else:
                n, rw = _render_chunks_to_wav(
                    batch, chunks, vocab, hnsf["weights_sha256"], args.voice,
                    args.speed, chap_wav, tmp, f"ch{ci:03d}",
                    progress=progress, chapter_idx=ci,
                )
                # Save the freshly rendered chapter to the persistent cache.
                if cache_wav is not None:
                    try:
                        tmp_cache = cache_wav.with_suffix(".wav.partial")
                        shutil.copyfile(chap_wav, tmp_cache)
                        tmp_cache.replace(cache_wav)
                    except Exception as e:
                        sys.stderr.write(f"  [cache] could not store {cache_wav.name}: {e}\n")
            if cache_hit:
                cache_hits.append(ci)
            audio_s = _wav_duration_s(chap_wav)
            total_render_wall += rw
            chapter_wavs.append(chap_wav)
            chapter_titles.append(title.rstrip("."))
            progress.newline()
            _tag = " (cache hit)" if cache_hit else (" (cached)" if reused else "")
            print(
                f"  [{ci}/{len(chapters)}] chunks={n} audio={audio_s:.1f}s "
                f"render={rw:.1f}s{_tag} "
                f"({audio_s / max(rw, 0.01):.1f}x rt)",
                flush=True,
            )
    finally:
        batch.close()

    if cache_dir is not None:
        if cache_hits:
            print(
                f"  [cache] {len(cache_hits)}/{len(chapters)} chapter(s) served "
                f"from cache: {', '.join(str(i) for i in cache_hits)}",
                flush=True,
            )
        else:
            print(f"  [cache] 0/{len(chapters)} cache hits (all synthesized)", flush=True)

    _lang_iso = {"a": "eng", "b": "eng", "e": "spa", "es": "spa", "f": "fra", "i": "ita", "p": "por"}.get(args.lang, "und")

    # Assemble the m4b in a worker thread so we can show a live timer: it is the
    # final step after the chunk ETA hits 0, and on a long book it is not free.
    import threading
    _asm_err = []
    _asm_done = threading.Event()

    def _assemble_worker():
        try:
            _assemble_m4b(chapter_wavs, chapter_titles, args.out, args.title,
                          args.artist, _lang_iso)
        except BaseException as e:  # surface in the main thread
            _asm_err.append(e)
        finally:
            _asm_done.set()

    print(flush=True)
    _asm0 = time.time()
    _asm_t = threading.Thread(target=_assemble_worker, daemon=True)
    _asm_t.start()
    while not _asm_done.wait(0.5):
        if sys.stderr.isatty():
            sys.stderr.write(f"\r  Assembling m4b ... ({time.time() - _asm0:.0f}s)\033[K")
            sys.stderr.flush()
    _asm_t.join()
    if sys.stderr.isatty():
        sys.stderr.write("\n")
    if _asm_err:
        raise _asm_err[0]
    assemble_wall = time.time() - _asm0
    book_wall = time.time() - book_wall0
    total_audio = sum(_wav_duration_s(w) for w in chapter_wavs)
    size_mb = args.out.stat().st_size / (1024 * 1024)

    print(
        f"\nDONE: {args.out}\n"
        f"  chapters={len(chapter_wavs)}  total_chunks={total_chunks}\n"
        f"  total_audio={total_audio / 60:.1f} min ({total_audio:.0f}s)\n"
        f"  render_wall={total_render_wall:.1f}s  assemble_wall={assemble_wall:.1f}s  book_wall={book_wall:.1f}s\n"
        f"  render_speed={(total_audio / total_render_wall if total_render_wall else 0.0):.1f}x realtime\n"
        f"  end_to_end_speed={(total_audio / book_wall if book_wall else 0.0):.1f}x realtime\n"
        f"  file_size={size_mb:.1f} MB"
    )
    # The audiobook is written and the stats are flushed. Exit HARD right here:
    # returning up through cmd_book/main hangs (a library -- KPipeline / torch /
    # multiprocessing's resource tracker -- leaves a lingering non-daemon thread),
    # and kab waits on this process, so the hang propagates up the whole chain.
    # The kokoro-bench batch was already reaped in the render loop's finally.
    sys.stdout.flush()
    sys.stderr.flush()
    # A multiprocessing resource tracker (spawned by torch / kokoro) is a separate
    # child python that survives os._exit and keeps the `uv` wrapper -- and thus
    # the waiting kab -- alive. Kill it (and any sibling daemon) before exiting.
    try:
        from multiprocessing import resource_tracker as _rt
        _trk = getattr(_rt, "_resource_tracker", None)
        _pid = getattr(_trk, "_pid", None) if _trk is not None else None
        if _pid:
            os.kill(_pid, signal.SIGKILL)
    except Exception:
        pass
    os._exit(0)
    return {  # unreachable; kept for any in-process caller (the queue spawns a subprocess)
        "out": str(args.out),
        "chapters": len(chapter_wavs),
        "total_chunks": total_chunks,
        "total_audio_s": total_audio,
        "size_mb": size_mb,
    }


def cmd_book(args):
    bench = Path(args.bench)
    if not bench.exists():
        sys.exit(f"kokoro-bench not found: {bench}")
    # Stable work dir next to the output enables resume across interrupted runs.
    if getattr(args, "work_dir", None):
        tmp = Path(args.work_dir)
    else:
        tmp = Path(str(args.out) + ".ane_work")
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        _build_book(args, bench, tmp)
    finally:
        _kill_all_children()  # belt-and-suspenders: nothing survives


# --------------------------------------------------------------------------- #
# Queue + concurrency (multiple books)
# --------------------------------------------------------------------------- #
#
# A book conversion is GPU/ANE+compiler bound: each kokoro-bench batch process
# saturates the Apple Neural Engine and the CoreML on-device compiler. Running
# several at once thrashes that one shared accelerator -- the cure for the hang,
# not a second cause of it -- so the DEFAULT concurrency is 1 (serial drain).
# ``--concurrency N`` is offered for machines that benefit (e.g. very short
# books where G2P dominates), but 1 is the safe, fastest-in-practice choice.
#
# The queue is a single JSON file: a list of job dicts. ``enqueue`` appends;
# ``run`` drains pending jobs, marking each done/failed, and ALWAYS reaps every
# child on exit. Re-running ``run`` resumes where it left off.

_DEFAULT_QUEUE = _ROOT / "outputs" / "ane_queue.json"


def _load_queue(path: Path) -> list:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


def _save_queue(path: Path, jobs: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(jobs, indent=2, ensure_ascii=False))
    tmp.replace(path)


def cmd_enqueue(args):
    qpath = Path(args.queue)
    jobs = _load_queue(qpath)
    job = {
        "id": f"job{int(time.time() * 1000) % 10_000_000}_{len(jobs)}",
        "status": "pending",
        "chapters_dir": args.chapters_dir,
        "glob": args.glob,
        "prepend": args.prepend,
        "out": str(args.out),
        "title": args.title,
        "artist": args.artist,
        "voice": args.voice,
        "lang": args.lang,
        "speed": args.speed,
        "compute_units": args.compute_units,
        "drop_title": args.drop_title,
        "cache_dir": getattr(args, "cache_dir", None),
        "models_dir": args.models_dir,
        "bench": args.bench,
        "repo_id": args.repo_id,
        "enqueued_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    jobs.append(job)
    _save_queue(qpath, jobs)
    print(f"Enqueued {job['id']} -> {args.out}\n  queue: {qpath} ({len(jobs)} job(s))")


def cmd_queue_status(args):
    qpath = Path(args.queue)
    jobs = _load_queue(qpath)
    if not jobs:
        print(f"queue empty: {qpath}")
        return
    for j in jobs:
        print(f"  {j['id']:<20} {j['status']:<8} {j.get('out')}")
    counts = {}
    for j in jobs:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print("  --")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


def _job_to_book_args(job) -> argparse.Namespace:
    return argparse.Namespace(
        chapters_dir=job["chapters_dir"],
        glob=job.get("glob", "capitulo-*.md"),
        prepend=job.get("prepend"),
        out=Path(job["out"]),
        title=job.get("title", "Audiobook"),
        artist=job.get("artist", ""),
        voice=job.get("voice", "af_heart"),
        lang=job.get("lang", "a"),
        speed=job.get("speed", 1.0),
        compute_units=job.get("compute_units", "staged"),
        drop_title=job.get("drop_title", False),
        cache_dir=job.get("cache_dir"),
        models_dir=job.get("models_dir", str(_ROOT / "coreml")),
        bench=job.get("bench", DEFAULT_BENCH),
        repo_id=job.get("repo_id", "hexgrad/Kokoro-82M"),
    )


def _run_one_job(job, qpath):
    """Render one queued job in its own subprocess so a hard crash in one book
    cannot abort the rest of the queue. The child is a plain `book` invocation
    of this same script (which itself reaps its kokoro-bench children)."""
    a = _job_to_book_args(job)
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "book",
        "--chapters-dir", a.chapters_dir,
        "--glob", a.glob,
        "--out", str(a.out),
        "--title", a.title,
        "--artist", a.artist,
        "--voice", a.voice,
        "--lang", a.lang,
        "--speed", str(a.speed),
        "--compute-units", a.compute_units,
        "--models-dir", a.models_dir,
        "--bench", a.bench,
        "--repo-id", a.repo_id,
    ]
    if a.drop_title:
        cmd.append("--drop-title")
    if getattr(a, "cache_dir", None):
        cmd += ["--cache-dir", a.cache_dir]
    for pre in (a.prepend or []):
        cmd += ["--prepend", pre]
    proc = subprocess.Popen(cmd, start_new_session=True)
    _register_proc(proc)
    try:
        rc = proc.wait()
    finally:
        _kill_proc_tree(proc)
    return rc


def cmd_queue_run(args):
    qpath = Path(args.queue)
    concurrency = max(1, args.concurrency)
    if concurrency > 1:
        sys.stderr.write(
            "  [warn] concurrency > 1 shares one ANE/CoreML compiler; "
            "serial (1) is usually faster and avoids compiler thrash.\n"
        )

    def claim_next():
        jobs = _load_queue(qpath)
        for j in jobs:
            if j["status"] == "pending":
                j["status"] = "running"
                _save_queue(qpath, jobs)
                return j["id"]
        return None

    def mark(job_id, status):
        jobs = _load_queue(qpath)
        for j in jobs:
            if j["id"] == job_id:
                j["status"] = status
                j["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _save_queue(qpath, jobs)

    if concurrency == 1:
        while not _INTERRUPTED.is_set():
            job_id = claim_next()
            if job_id is None:
                break
            jobs = _load_queue(qpath)
            job = next(j for j in jobs if j["id"] == job_id)
            print(f"\n=== running {job_id} -> {job['out']} ===", flush=True)
            rc = _run_one_job(job, qpath)
            mark(job_id, "done" if rc == 0 else "failed")
            print(f"=== {job_id} {'done' if rc == 0 else 'FAILED (rc=%d)' % rc} ===", flush=True)
    else:
        import concurrent.futures as cf
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {}
            try:
                while True:
                    while len(futures) < concurrency:
                        job_id = claim_next()
                        if job_id is None:
                            break
                        jobs = _load_queue(qpath)
                        job = next(j for j in jobs if j["id"] == job_id)
                        print(f"\n=== running {job_id} -> {job['out']} ===", flush=True)
                        futures[ex.submit(_run_one_job, job, qpath)] = job_id
                    if not futures:
                        break
                    done, _ = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
                    for fut in done:
                        job_id = futures.pop(fut)
                        rc = fut.result()
                        mark(job_id, "done" if rc == 0 else "failed")
                        print(f"=== {job_id} {'done' if rc == 0 else 'FAILED'} ===", flush=True)
            finally:
                _kill_all_children()
    print("\nqueue drained.", flush=True)


def main():
    _install_signal_handlers()
    ap = argparse.ArgumentParser(description="ANE audiobook pipeline (Kokoro CoreML, batch).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--voice", default="af_heart")
    common.add_argument("--lang", default="a", help="a=American English, e=Spanish, etc.")
    common.add_argument("--models-dir", default=str(_ROOT / "coreml"))
    common.add_argument("--bench", default=DEFAULT_BENCH)
    common.add_argument("--repo-id", default="hexgrad/Kokoro-82M")
    common.add_argument("--speed", type=float, default=1.0)
    common.add_argument("--compute-units", default="staged")

    sp = sub.add_parser("synth", parents=[common], help="single text file -> WAV")
    sp.add_argument("--text-file", required=True, type=Path)
    sp.add_argument("--out", required=True, type=Path)
    sp.set_defaults(func=cmd_synth)

    bp = sub.add_parser("book", parents=[common], help="chapter markdown dir -> m4b")
    bp.add_argument("--chapters-dir", required=True)
    bp.add_argument("--glob", default="capitulo-*.md")
    bp.add_argument("--prepend", action="append", default=None,
                    help="front-matter md file spoken first (e.g. copyright-epub.md); repeatable")
    bp.add_argument("--out", required=True, type=Path)
    bp.add_argument("--title", default="Audiobook")
    bp.add_argument("--artist", default="")
    bp.add_argument("--drop-title", action="store_true",
                    help="do NOT speak the chapter heading (markers only)")
    bp.add_argument("--work-dir", default=None,
                    help="stable scratch dir for resume (default: <out>.ane_work)")
    bp.add_argument("--cache-dir", default=None,
                    help="persistent per-chapter audio cache keyed by spoken-text "
                         "content hash; reuses unchanged chapters across runs")
    bp.set_defaults(func=cmd_book)

    # --- queue: enqueue / status / run -------------------------------------- #
    eq = sub.add_parser("enqueue", parents=[common],
                        help="add a book conversion to the queue")
    eq.add_argument("--queue", default=str(_DEFAULT_QUEUE))
    eq.add_argument("--chapters-dir", required=True)
    eq.add_argument("--glob", default="capitulo-*.md")
    eq.add_argument("--prepend", action="append", default=None)
    eq.add_argument("--out", required=True, type=Path)
    eq.add_argument("--title", default="Audiobook")
    eq.add_argument("--artist", default="")
    eq.add_argument("--drop-title", action="store_true")
    eq.add_argument("--cache-dir", default=None,
                    help="persistent per-chapter audio cache dir (see `book`).")
    eq.set_defaults(func=cmd_enqueue)

    qs = sub.add_parser("queue-status", help="show queued jobs")
    qs.add_argument("--queue", default=str(_DEFAULT_QUEUE))
    qs.set_defaults(func=cmd_queue_status)

    qr = sub.add_parser("queue-run", help="drain the queue (serial by default)")
    qr.add_argument("--queue", default=str(_DEFAULT_QUEUE))
    qr.add_argument("--concurrency", type=int, default=1,
                    help="parallel books (default 1; >1 thrashes the shared ANE)")
    qr.set_defaults(func=cmd_queue_run)

    args = ap.parse_args()
    rc = 0
    try:
        args.func(args)
    except KeyboardInterrupt:
        rc = 130
    # Reap children, but never let a wedged kill -- or a library's lingering
    # non-daemon thread (KPipeline / torch / multiprocessing resource tracker) --
    # block the exit. kab waits on this process, so a hang here hangs the whole
    # chain. Cap the reap with a watchdog, then hard-exit.
    _reaper = threading.Thread(target=_kill_all_children, daemon=True)
    _reaper.start()
    _reaper.join(5.0)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _kill_all_children()
        sys.stderr.write("\ninterrupted -- all children terminated.\n")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    # Hard exit. Some libraries (KPipeline / torch / multiprocessing's resource
    # tracker) leave a non-daemon thread alive that would otherwise hang the
    # process long after the audiobook is written, so kab (which waits on this
    # child) never returns. main()'s finally already reaped every child and the
    # output is flushed, so exiting now is safe and immediate.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
