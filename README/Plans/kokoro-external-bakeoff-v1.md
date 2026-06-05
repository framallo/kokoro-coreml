# Kokoro External Bakeoff Plan

**Date:** 2026-06-04
**Status:** Planned

> Internal bakeoff methodology lives in `README/Plans/kokoro-bakeoff-v2.md`.
> This plan extends that methodology to external Apple Silicon Kokoro
> implementations. Internal Config F results from `README/Notes/bakeoff-results-v2.md`
> serve as our baseline — no re-running of internal configs required.

## Executive Summary

Run a latency comparison of our Swift+CoreML pipeline (Config F, RTF ~0.017 on
M2 Ultra) against the two leading external Kokoro-on-Apple-Silicon
implementations — **Blaizzy/mlx-audio** (7k stars, MLX/GPU) and
**mlalma/kokoro-ios** (256 stars, Swift/CoreML) — across all three botnet fleet
machines. Produce a head-to-head RTF table per machine, audio quality
spot-check, and a performance-notes entry. mlx-audio uses the GPU path so we
expect significantly higher RTF; kokoro-ios uses CoreML like us and will reveal
the implementation quality gap within the same hardware path.

## Problem Statement

- **Symptom:** No controlled, apples-to-apples latency comparison exists between
  our CoreML/ANE pipeline and the MLX-based implementations that dominate
  community mindshare.
- **Root Cause:** External repos were not available when the internal bakeoff
  was designed; our harness is internal-only.
- **Impact:** We cannot make confident public claims about our performance
  advantage without reproducible numbers against the real competition.

## Competitor Inventory

| Handle | Repo | Stars | Framework | HW Target | Notes |
| --- | --- | ---: | --- | --- | --- |
| **mlx-audio** | github.com/Blaizzy/mlx-audio | 7,186 | MLX | GPU (Apple Silicon) | Broad TTS/STT/STS library; actively maintained; Kokoro is one engine |
| **kokoro-ios** | github.com/mlalma/kokoro-ios | 256 | Swift/CoreML | ANE/GPU | Native Swift; same hardware path as ours; reveals impl quality gap |

**Cut:** gabrimatic/kokoro-mlx (6 stars, 1 fork — not community-relevant).
**Cut:** TTS.cpp (GGML, 241 stars, stalled Oct 2025), pykokoro (ONNX, 4 stars) — not Apple Silicon–optimized.

## Goals and Non-Goals

### Goals

- [ ] Establish median end-to-end RTF for mlx-audio and kokoro-ios on each of
      the three fleet machines (m2-studio, irvine-m1, m2-air).
- [ ] Use identical input texts and voice (af_heart or closest equivalent) as
      our internal bakeoff inputs (3s / 7s / 15s / 30s canonical strings).
- [ ] Use identical methodology: N=5 warm iterations, median wall time, same
      timing boundaries (function call → audio array ready, not file write).
- [ ] Spot-check audio quality: confirm voice/prosody parity is close enough
      for a fair comparison.
- [ ] Write results into `README/Notes/performance-notes.md` with an
      external-competitor table following existing note style.

### Non-Goals

- Re-running our internal Config F — use existing bakeoff-results-v2.md numbers.
- Benchmarking TTS.cpp, pykokoro, ONNX variants, or non-Kokoro engines.
- Automating the external installs into the production Kokoro worker or the
  existing `bakeoff_harness.py` harness (adapters are standalone scripts).
- Audio quality objective scoring (PESQ, MCD) — listening spot-check is
  sufficient for this round.
- Any changes to production worker LaunchAgents, env, or model bundles.

## Scope and Constraints

- **Scope:** Three machines × two external impls × four input lengths × N=5
  warm iterations = 120 timed synthesis calls.
- **Constraints:**
  - Each external impl must live in a **separate, isolated uv virtualenv**
    that does not touch the production `botnet` worker env or our `uv.lock`.
  - No persistent changes to `/Users/mm/Documents/GitHub/kokoro-coreml` model
    outputs or production LaunchAgent plists.
  - MLX requires Apple Silicon — no Intel fallback needed.
  - Fleet machines are production workers; benchmarks must not saturate CPU/ANE
    during production TTS queue drain. Run during low-traffic windows.
- **Guardrails:**
  - Do not install anything system-wide with `pip install --user` or `brew`.
  - Do not modify `.env` or LaunchAgent plists on fleet machines.
  - Adapter scripts must be safe to delete after the run.

## Ground Truth Contracts

- **Timing boundary:** Wall time starts immediately before the Python
  synthesis API call and ends when the audio array/bytes object is in memory.
  File I/O, audio playback, and warm-up calls are excluded.
- **Warmup policy:** First call on a fresh process is warmup; timing starts on
  call 2 and we collect N=5 timed calls. Report median.
- **Input identity:** Use the exact text strings from
  `outputs/bakeoff/input_manifest.json` (the same texts as Config A/D/E/F).
  If an external impl tokenizes differently and produces different audio
  lengths, record the actual audio duration as denominator for RTF.
- **Voice:** `af_heart` (our canonical voice). If an external impl does not
  have `af_heart`, use the closest female American English voice and document
  the substitution.

## Already Shipped (Do Not Re-Solve)

- **Internal bakeoff harness:** `scripts/bakeoff_harness.py` — not modified.
- **Config F numbers:** `README/Notes/bakeoff-results-v2.md` — the authoritative
  baseline. M2 Ultra: 57ms / 124ms / 239ms / 476ms (3s/7s/15s/30s).
- **Input manifest:** `outputs/bakeoff/input_manifest.json` — canonical texts
  and expected audio durations. Read, do not regenerate.
- **Fleet SSH access:** `mm@m2-studio.local`, `mattmireles@irvine-m1.local`,
  `mattmireles@M2-Air.local` — established in botnet reference.

## Fresh Baseline (Current State)

Internal Config F medians (M2 Ultra, warm):

| Input | Audio | F Wall (ms) | F RTF |
| --- | ---: | ---: | ---: |
| 3s | 2.80s | 57 ms | 0.020 |
| 7s | 6.75s | 124 ms | 0.018 |
| 15s | 13.90s | 239 ms | 0.017 |
| 30s | 27.38s | 476 ms | 0.017 |

No external numbers exist yet.

## Solution Overview

Write thin Python adapter scripts for each external implementation that:
1. Load the exact same input texts from the manifest.
2. Call each impl's synthesis API inside a timing bracket.
3. Run N=5+1 iterations (drop first), record wall times.
4. Output a JSON result file compatible enough to report alongside bakeoff-v2
   numbers in performance-notes.md.

Run each adapter on each machine via SSH. Collect JSON results locally.
Write a results section in performance-notes.md.

```
input_manifest.json
        |
        v
scripts/external_bakeoff/
  run_mlx_audio.py        ← Blaizzy/mlx-audio adapter (Python)
  run_kokoro_ios.py       ← mlalma/kokoro-ios driver (calls Swift CLI)
  KokoroIOSBench/         ← small Swift package wrapping kokoro-ios
  summarize_external.py   ← read JSONs, emit RTF table
        |
        v
outputs/external_bakeoff/
  results_mlx_audio_<machine_id>.json
  results_kokoro_ios_<machine_id>.json
        |
        v
README/Notes/performance-notes.md   ← new external-competitor section
```

> kokoro-ios is a Swift package, not a Python library — it gets a thin Swift
> CLI wrapper (like our `kokoro-bench`) rather than a Python pip install.

## Implementation Phases

### Phase 0: Research and API Audit

**Goal:** Understand the exact Python API surface of each external impl before
writing any adapter, so we time the right thing and pick the right voice.

**Tasks:**

- [ ] On the operator Mac, install mlx-audio in an isolated env and run a
      test synthesis call. Confirm the API: what function, what args, what
      return type (numpy array? bytes? file path?).
      ```bash
      uv venv .venv-mlx-audio && source .venv-mlx-audio/bin/activate
      uv pip install mlx-audio
      python -c "import mlx_audio; help(mlx_audio)"
      ```
- [ ] Audit mlalma/kokoro-ios: clone the repo, read its Swift API surface
      (`Sources/`), understand how to invoke synthesis and what it returns.
      Confirm it compiles on macOS (not iOS-only target). Identify the voice
      selection mechanism.
      ```bash
      git clone https://github.com/mlalma/kokoro-ios /tmp/kokoro-ios
      ls /tmp/kokoro-ios/Sources/
      ```
- [ ] Confirm which voice identifier corresponds to `af_heart` in each impl,
      or document the closest substitute.
- [ ] For mlx-audio: confirm GPU (not CPU fallback) via `sudo powermetrics
      -i 1000 --samplers gpu_power | grep GPU` during a synthesis call.
- [ ] For kokoro-ios: confirm ANE/GPU participation via `powermetrics` or
      Instruments during a test synthesis call.
- [ ] Record exact versions (pip freeze for mlx-audio; git SHA for kokoro-ios).

**Verification:** Can synthesize one sentence with each impl from the operator
Mac. Timing boundary is understood. Voice choice is documented.

---

### Phase 1: Write Adapter Scripts

**Goal:** Two standalone adapter scripts that produce timing JSON in a format
we can report. Created in `scripts/external_bakeoff/`.

**Tasks:**

- [ ] Create `scripts/external_bakeoff/run_mlx_audio.py`:
  - Reads `outputs/bakeoff/input_manifest.json` for the 3s/7s/15s/30s texts.
  - Loads mlx-audio model once before timing loop.
  - Runs 1 warmup + 5 timed calls per input text.
  - Times from `time.perf_counter()` before synthesis call to after audio
    array is in memory (not file write).
  - Records machine_id (`--machine-id` CLI flag), library version, voice,
    per-iteration wall times.
  - Writes `outputs/external_bakeoff/results_mlx_audio_<machine_id>.json`.

- [ ] Create `scripts/external_bakeoff/KokoroIOSBench/` — a minimal Swift
  package with kokoro-ios as a dependency:
  - `Package.swift` declaring the kokoro-ios dependency at a pinned git SHA.
  - `Sources/main.swift`: reads `--input-key` (3s/7s/15s/30s) and text from
    stdin, runs 1 warmup + 5 timed synthesis calls, prints JSON timing to
    stdout. Same interface pattern as `swift/Sources/KokoroBenchmark/main.swift`.
  - Build: `swift build -c release --product KokoroIOSBench`.

- [ ] Create `scripts/external_bakeoff/run_kokoro_ios.py`:
  - Thin Python driver that invokes the `KokoroIOSBench` binary for each
    input text, collects stdout JSON, aggregates into the standard result file.
  - Writes `outputs/external_bakeoff/results_kokoro_ios_<machine_id>.json`.

- [ ] Create `scripts/external_bakeoff/summarize_external.py`:
  - Reads all result JSONs in `outputs/external_bakeoff/`.
  - Emits a markdown table: impl × machine, median RTF per input length.
  - Include our Config F numbers from bakeoff-results-v2.md as a hardcoded
    reference row (no re-running).

- [ ] Create `scripts/external_bakeoff/requirements_mlx_audio.txt` with
      pinned versions from Phase 0.

**Verification:** Run both adapter scripts locally on the operator Mac.
Both produce valid JSON. Summarizer emits a readable markdown table.

---

### Phase 2: Deploy and Run on Fleet

**Goal:** Collect results from all three fleet machines.

**Tasks:**

- [ ] Choose a low-traffic window (check `pnpm check:tts-worker-health` on
      each host first; do not run during active queue drain).
- [ ] For each fleet machine (m2-studio, irvine-m1, m2-air):
  - SSH in.
  - `rsync` or `scp` the `scripts/external_bakeoff/` directory to the machine.
  - Create isolated venvs and install deps from requirements txt files.
  - Run both adapter scripts with the correct `--machine-id`.
  - `scp` result JSON files back to the operator Mac into
    `outputs/external_bakeoff/`.
  - Remove the venvs and build artifacts after collection
    (`rm -rf .venv-mlx-audio scripts/external_bakeoff/KokoroIOSBench/.build`).

- [ ] Verify no production worker disruption: check worker freshness after each
      machine's run with `pnpm check:tts-worker-health`.

**Notes on m2-air (fanless):** The M2 Air thermal notes
(`gist/README/Notes/infrastructure/m2-air-kokoro-thermal-soak-notes.md`)
show that sustained parallel load raises heat. Run only one impl at a time,
sequentially, with a 60s cooldown between runs. Do not run the external
bakeoff on m2-air while its production Kokoro worker is active.

**Verification:** 6 result JSON files exist in `outputs/external_bakeoff/`
(mlx-audio + kokoro-ios × 3 machines). Each has 5 timing entries per input
length. No production worker downtime during the window.

---

### Phase 3: Quality Spot-Check

**Goal:** Confirm the MLX outputs are the same voice / similar quality to ours
so the comparison is fair. We are not running PESQ/MCD — listening is enough.

**Tasks:**

- [ ] For each external impl, synthesize the 7s input text and save to WAV.
- [ ] Compare against our Config F output for the same text (generate locally
      with the Swift CLI or Python harness).
- [ ] Listen and confirm: same voice (`af_heart` or equivalent), no garbling,
      similar prosody. Document voice name and any quality notes.
- [ ] If quality is substantially degraded (robot voice, wrong speaker, missing
      phonemes), flag it in the results note — a speed advantage on garbage
      output is not a fair comparison.

**Verification:** Short note written for each impl confirming voice parity or
documenting any prosody caveat.

---

### Phase 4: Document Results

**Goal:** Add a new section to `README/Notes/performance-notes.md` with the
external-competitor comparison, following the existing note style.

**Tasks:**

- [ ] Run `summarize_external.py` to produce the final markdown table.
- [ ] Add a section to `performance-notes.md`:
  - Section title: "External Bakeoff: our CoreML pipeline vs mlx-audio (MLX/GPU) and kokoro-ios (CoreML)"
  - Machine and library provenance
  - End-to-end wall time table (all impls, all machines)
  - RTF table
  - Speedup: our Config F vs each external impl
  - Quality caveat if any
  - 2–4 bullet interpretation
- [ ] Use `git-commit` to commit the plan, adapter scripts, and notes.
      Do NOT commit JSON files in `outputs/` — they are git-ignored.

**Verification:** `performance-notes.md` has the new section. Someone reading
it could reproduce the benchmark from the adapter scripts and requirements files.

---

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] Results cover all three fleet machines for both primary impls.
- [ ] Each impl × machine × input produces N=5 timed calls; median is reported.
- [ ] Timing boundary is identical to our internal methodology (call start →
      audio in memory).
- [ ] No production worker disruption confirmed post-run.
- [ ] Adapter scripts are checked into the repo with pinned requirements.
- [ ] Quality spot-check is documented.

### Definition of Done

- [ ] `performance-notes.md` external-competitor section committed.
- [ ] Adapter scripts and requirements files committed.
- [ ] This plan updated to `Status: Complete`.

## Open Questions

### Unresolved

- **Q:** Does mlx-audio Kokoro's `af_heart` voice produce audio comparable
  enough in quality to ours that an RTF comparison is fair?
  **Options:** (A) Same voice string → probably yes; (B) Different voice ID
  but similar quality → fair with caveat; (C) Clearly different speaker →
  note in results, comparison is still valid for latency but not quality.
  **Lean:** Resolve in Phase 0/3 by listening.

- **Q:** Does m2-air's production queue need to be paused during the external
  bakeoff run?
  **Options:** (A) Run during low-traffic window and monitor; (B) Explicitly
  disable TTS worker for the duration.
  **Lean:** Option A. Check queue depth before each run; abort and reschedule
  if the worker is active.

- **Q:** Does kokoro-ios compile cleanly as a macOS target (not iOS-only)?
  **Options:** (A) Yes, macOS target works out of the box; (B) Needs minor
  target/manifest tweaks.
  **Lean:** Resolve in Phase 0 by attempting a local build.

### Resolved

- **Q:** Should we extend the existing bakeoff_harness.py?
- **A:** No. External impls have incompatible envs and APIs. Standalone adapter
  scripts are simpler, safer, and non-destructive to the existing harness.

- **Q:** Which external implementations are in scope?
- **A:** Blaizzy/mlx-audio (7,186 stars, MLX/GPU, actively maintained) and
  mlalma/kokoro-ios (256 stars, Swift/CoreML, same hardware path as ours).
  gabrimatic/kokoro-mlx cut (6 stars — not community-relevant).
  TTS.cpp (GGML) and pykokoro (ONNX) out — not Apple Silicon–optimized.

## References

### Internal

- [Bakeoff v2 Plan](kokoro-bakeoff-v2.md) — internal harness design
- [Bakeoff Results v2](../Notes/bakeoff-results-v2.md) — our Config F baseline
- [Performance Notes](../Notes/performance-notes.md) — where results go
- [M2 Air Thermal Notes](../../gist/README/Notes/infrastructure/m2-air-kokoro-thermal-soak-notes.md) — fanless constraints
- [Botnet Reference](../../botnet/.claude/skills/botnet/reference.md) — fleet SSH targets and ops

### External

- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios)

## Files to Create

| File | Change Type | Notes |
| --- | --- | --- |
| `scripts/external_bakeoff/run_mlx_audio.py` | Create | Blaizzy adapter (Python) |
| `scripts/external_bakeoff/run_kokoro_ios.py` | Create | mlalma driver (calls Swift CLI) |
| `scripts/external_bakeoff/KokoroIOSBench/Package.swift` | Create | Swift package wrapping kokoro-ios |
| `scripts/external_bakeoff/KokoroIOSBench/Sources/main.swift` | Create | Swift timing CLI |
| `scripts/external_bakeoff/summarize_external.py` | Create | table generator |
| `scripts/external_bakeoff/requirements_mlx_audio.txt` | Create | pinned pip deps |
| `README/Notes/performance-notes.md` | Modify | add external section |
| `README/Plans/kokoro-external-bakeoff-v1.md` | Create | this plan |

## Risks and Mitigations

- **MLX API changes between versions:** Pin versions in requirements txt. If
  the API breaks post-install, document and move on — the comparison is still
  valid for whatever version was pinned.
- **m2-air thermal pressure:** Sequential runs with cooldown; abort if
  `pmset -g therm` shows throttling before all runs complete.
- **Production queue disruption:** Check worker freshness before and after each
  machine run. If a worker goes stale, restart it before declaring the
  benchmark done.
- **External impls produce different audio lengths for the same text:**
  Record actual audio duration as RTF denominator, not the nominal "3s" label.
  The manifest already has precise durations from our own runs.

---

> SIMPLER IS BETTER. Standalone adapter scripts, not a unified harness.
> Config F numbers come from existing bakeoff-results-v2.md, not a re-run.
