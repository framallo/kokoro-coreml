# Kokoro Audio Parity Recovery Plan

**Date:** 2026-04-16
**Status:** Planned

## Executive Summary

Recover human-sounding Kokoro audio before doing any more performance claims. The
current post-update Core ML and Swift bakeoff samples are unproven and the
available waveform statistics show near-inactive output compared with the known
HAR-post demo, so this plan first creates listenable reference samples, then
bisects the Python-to-Swift/Core ML runtime until the first semantic audio
divergence is identified and fixed.

## Problem Statement

- **Symptom:** WAVs in `outputs/bakeoff/listen/` from the bakeoff winner do not
  sound like human speech. Objective waveform checks agree: the samples are
  mostly near-silence with spikes, not dense voiced audio.
- **Root Cause:** Not yet proven for the current Swift + Core ML winner. Existing
  notes already prove a related failure mode: Core ML cannot faithfully carry the
  `SourceModuleHnNSF` / `SineGen` harmonic source path, and the full-decoder
  export was not stock Kokoro because `IdentityAdaIN` replaced real AdaIN.
- **Impact:** Any bakeoff result after the big update is invalid as a quality or
  product claim until the generated audio is proven human-sounding against a
  reference pipeline.

## Mode Definitions

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| Reference generation | Produce PyTorch and known-good HAR-post WAVs plus objective reports. | Gives the user listenable samples quickly and anchors all thresholds. |
| Stage parity | Run identical tensors through Python, Core ML, and Swift boundaries. | Finds the first layer where speech semantics are lost. |
| Recovery implementation | Apply the smallest fix to the first divergent boundary. | Avoids rewriting the pipeline around an assumed bug. |
| Bakeoff reinstatement | Re-run performance comparisons only after audio proof passes. | Prevents speed numbers from masking broken synthesis. |

## Required Skills

Use these skills around this plan and during execution:

| Skill | When to use | Required output |
| --- | --- | --- |
| `create-plan` | Authoring or materially restructuring this checked-in plan. | A plan under `README/Plans/` that follows the canonical template and is grounded in repo notes and guides. |
| `markdown` | Any edit to this plan or related README material. | Clean markdown with real links, the plan template structure, and no duplicate prose. |
| `audit-fix-loop` | Self-auditing this plan document, or auditing a completed fix set after implementation. | Findings fixed and re-audited until architecture, correctness risk, and complexity debt are all A. |
| `debug` | Primary execution workflow for root-cause investigation and fix proof. | Reads relevant `README/Guides` and `README/Notes`, proves the fix, then ends with a consolidated note via `write-notes`. |
| `ilya-sutskever` | Architecture judgment for Core ML conversion, CPU/ANE split, and parity gates. | Keeps the recovery simple: dynamic DSP on CPU/Swift, conv-heavy math on Core ML, empirical proof before claims. |
| `phase-audit` | After each phase is implemented. | Findings-first review against this plan, guide alignment, tests, and checkbox accuracy. |
| `write-notes` | Final step after the root cause is confirmed. | Updates `README/Notes/debug-notes.md` or the right consolidated note with the cause, fix, commands, and remaining risks. |
| `execute-plan-hardcore` | Optional end-to-end executor after this plan is approved. | Implements phases, audits each phase, commits per phase, then runs audit-to-A. |
| `bakeoff` | Only after the audio quality gates pass. | Produces performance data for known-human audio, not broken waveforms. |

Do not use `bakeoff` as a correctness signal. Do not treat `audit-fix-loop` as a
replacement for the stage-parity investigation; it is appropriate for auditing
this plan document or for a completed fix set.

## Goals and Non-Goals

### Goals

- [ ] Produce reference WAVs the user can listen to before changing model code.
- [ ] Freeze the failing bakeoff WAVs and record objective waveform and
      spectrogram evidence for regression comparison.
- [ ] Establish a stage-parity harness that can compare Python reference tensors
      with the Swift + Core ML winner at every boundary.
- [ ] Identify and fix the first semantic audio divergence in the current
      post-update runtime.
- [ ] Replace weak listen-sample validation with gates derived from PyTorch and
      known-good HAR-post references.
- [ ] Re-run the bakeoff only after short and medium samples sound human.

### Non-Goals

- Optimizing RTF, ANE utilization, memory, or package size before speech quality
  is recovered.
- Publishing or relying on post-update bakeoff rankings before audio proof.
- Rebuilding the whole Kokoro pipeline unless stage parity proves the current
  split cannot be salvaged.
- Accepting waveform non-emptiness, duration agreement, or low RMS checks as
  proof of speech quality.

## Scope and Constraints

- **Scope:** Python reference generation, waveform inspection, stage parity,
  Swift/Core ML runner instrumentation, minimal correctness fixes, quality gates,
  and final bakeoff rerun.
- **Constraints:** Generated WAVs, tensor dumps, reports, and model artifacts
  stay under `outputs/` and remain gitignored unless explicitly promoted.
- **Constraints:** The first shippable recovery should prefer the existing
  HAR-post split: CPU/Swift harmonic source and Core ML `GeneratorFromHar`.
- **Guardrails:** Preserve existing model-loading and export entry points:
  `HybridTTSPipeline.extract_vocoder_inputs()`,
  `build_decoder_har_post_inputs_np()`, `export_synth.main`, and the Swift
  `KokoroPipeline` package structure.

## Ground Truth Contracts (Do Not Violate)

- **Human quality baseline:** Full PyTorch Kokoro from
  `examples/example_synthesis.py --engine pytorch` is the primary speech
  reference.
- **Known-good local comparator:** `outputs/decoder_har_post_demo.wav` is a
  useful comparator only after its provenance is recorded in the recovery report.
- **Current bakeoff samples are failing evidence:** Files under
  `outputs/bakeoff/listen/` are regression artifacts, not acceptable output.
- **Post-update outputs are unproven:** No Core ML or Swift model produced after
  the big update should be treated as speech-capable until it passes this plan's
  reference, parity, and listening gates.
- **Stage parity before architecture claims:** A Core ML or Swift path cannot be
  called correct until identical inputs are compared against the Python reference
  at the same boundary.
- **Quality before speed:** A faster runtime that emits non-human audio fails the
  plan.
- **DSP split stays intentional:** If `SourceModuleHnNSF` / `SineGen` is involved
  in the divergence, keep that path off full-decoder Core ML unless a new
  parity proof shows Core ML can carry it.

## Already Shipped (Do Not Re-Solve)

- **Duration and shape validation fix:** Commit `1e48249` tightened several
  bakeoff acceptance checks, but it did not prove semantic speech quality.
- **HAR-post tensor builder:** `build_decoder_har_post_inputs_np()` in
  [kokoro/synthesis_backends.py](../../kokoro/synthesis_backends.py) is the
  Python single source of truth for `x_pre`, `ref_s`, and `har` geometry.
- **Swift runtime:** [swift/Sources/KokoroPipeline/KokoroPipeline.swift](../../swift/Sources/KokoroPipeline/KokoroPipeline.swift)
  is the current native runtime path for the bakeoff winner.
- **Swift harmonic source implementation:** [swift/Sources/KokoroPipeline/HarmonicSource.swift](../../swift/Sources/KokoroPipeline/HarmonicSource.swift)
  is already intended to keep hn-nsf on CPU/Accelerate.
- **Existing hn-nsf validator:** [scripts/validate_hnsf_swift.py](../../scripts/validate_hnsf_swift.py)
  already generates PyTorch harmonic-source references for Swift comparison.
- **Listen-sample helper:** [scripts/bakeoff_listen.py](../../scripts/bakeoff_listen.py)
  generates Config F WAVs, but its current thresholds are not sufficient proof
  of speech.

## Fresh Baseline (Current State)

- **Known-good comparator:** `outputs/decoder_har_post_demo.wav` is 1.363 s,
  24 kHz mono PCM, RMS about `6973`, active fraction above 32 PCM counts about
  `77.8%`, and zero-crossing rate about `8.9%`.
- **Current failing Config F samples:** `outputs/bakeoff/listen/config_f_3s.wav`,
  `config_f_7s.wav`, `config_f_15s.wav`, and `config_f_30s.wav` have RMS roughly
  `491`, `488`, `300`, and `42`; active fraction above 32 PCM counts roughly
  `2.0%`, `2.2%`, `2.1%`, and `0.12%`; and zero-crossing rate below `0.36%`.
- **Known prior bisect:** [README/Notes/debug-notes.md](../Notes/debug-notes.md)
  reports `pre_generator`, `post_conv`, `stft_transform`, and
  `spectral_head_inverse` near exact, with the first major full-decoder Core ML
  failure at `SourceModuleHnNSF` / `SineGen`.
- **Risky validator gap:** The listen helper currently accepts extremely low
  activity thresholds. It can reject empty files, but it cannot distinguish
  speech from near-silence or impulse noise.

## Solution Overview

```text
Reference WAVs
  PyTorch full path + known HAR-post demo
        |
        v
Objective report
  waveform stats + spectrogram snapshots + provenance
        |
        v
Stage parity ladder
  Python vi -> Swift prefix -> DecoderPre -> hn-nsf/HAR -> GeneratorFromHar
        |
        v
Smallest fix at first divergent boundary
        |
        v
Human listen samples + objective gates
        |
        v
Bakeoff rerun
```

The recovery should be empirical and conservative: first make the broken output
measurable, then compare identical tensors, then patch the earliest bad boundary.

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Freeze Evidence and Produce Listen References

**Goal:** Give the user trustworthy samples immediately and preserve the failing
ones for comparison.

**Tasks:**

- [ ] Create `outputs/audio-parity/` with a manifest containing git commit,
      dirty-tree status, machine info, voice, speed, text, and artifact paths.
- [ ] Generate full PyTorch reference WAVs for the same short inputs used by
      `scripts/bakeoff_listen.py`.
- [ ] Copy or link `outputs/decoder_har_post_demo.wav` into the report set with
      its provenance stated as known-local comparator, not primary truth.
- [ ] Snapshot current failing files from `outputs/bakeoff/listen/` into
      `outputs/audio-parity/failing-current/`.
- [ ] Write `outputs/audio-parity/index.md` listing the reference and failing
      WAV paths for listening.

**Verification:** The user has at least one PyTorch reference WAV and one failing
Config F WAV for the same text, plus a manifest tying both to the repo state.

---

### Phase 1: Add Objective Audio Inspection

**Goal:** Turn "sounds like garbage" into repeatable numeric and visual evidence.

**Tasks:**

- [ ] Add `scripts/audio_quality_probe.py` to report duration, sample rate, RMS,
      DC offset, peak, clipping fraction, active-sample fractions, zero-crossing
      rate, spectral centroid, voiced-band energy, and optional spectrogram PNGs.
- [ ] Run the probe over PyTorch reference WAVs, `outputs/decoder_har_post_demo.wav`,
      and all files under `outputs/bakeoff/listen/`.
- [ ] Store reports under `outputs/audio-parity/reports/`.
- [ ] Derive provisional speech-health thresholds from the reference set; do not
      hard-code thresholds from the broken samples.

**Verification:** The report clearly separates PyTorch or known HAR-post speech
from the current bakeoff outputs without using subjective listening alone.

---

### Phase 2: Build the Stage-Parity Ladder

**Goal:** Compare identical tensors across Python and Swift/Core ML boundaries.

**Tasks:**

- [ ] Extend or add a Python capture script, likely
      `scripts/capture_audio_parity_tensors.py`, that dumps `tokens`, `ref_s`,
      duration output, alignment output, `asr`, `f0`, `n`, `x_pre`, `har`, and
      final waveform for one short input.
- [ ] Extend `swift/Sources/KokoroBenchmark/main.swift` or add a debug subcommand
      to dump the same boundaries from `KokoroPipeline`.
- [ ] Add `scripts/compare_audio_parity_tensors.py` with shape, dtype, max error,
      cosine similarity, and correlation checks.
- [ ] Include the existing `scripts/validate_hnsf_swift.py generate` and Swift
      harmonic-source comparison in the ladder.

**Verification:** One command can show the first failing boundary between Python
reference and the current Swift + Core ML path for the short sample.

---

### Phase 3: Diagnose the First Divergence

**Goal:** Decide which subsystem is responsible before applying a fix.

**Tasks:**

- [ ] Compare `HybridTTSPipeline.extract_vocoder_inputs()` output with Swift
      tokenization, duration, alignment, and `asr` construction.
- [ ] Compare Python `F0Ntrain` outputs against Swift/Core ML `f0` and `n`.
- [ ] Compare Python `DecoderPre` `x_pre` against Swift/Core ML `x_pre` with the
      same padded `asr`, `f0`, `n`, and `ref_s`.
- [ ] Compare Python `build_decoder_har_post_inputs_np()` `har` against Swift
      `buildHar()` with the same padded F0 and hn-nsf weights.
- [ ] Feed Python reference `x_pre`, `ref_s`, and `har` into Swift
      `GeneratorFromHar` Core ML and compare final waveform.

**Verification:** The investigation names exactly one earliest divergent boundary
or records a ranked list if two boundaries fail independently.

---

### Phase 4: Apply the Smallest Correctness Fix

**Goal:** Restore human speech with minimal architecture churn.

**Tasks:**

- [ ] If Swift hn-nsf or STFT diverges, fix
      [swift/Sources/KokoroPipeline/HarmonicSource.swift](../../swift/Sources/KokoroPipeline/HarmonicSource.swift)
      against PyTorch references and strengthen
      [scripts/validate_hnsf_swift.py](../../scripts/validate_hnsf_swift.py).
- [ ] If `DecoderPre` or `GeneratorFromHar` package geometry diverges, fix the
      wrapper or export path in [export_synth/wrappers.py](../../export_synth/wrappers.py)
      and [export_synth/convert.py](../../export_synth/convert.py), then
      re-export only the affected packages.
- [ ] If Swift input preparation diverges, fix
      [swift/Sources/KokoroPipeline/KokoroPipeline.swift](../../swift/Sources/KokoroPipeline/KokoroPipeline.swift)
      or [swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift](../../swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift)
      and add focused Swift tests.
- [ ] If identical known-good tensors still produce garbage from Core ML
      `GeneratorFromHar`, quarantine the current package and route through the
      last known-good HAR-post artifact while rebuilding the export.

**Verification:** The first divergent boundary now passes parity on the short
sample, and generated WAVs sound recognizably human before moving on.

---

### Phase 5: Replace Weak Listen Gates

**Goal:** Prevent future bakeoff winners from passing with non-human audio.

**Tasks:**

- [ ] Update [scripts/bakeoff_listen.py](../../scripts/bakeoff_listen.py) to use
      thresholds derived in Phase 1 and to emit the full audio-quality report.
- [ ] Add tests for silence, impulses, clipped output, and a known-good reference
      fixture where practical.
- [ ] Add a manifest field that marks listen samples as `quality_pass: true` only
      when both duration and speech-health gates pass.
- [ ] Document that these gates are smoke tests, not replacements for human
      listening or tensor parity.

**Verification:** The current failing Config F files fail the new gate, while the
PyTorch and known-good HAR-post references pass.

---

### Phase 6: Re-run Quality Proof and Bakeoff

**Goal:** Restore the benchmark only after audio quality is demonstrably sane.

**Tasks:**

- [ ] Regenerate listen samples for short and medium inputs.
- [ ] Have the user listen to the new samples before treating the fix as done.
- [ ] Run `pytest` at the repo root and focused Swift tests under
      `swift/Tests/KokoroPipelineTests/`.
- [ ] Run the `bakeoff` skill only after the samples pass listening and objective
      gates.
- [ ] Update [README/Notes/debug-notes.md](../Notes/debug-notes.md) through
      `write-notes` with cause, fix, commands, and residual risks.

**Verification:** The user confirms at least short and medium outputs sound
human, objective gates pass, and the bakeoff results are regenerated from the
fixed path.

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] At least one PyTorch reference WAV and one fixed Swift/Core ML WAV are
      available for the same text and voice.
- [ ] The current failing `outputs/bakeoff/listen/config_f_*.wav` files fail the
      new quality gate.
- [ ] The fixed output passes stage parity at the previously divergent boundary.
- [ ] Short and medium fixed samples sound human to the user.
- [ ] Performance bakeoff results are not regenerated or cited until quality
      gates pass.

### Definition of Done

- [ ] Plan phases are checked off only after `phase-audit` verifies each phase.
- [ ] `pytest` passes or any unrelated failure is documented with evidence.
- [ ] Focused Swift tests pass for any touched Swift code.
- [ ] `README/Notes/debug-notes.md` records the final root cause and fix.
- [ ] Final bakeoff artifacts include `quality_pass: true` for listen samples.

## Open Questions

### Resolved

- **Q:** Should performance tuning continue before audio quality is fixed?
- **A:** No. Quality proof gates all further bakeoff claims.

- **Q:** Should the plan assume the current bug is the same
  `SourceModuleHnNSF` / `SineGen` Core ML failure from earlier notes?
- **A:** No. That failure is a strong prior and informs the CPU/Swift split, but
  the current Swift + Core ML winner still needs fresh stage parity.

### Unresolved

- **Q:** Is `outputs/decoder_har_post_demo.wav` exactly from the older
  `decoder_har_post_bucket_impl()` path or from another local experiment?
- **Options:** Confirm from artifact metadata if present, regenerate from the
  older path, or treat it as a useful but secondary comparator.

- **Q:** Can objective gates catch every bad speech sample?
- **Options:** Use gates as smoke tests, add mel/MCD/PESQ-style metrics if
  feasible, and keep human listening as a required final check.

## References

### Internal

- [Debug Notes](../Notes/debug-notes.md)
- [Kokoro generator rebuild notes](../kokoro-generator-rebuild.md)
- [Kokoro to Core ML conversion](../Kokoro-to-CoreML-conversion.md)
- [Core ML conversion guide](../coreml-conversion-guide.md)
- [Core ML compute-unit scheduling guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [Plan workflow skills guide](../Skills/plan-workflow-skills-guide.md)
- [Phase audit rubric](../Skills/phase-audit-rubric.md)
- [Kokoro bakeoff v2 plan](kokoro-bakeoff-v2.md)

### Local Artifacts

- `outputs/decoder_har_post_demo.wav`
- `outputs/bakeoff/listen/config_f_3s.wav`
- `outputs/bakeoff/listen/config_f_7s.wav`
- `outputs/bakeoff/listen/config_f_15s.wav`
- `outputs/bakeoff/listen/config_f_30s.wav`

## Degradation and Rollback

- **If Swift recovery remains blocked:** Use Python `build_decoder_har_post_inputs_np()`
  for hn-nsf/HAR generation and Core ML only for `GeneratorFromHar` until Swift
  parity is proven.
- **If current packages are corrupt:** Quarantine the package set and re-export
  from the last known-good wrapper path.
- **If objective thresholds overfit:** Keep them as smoke gates, lower their
  authority, and require tensor parity plus human listening before success.

## Monitoring and Observability

- `audio_quality.active_fraction_32` - rejects near-silent speech outputs.
- `audio_quality.rms_pcm` - catches implausibly low energy.
- `audio_quality.zero_crossing_rate` - catches spike-only or DC-heavy output.
- `audio_quality.voiced_band_energy_ratio` - catches output with no speech-band
  structure.
- `parity.correlation` and `parity.max_abs_error` per stage - locate the first
  divergent boundary.
