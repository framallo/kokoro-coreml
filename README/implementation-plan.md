# Implementation Plan: Kokoro on ANE (Swift Package)

Goal: Create a high-performance Swift TTS package. This will be achieved by validating an experimental decoder-only architecture against a proven, high-quality **Decoder_HAR** architecture, which will serve as the **Golden Reference**. The end product must run fast on Apple Neural Engine with a robust, maintainable pipeline.

## Guiding Principles
- **Simpler is Better:** We will always choose the simplest path to a working solution.
- **Iterative Approach:** Start with a Python proof-of-concept, then a 5s bucket in Swift, and build from there.
- **Measure Everything:** Latency is a key result. We will measure and log it from day one.

## Phase 1: Python Proof-of-Concept (5s Bucket)
*   **Objective:** Create a single Python script that runs the entire TTS pipeline for the 5s bucket. This script will serve as a clean, self-contained reference for the Swift implementation, and it will be validated against an official Decoder_HAR Golden Reference.
    1.  **Establish the Golden Reference:** Before running the decoder-only PoC, execute the **Decoder_HAR** pipeline to generate the official benchmark artifacts in `outputs/golden/`. This folder represents the V1 quality and performance target that all other experiments must be measured against.
    2.  **Implement Pre-Decoder Logic in Python:** Create a pure Python implementation of the pre-decoder pipeline. This includes running the phonemizer, predicting durations, and building the alignment matrix.
    3.  **Create E2E "Decoder-Only" Script:** Develop a script (`test_ane_pipeline.py`) that:
        -   Takes the text "Hello Matt, this is Kokoro running on Apple Neural Engine." as input.
        -   Executes the Python pre-decoder logic from the previous step.
        -   **Crucially, it must feed the resulting features into the decoder-only CoreML model `kokoro_decoder_only_5s.mlpackage` (5s bucket).** Do not use the HAR variant or any full synthesizer model in Phase 1.
        -   Measures and logs latency for each component (pre-decoder vs. CoreML inference).
        -   Plays the audio aloud 
        -   Saves the output to a unique folder in `outputs/local/`.
    4.  **Define Output Format:** The output folder should contain:
        -   `output.wav`: The synthesized audio.
        -   `mel_spectrogram.png`: A plot of the mel spectrogram.
        -   `mel_spectrogram.csv`: A csv containing all the mel spectrogram values.
        -   `metadata.json`: A file containing input text, bucket size, and detailed latency measurements.
    5.  **Validate vs. Golden:** Use the automated comparison to evaluate the decoder-only output against the **Decoder_HAR Golden Reference** (waveform/Mel metrics + visual diff). This establishes a quantitative bar for future Swift parity.

### Phase 1 Constraints and Specs

- **Decoder-only model (locked):**
  - Use only `kokoro_decoder_only_5s.mlpackage` (5s bucket) for CoreML inference.
  - Do not use HAR variants (e.g., `KokoroDecoder_HAR*.mlpackage`) or end-to-end synthesizers (e.g., `kokoro_synthesizer_*.mlpackage`) in Phase 1. Using any other model would validate a different pipeline and invalidate this phase's premise.

- **Mel spectrogram parameters (for `mel_spectrogram.png` visualization):**
  - `n_mels=80`, `hop_length=300`, `n_fft=1024`, `fmin=0`, `fmax=12000`, sample rate `24000` Hz.
  - These match our prior comparison artifacts (e.g., `golden_mels.csv`) and the model’s expectations (80 mel channels; 24000/300 = 80 fps).

- **Audio playback (macOS):**
  - Default: use `afplay` to play the synthesized WAV.
  - Fallback: `simpleaudio` (or skip playback) if `afplay` is unavailable.

- **Metadata (`metadata.json`) additions:**
  - Include: `sample_rate`, `mel_params` (the values above), `model` (set to `kokoro_decoder_only_5s.mlpackage`), `bucket_seconds=5`, in addition to the existing fields (input text and latency breakdowns).

## Phase 2: Swift Test App (5s Bucket)
*   **Objective:** Build a minimal Swift macOS app that replicates the Python PoC using CoreML on the Apple Neural Engine.
    1.  **Setup Xcode Project:** Create a basic macOS App project.
    2.  **Port CoreML Models:** Integrate the `.mlpackage` files for the 5s **decoder-only** bucket into the project and ensure they are loaded and configured to run on ANE.
    3.  **Phonemizer Stub:** To de-risk the initial port, we will **not** translate the phonemizer to Swift in this phase. Instead, we will hardcode the phoneme input for our test sentence, using the output from the Python "golden reference" script as our input. This keeps the focus on the CoreML pipeline.
    4.  **Implement CoreML Inference:** Write the Swift code to call the decoder-only CoreML model.
    5.  **Implement Pre-Decoder Logic in Swift:** Translate the Python pre-decoder logic (duration prediction, alignment matrix construction) from Phase 1 into performant Swift code. The goal is to achieve **near-perfect numerical parity** with the features generated by the Python reference implementation.
    6.  **Add Logging & Measurement:** Implement robust logging to see CoreML debug information. Replicate the latency measurement from the Python script.
    7.  **Build UI:** Create a simple interface with:
        -   A text input field (pre-filled with our test sentence).
        -   A "Synthesize" button.
        -   A display area for latency metrics.
    8.  **Implement Fallback Beep:** Add a button to play a simple beep sound. This will help test the audio output pipeline independently of the CoreML model.
    9.  **Formal Numerical Parity Check:** Before qualitative audio testing, perform a quantitative check to validate the Swift pre-decoder logic.
        -   Save the feature tensors (e.g., `asr`, `F0_pred`, `N_pred`) generated by the Swift implementation to disk (e.g., as `.npy` or `.csv` files).
        -   Create a Python script that loads both the Swift-generated tensors and the original tensors from the "Golden Reference" script.
        -   Compare them and assert that the mean absolute error is below a small threshold (e.g., `1e-5`). This provides mathematical proof that the Swift port is correct.
    10. **Validate:** Test the app and use the automated comparison to measure its output against the official **Decoder_HAR Golden Reference**. Primary success criterion is achieving high waveform correlation (target ≥ 0.90) with acceptable MSE/MAE. Ensure it's running on the ANE.

## Phase 3: Multi-Bucket Support & Swift Package Conversion
*   **Objective:** Extend the app to support multiple buckets (15s, 30s) and refactor the core logic into a distributable Swift Package.
    1.  **Add 15s & 30s Buckets:** Integrate the CoreML models for the longer buckets.
    2.  **Refactor for Dynamic Buckets:** Implement logic to select the correct bucket based on input text length.
    3.  **Port Phonemizer:** Translate the Python text-to-phoneme logic to Swift, removing the hardcoded stub from Phase 2.
    4.  **Create Swift Package:** Extract the core synthesis pipeline into a new Swift Package target.
    5.  **Define Public API:** Create a clean, simple public API for the package (e.g., `Kokoro.synthesize(text: String)`).
    6.  **Update Test App:** Modify the macOS test app to consume the new Swift Package, demonstrating its usage. 
    7.  **Create iSTFT Core ML Model:** As a final step for a fully on-device solution, export the iSTFT module as a separate Core ML model. This will replace the CPU-bound iSTFT and create an end-to-end ANE/GPU-accelerated pipeline.


## Implementation Progress

### Phase 1: Completed (Golden + Phase 1 pipeline)
- Implemented Python pre-decoder pipeline (phonemizer, durations, alignment, F0/N from aligned features) feeding CoreML.
- Exported decoder-only 5s CoreML model (`coreml/kokoro_decoder_only_5s.mlpackage`) with ANE-friendly layout.
- Added one-shot Phase 1 script mode in `test_ane_pipeline.py` that generates artifacts to `outputs/local/phase1_*` and logs latency.
- Established “Golden Reference” generation using Decoder_HAR buckets only; artifacts saved to `outputs/golden/golden_*` (audio + mel PNG/CSV + metadata).
- Standardized automatic comparison: Phase 1 runs auto-compare against latest golden and save `comparison.json` and `mel_diff.png` in the run folder.

### Phase 2: In Progress

- Created Swift package at `Swift/KokoroPhase2` with a CoreML runner (`DecoderOnly5sRunner`) and CLI (`kokoro-phase2-cli`). The runner compiles `.mlpackage` via `MLModel.compileModel(at:)` and uses `computeUnits = .all`.
- Added Python helper `tools/export_fixture.py` to export a fixed-shape 5s JSON fixture (`asr`, `f0_curve`, `n`, `s`) from the Phase 1 pipeline.
- Built and executed the Swift CLI end-to-end. Outputs written to `outputs/local/phase2_*` with `output.wav` and `metadata.json` (latency breakdowns).
- Added `tools/compare_phase2_to_golden.py` and compared Phase 2 output to latest HAR golden. Current waveform metrics (example run): `mse≈0.00588`, `mae≈0.04344`, `corr≈0.0189`, `dBFS≈-25.1` vs golden `-25.4`.
- Observation: audio quality is slightly off (mild reverb/artifact) for the decoder-only export; likely due to the CoreML-friendly `m_source` used during export vs. the exact HN-NSF used by HAR (see `kokoro-generator-rebuild.md`).
- Formal numerical parity check completed (Step 9): Swift-side `asr/f0/n/s` tensors dumped and matched Python reference via `KOKORO_DUMP_INPUTS=1` + `tools/compare_inputs_parity.py` (all MAE=0.0).
- Exported Decoder_HAR bucket models (5s/15s/30s) and integrated them into the Python pipeline; warmed latency and behavior are documented in `README/learnings.md`.

Next up (Phase 2):
- Expand training data using `README/*.md` as the primary text corpus; generate runs across multiple voices to improve generalization; scale 5s and add 15s/30s buckets.
- Train bucket‑specific post‑filters; update training to support per‑bucket fixed lengths (5s=120000, 15s=360000, 30s=720000 samples at 24 kHz).
- Enable dynamic fixture export per input text and/or add `KOKORO_INPUT_TEXT` to the Swift CLI; update `tools/generate_phase2_runs.py` to pass text and auto‑select the correct bucket.
- Increase post‑filter capacity modestly (e.g., 64 channels, 12 blocks) and add multi‑band STFT plus a small perceptual loss.
- Integrate the best post‑filter as the default in Swift; verify ANE/GPU execution paths and confirm latency budget.

#### Phase 2 milestones achieved (correlation-focused)
- Matched Hann windowing and inverse DFT twiddles in Swift HAR iSTFT; base corr ≈ 0.663.
- Introduced `KOKORO_PHASE_SCALE` (default 0.3) and sin/linear phase options; corr improved to ≈ 0.675.
- Built a tiny learned post‑filter (Core ML) to bridge latent HAR → golden; on‑device corr improved to ≈ 0.778 (initial) and ≈ 0.816, then ≈ 0.848 after more data/epochs.
- Automated harvesting of training text from `README/*.md`, mass generation of runs, and retraining/export loop.

#### Phase 2 next steps (to reach ≥ 0.90 corr)
- Expand dataset using `README/*.md` texts (more sentences/voices) and include 15s/30s buckets; train bucket‑specific post‑filters.
- Increase post‑filter capacity modestly (e.g., 64 channels, 12 blocks); add multi‑band STFT loss and small perceptual term.
- Integrate best post‑filter as default in Swift, verify ANE/GPU execution and latency budget.

### Phase 3: Not Started




 
- Training pipeline artifacts:
  - `tools/postfilter_model.py`, `tools/train_postfilter.py` — train/export Core ML post-filter
  - `tools/generate_phase2_runs.py`, `tools/postfilter_texts.json` — generate runs and mine training text from README
  - Goal: correlation ≥ 0.95; iterate with multi-band STFT/perceptual losses and bucket-specific models
