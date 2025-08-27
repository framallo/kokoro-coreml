# Learnings from Kokoro TTS Core ML Conversion

This document captures the key challenges and solutions discovered while converting the Kokoro TTS model from PyTorch to Core ML for on-device inference on Apple Silicon.

## 1. The Core Problem: Dynamic Shapes vs. Static Graphs

The fundamental challenge was the model's heavy reliance on dynamic operations that are incompatible with Core ML's requirement for a static or predictably dynamic computation graph.

- **`torch.full` with Dynamic Inputs**: The tracer failed when creating tensors with shapes derived from dynamic inputs (e.g., `torch.full((1, input.shape[1]), ...)`).
  - **Solution**: Replace with traceable equivalents like `torch.ones_like(input).sum()`.
- **`torch.repeat_interleave`**: This was the primary blocker. The model creates an alignment matrix whose shape depends on the *values* inside the predicted duration tensor. This is impossible to represent in a static graph.
- **`pack_padded_sequence`**: The LSTMs used this for handling variable-length sequences, which is not supported by the Core ML tracer.

## 2. The Solution: A Two-Stage, Bucketed Architecture

A direct, one-to-one conversion was not feasible. The winning strategy was to re-architect the *inference pipeline* without changing the core model weights, splitting the model into two parts and using bucketing for the final stage.

### Stage 1: The `DurationModel` (Dynamic)

- **Responsibility**: Runs the expensive Transformer and LSTMs to predict phoneme durations and extract intermediate hidden states.
- **Implementation**:
  - Takes `input_ids`, `ref_s`, `speed`, and an `attention_mask` as inputs.
  - All inputs with a sequence dimension use `ct.RangeDim` to allow for variable-length text.
  - **Key Fixes**:
    - **Monkey-Patching**: We created CoreML-friendly versions of the `TextEncoder` and `DurationEncoder` in the export script. These custom modules remove the `pack_padded_sequence` calls and run the LSTMs directly on the padded tensors.
    - **BERT Buffer Removal**: We programmatically deleted the `buffered_token_type_ids` from the `AlbertModel` instance before tracing to prevent a `slice` error. The `token_type_ids` were then passed in as an input during the forward pass.
- **Output**: A set of tensors containing the predicted durations and the hidden states needed for synthesis.
- **Result**: A single, flexible `.mlpackage` that runs efficiently on the ANE.

### Stage 2: The `SynthesizerModel` (Fixed-Size Buckets)

- **Responsibility**: Takes the intermediate features and a pre-built alignment matrix and generates the final audio waveform.
- **Implementation**:
  - We created multiple `SynthesizerModel`s, each one compiled for a **fixed-size** audio output (e.g., 3s, 5s, 10s, 30s). This is known as **bucketing**.
  - By using fixed-size inputs for the alignment matrix, we completely remove the dynamic shape problem that was blocking the conversion.
- **Output**: A fixed-length audio waveform.
- **Result**: A set of highly optimized `.mlpackage` files, one for each bucket, that run entirely on the ANE.

## 3. The Client's Role: The Conductor

The complexity that was removed from the model graph is now managed by the native Swift client code. The client is responsible for:
1. Running the `DurationModel` once.
2. Summing the predicted durations to determine the final audio length.
3. Selecting the appropriate `SynthesizerModel` bucket.
4. Building the alignment matrix on the CPU (a fast, simple operation).
5. Padding the matrix to the bucket's fixed size.
6. Calling the selected `SynthesizerModel`.
7. Trimming any padding silence from the end of the final audio buffer.

## 4. Key Takeaways

- **Simpler is Better**: When faced with an impossible conversion, don't fight the tools. Redesign the *pipeline*, not the model.
- **Divide and Conquer**: Isolate dynamic, data-dependent logic from the heavy, parallelizable math.
- **CPU is Not the Enemy**: Offloading small, complex operations (like building the alignment matrix) to the CPU is a valid and powerful strategy that unlocks the ANE for the 99% of work that matters.
- **Monkey-Patching is a Powerful Tool**: For stubborn models, modifying the model instance in-memory during the export process is a clean way to fix incompatible layers without forking the original library.
- **Bucketing Beats Dynamic Hell**: When a model's output is fundamentally dynamic, creating a few fixed-size versions is often the most pragmatic path to a shippable, high-performance solution.

## 5. Export Tooling Challenges and Resolutions

- **Tracing Hangs with torch.jit.trace**: The original tracing tool often entered infinite loops or hung indefinitely when dealing with the model's complex architecture, especially in custom layers like AdainResBlk1d. This was due to its inability to handle dynamic behaviors and large graphs efficiently.
- **Switch to torch.export**: Moving to the modern torch.export API resolved the hanging issues, as it is designed for more complex models. It provided faster failures with actionable error messages, allowing for targeted fixes.
- **TRAINING Dialect Error**: Even in eval mode, dropout layers caused the graph to retain training operations. Recursively replacing nn.Dropout with nn.Identity before export created a pure inference graph.
- **Import and Typo Issues**: Small errors like missing imports or calling modules instead of functions caused quick failures. These emphasized the need for careful code review in iterative debugging.
- **Debug Strategy**: Adding timed print statements and using Ctrl+C to interrupt hangs provided stack traces that pinpointed problematic operations. Force-quitting via Activity Monitor was essential for stuck processes.
- **Overall Lesson**: When old tools fail silently, switch to modern alternatives. Combine surgical model modifications (like removing dropout) with the right export API to succeed. Persistence and fast iteration beat deep research when debugging tooling issues.
- **TRAINING Dialect Error in coremltools.convert**: During Synthesizer export with torch.export, coremltools rejected the graph with a 'Provided Dialect: TRAINING' error, even after model.eval() and basic dropout removal. This indicates residual training operations persisting in the exported program.
  - **Resolution**: Enhance the remove_dropout function to include logging for each replacement, recursive eval() calls, and requires_grad_(False) to fully strip training hints. If no dropouts are found, add a warning to check for other training-mode modules like BatchNorm.
- **torch.export Hangs and Instability**: torch.export sometimes hung for minutes before failing, especially on complex graphs like the Synthesizer's LSTMs and matrix ops.
  - **Resolution**: Fallback to torch.jit.trace with strict=False for a simpler, more reliable export that produces cleaner graphs compatible with CoreML's ANE optimizations. Validate post-export with Instruments to ensure full ANE usage.
- **Version Compatibility Warnings**: Untested Torch versions (e.g., 2.7.1) with coremltools led to potential instability.
  - **Resolution**: Downgrade to tested versions like Torch 2.5.0 and coremltools 7.x in a fresh environment before retrying exports.

- **Virtual Environment (Venv) Hell**: The environment setup was a major blocker. Issues included:
  - `pip` failing because a specified beta version (`coremltools==7.0b5`) from a guide was unavailable for the target architecture.
  - Running scripts with an absolute path to the wrong venv's Python interpreter, ignoring the activated environment.
  - Pasting multi-line commands with comments into the shell, causing errors.
  - **Resolution**: Switched to a stable, available version of `coremltools` (e.g., `7.2`). Used a single, clean, multi-command line with `&&` to handle venv creation, activation, and dependency installation without user error. Always run scripts with just `python script_name.py` inside an activated venv.

- **`NameError` on `example_inputs`**: A simple but fatal bug where the tuple of example tensors for `torch.jit.trace` was not defined before being used, causing an immediate crash.
  - **Resolution**: Defined `example_inputs` on the line immediately before the `torch.jit.trace` call.

- **Process Killed During Tracing**: `torch.jit.trace` was silently killed by the OS, likely due to excessive memory usage when tracing a large model with massive dummy inputs (e.g., a `72000`-frame tensor).
  - **Resolution**: Temporarily reduce the `trace_length` and other tensor dimensions during debugging to get a faster, less resource-intensive trace. Using `check_trace=False` can also help the tracer be more lenient with dynamic-looking operations.

- **FP32 Tracing to FP16 Conversion**: The most stable path to an ANE-compatible model was to keep the PyTorch model and inputs in `float32`, trace it, and then convert to Core ML with `compute_precision=ct.precision.FLOAT16`.
  - **Resolution**: Removed all `.half()` calls before tracing. Ensured all `ct.TensorType` dtypes were `np.float32`. Set `compute_precision` in `ct.convert` to `ct.precision.FLOAT16` for the final, optimized model.

## 6. Baseline Performance (CPU/PyTorch Fallback) — 2025‑08‑18

This baseline was captured before enabling GPU/ANE acceleration. The hybrid pipeline fell back to pure PyTorch (CPU) because the CoreML vocoder was not yet integrated.

- **Environment**:
  - Hardware: Mac Studio (Model Identifier: Mac14,14), Apple M2 Ultra (24‑core CPU: 16P + 8E), 64 GB RAM
  - Software: macOS 15.6 (24G84), Darwin 24.6.0
  - Torch 2.5.0, coremltools 8.3.0
  - Acceleration: MPS/GPU and ANE not used (CPU fallback)
- **Method**:
  - Ran `kokoro-coreml/test_ane_pipeline.py` which generates and times several sentences; saved WAVs to `kokoro-coreml/outputs/`.
  - Voice: `af_heart`, speed: `1.0`, sample rate: 24 kHz.
- **Results** (synthesis time vs. audio duration; lower RTF is faster):
  - "Hello world!": 2.994 s compute for 1.550 s audio → RTF ≈ 1.93× (overhead‑dominated)
  - "The quick brown fox …": 1.768 s for 3.250 s → RTF ≈ 0.54× (faster than real‑time)
  - Longer sentence A: 2.158 s for 5.950 s → RTF ≈ 0.36×
  - Longer sentence B: 2.378 s for 6.000 s → RTF ≈ 0.40×
- **Takeaway**: Even on CPU, typical sentences are already sub‑real‑time. Short clips look slower due to fixed startup overhead.

### Immediate Optimization Plan
1. Enable GPU (MPS) for PyTorch components to reduce latency 2–3×.
2. Finish CoreML vocoder export and integrate it (ANE acceleration) for an additional ~30–50% overall speedup.
3. Add duration‑>bucket selection + alignment build in Swift to drive the CoreML synthesizer models.
4. Verification: Instruments Core ML template (Neural Engine activity) and `sudo powermetrics -i 1000 --samplers ane`.

## 7. Vocoder Export Breakthrough — 2025‑08‑18

- Problem: Converter errored on `multiply` inside harmonic/noise source of `Generator.m_source` and had f0/asr temporal mismatches when tracing generator-only.
- Fix:
  - Forced full `Decoder` export to keep F0/N alignment correct.
  - Introduced a minimal `DummySource` to replace `generator.m_source` during export, returning zeros for harmonic and noise sources. This avoids unsupported ops while preserving shapes.
  - Used FP32 input dtypes with `minimum_deployment_target=macOS13` and `compute_precision=FLOAT16` for ANE-friendly weights.
- Result: Successful Core ML conversion of vocoder to `kokoro-coreml/KokoroVocoder.mlpackage`. Next step is to validate ANE usage and objective audio quality vs PyTorch baseline.
- Result: Successful Core ML conversion of vocoder to `kokoro-coreml/coreml/KokoroVocoder.mlpackage` (output name currently `var_2778`; will remap to `waveform` at integration time). Initial audio revealed timbre issues due to a simplified source; replaced with a multi‑harmonic CoreML‑friendly source (cumsum/sin over overtones) and added overlap‑add stitching to reduce seam artifacts. Next: implement exact hn‑nsf source via MIL custom ops to match PyTorch parity.

### Reproduce Baseline Audio Locally
Outputs saved by the baseline run:

```
kokoro-coreml/outputs/sample_01.wav
kokoro-coreml/outputs/sample_02.wav
```

Quick one‑off generation (PyTorch path):

```bash
/Users/mattmireles/Documents/GitHub/talktome/.venv-coreml/bin/python - <<'PY'
from kokoro import KPipeline
import soundfile as sf
text = "TalkToMe is speaking using Kokoro. This sounds pretty good."
pipeline = KPipeline(lang_code='a')
for _, _, audio in pipeline(text, voice='af_heart', speed=1.0):
    sf.write('out.wav', audio, 24000)
    break
print('Wrote out.wav')
PY
```

## 8. GPU (MPS) Benchmark — 2025‑08‑18

Ran a quick pass forcing PyTorch to use Apple GPU (MPS) for Kokoro’s PyTorch path.

- Command (from `kokoro-coreml/`):

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /Users/mattmireles/Documents/GitHub/talktome/.venv-coreml/bin/python - <<'PY'
import os, time, torch, soundfile as sf
from kokoro import KPipeline
from kokoro.model import KModel
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
pipeline = KPipeline(lang_code='a', model=False)
model = KModel().to(device).eval()
texts = [
  "Hello world!",
  "The quick brown fox jumps over the lazy dog.",
  "This is a longer sentence that will test the performance of our pipeline running on the Apple GPU.",
]
os.makedirs('outputs_mps', exist_ok=True)
for i, text in enumerate(texts, 1):
  voice='af_heart'; phonemes=None
  for _, ps, _ in pipeline(text, voice=voice, speed=1.0): phonemes=ps; break
  ref_s = pipeline.load_voice(voice)[len(phonemes)-1].to(device)
  t0=time.time();
  with torch.no_grad(): audio = model(phonemes, ref_s, speed=1.0)
  dt=time.time()-t0
  a = audio.cpu().numpy(); sr=24000
  sf.write(f'outputs_mps/sample_mps_{i:02d}.wav', a, sr)
  L=len(a)/sr; print(f"{L:.3f}s audio in {dt:.3f}s → RTF {dt/L:.3f}x")
PY
```

- Results (M2 Ultra, macOS 15.6, Torch 2.5.0):
  - Hello world!: 1.550 s in 11.724 s → RTF ≈ 7.564× (slower than CPU; dominated by fallbacks/transfer)
  - Quick brown fox: 3.250 s in 4.958 s → RTF ≈ 1.526× (slower than real‑time)
  - Longer sentence: 6.575 s in 4.127 s → RTF ≈ 0.628× (faster than real‑time)

- Observation: `aten::angle` not supported on MPS, falls back to CPU in `istftnet.py`; mixed MPS↔CPU execution adds overhead, hurting short/medium inputs. MPS is not a net win here without removing fallbacks or moving to CoreML.

- Action: Prioritize CoreML vocoder (ANE) integration; continue synthesizer bucketing; keep PyTorch on CPU (or isolate ops) to avoid MPS<->CPU ping‑pong in interim.

## 9. CoreML Decoder_HAR Buckets + Latency — 2025‑08‑19

- Exported Decoder_HAR bucket models at 5s/15s/30s that accept exact hn‑nsf features from PyTorch (`har_spec`, `har_phase`).
- Implemented single‑shot and grouped bucket decoding in `test_ane_pipeline.py` with 10% overlap and Hann crossfades; inverse STFT remains in PyTorch for fidelity.
- End‑to‑end latency on a ~23.7 s utterance (user text) — warm, averaged over 5 runs:
  - 5s bucket: ~1.350 s total (RTF ≈ 0.057)
  - 15s bucket: ~1.413 s total (RTF ≈ 0.060)
  - 30s bucket: ~1.380 s total (RTF ≈ 0.058)
- Breakdown (typical warmed share): ANE (CoreML predict) ≈ 0.25–0.31 s; CPU prep (hn‑nsf + STFT) ≈ 0.15–0.17 s; inverse STFT ≈ 0.02–0.03 s; remainder orchestration/IO/overlap ≈ 0.55–0.60 s.

Key learnings:
- Larger buckets reduce CoreML call overhead and overlap tax; 15–30 s perform similarly for ~24 s clips. 5 s is slower due to more windows and crossfades.
- Warmup matters: once models are hot, user‑visible wait per long clip drops to ~1.3–1.4 s.
- Keeping hn‑nsf exact in PyTorch preserves quality while we iterate on Core ML fidelity; a composite operator rebuild of `generator.m_source` remains a post‑V1 option.


## 10. Key Architectural and Debugging Lessons — 2025-08-19

### Lesson 1: The Peril of Architectural Ambiguity (Decoder_Only vs. Decoder_HAR)
Our biggest challenge was the confusion between two competing architectures: the experimental Decoder-Only path and the proven Decoder_HAR path. The project stalled until we formally recognized from the documentation that Decoder_HAR was the V1 success story.

**Actionable Insight**: A project must have a single, explicitly defined target architecture. Without this clarity, debugging becomes nearly impossible as the team may be trying to fix a component that isn't even part of the correct, final design.

### Lesson 2: The Power of a "Golden Reference" Pipeline
We established that the most effective way to validate the on-device pipeline is to first build a "Golden Reference" script in Python.

**Actionable Insight**: This script should not use the original PyTorch models. It must load the converted `.mlpackage` files and execute them. This provides an unimpeachable, end-to-end benchmark for correctness, audio quality, and performance that the Swift implementation must match. It validates the model conversion process itself before any Swift code is written.

### Lesson 3: The "Feature Mismatch" Root Cause (Linguistic vs. Acoustic)
The final "whispering" bug was not a simple error; it was a fundamental data-type mismatch. We discovered that the `asr` tensor (derived from `t_en`) is a 512-channel linguistic embedding. A vocoder, however, expects an ~80-channel acoustic representation (a mel-spectrogram in a logarithmic scale). Our final, conclusive diagnosis was that the F0/N predictor must be fed features derived from asr (t_en), not en (d).

**Actionable Insight**: Simply aligning linguistic features does not make them a valid input for a vocoder. There is a critical, intermediate "decoder" step that converts linguistic features to an acoustic spectrogram. Bypassing this step will always result in unintelligible audio. This was the key technical insight that explained why the Decoder-Only path was failing.

### Lesson 4: The Value of Quantitative Feature Analysis
We only confirmed the "Feature Mismatch" when you ran the Python script to compare the numerical stats of your generated `asr.csv` and the `golden_mels.csv`. The massive discrepancies in shape, min/max values, and energy correlation were the undeniable proof.

**Actionable Insight**: Do not rely on subjective listening alone. When debugging quality issues in an ML pipeline, write scripts to compare the statistical properties of the intermediate tensors against a known-good reference. Quantitative analysis is faster and more reliable than subjective feedback.

### Lesson 5: The Final Bottleneck (The iSTFT)
Our final analysis revealed that even with the correct Decoder_HAR model, the final Inverse STFT step is a critical trade-off. A manual Swift iSTFT is fast but risks numerical errors (causing whispers), while a Python-bridge iSTFT is accurate but too slow.

**Actionable Insight**: The optimal production architecture is a three-stage Core ML pipeline:

1.  Duration Model
2.  Decoder_HAR Model (on ANE) to produce a latent tensor.
3.  A final, tiny iSTFT Model (on GPU) to convert the latent tensor to a waveform.

This keeps the entire process in the high-performance Swift/Core ML ecosystem while ensuring bit-for-bit accuracy.

### Lesson 6: The ref_s Zero-Vector Bug

**The Bug**: Feeding a zero-vector for the `ref_s` (style reference) input does not produce a "neutral" voice; it produces noise. The model interprets this as an out-of-distribution input and fails to generate a coherent signal.

**The Learning**: The `ref_s` input is mandatory for generating a stable, listenable voice, even in a decoder-only pipeline. Always provide a valid, pre-trained voice embedding.

### Lesson 7: The ANE's Silent Performance Fallback

**The Bug**: Some synthesis runs were inexplicably slow (~37 seconds), despite using the Decoder_HAR model that was previously benchmarked as extremely fast (~1-2 seconds).

**The Learning**: The Apple Neural Engine has a hard limit of 16,384 for any single tensor dimension. If any internal layer in a model exceeds this limit, Core ML will silently fall back to the much slower GPU or CPU for that operation, causing a catastrophic performance drop without any crash or obvious error. This must be fixed in the model exporter.

### Lesson 8: The AdaIN Identity Substitution Trap

**The Bug**: To solve an export issue, the AdaIN (Adaptive Instance Normalization) layers, which are critical for applying voice style, were replaced with an identity function. This allowed the model to convert but resulted in a robotic, "alien" sounding voice.

**The Learning**: Critical model components like styling layers cannot be removed without destroying the output quality. The correct solution is not to remove them, but to re-implement their mathematical function using a sequence of simpler, ANE-friendly primitives (a "Composite Operator").

## 11. Phase 2 Swift Parity and Decoder-Only Artifact — 2025-08-27

- We added a Swift Package (`Swift/KokoroPhase2`) and a CLI that runs the decoder-only 5s CoreML model using fixtures exported from the Python pipeline.
- Using `KOKORO_DUMP_INPUTS=1`, the CLI dumps `asr.csv`, `f0_curve.csv`, `n.csv`, `s.csv`. A Python checker confirmed exact numerical parity (MAE=0.0) between Swift inputs and the Python fixture. This conclusively proves the pre-decoder feature prep in Swift is correct.
- Despite input parity, decoder-only audio exhibits a slight reverb/artifact. Root cause is likely the CoreML-friendly source replacement used during decoder-only export (to avoid unsupported ops). Conclusion: the quality issue is not from features; it’s from the export’s source approximation.

Implications:
- Prefer Decoder_HAR as the Golden architecture for V1. The source is computed exactly in PyTorch and passed into CoreML, preserving timbre.
- If decoder-only is required, either (a) re-export with exact source (custom MIL op) or (b) split source generation to CPU and pass it as an input tensor to CoreML. The latter is simpler and aligns with “CPU is Not the Enemy.”

## 12. HAR On-Device Post-Processing in Swift — 2025-08-27

- The Decoder_HAR CoreML model outputs a latent tensor with interleaved log-magnitude and phase channels. To match Python, Swift must apply `exp` to magnitude channels and `sin` to phase channels, then perform an inverse STFT with the correct `n_fft`, hop, and Hann windowing.
- We implemented an Accelerate/vDSP inverse STFT in Swift. Parameter inference is derived from CoreML input shapes (e.g., for 5s: `har_spec` C=11, T=24001 → `n_fft≈800`, `hop≈300`). This produces a proper waveform without resorting to a Python bridge.
- Takeaway: Adding a tiny, deterministic DSP stage in Swift is tractable and unlocks a fully on-device Golden Reference path.




## 13. Correlation as the Primary Fidelity Metric — 2025-08-27

- Correlation between candidate and golden waveforms is the most sensitive indicator of structural fidelity (1.0 = perfect match). MSE/MAE can be dominated by gain; correlation tracks waveform shape.
- Decoder-only CoreML runs showed very low correlation (≈0.02–0.14) despite perfect input parity (MAE=0.0 for asr/f0/n/s), implicating the decoder-only export’s source approximation rather than feature prep.
- Switching to the Decoder_HAR path with a corrected Swift inverse STFT (including 1/N IFFT scaling and proper onesided-to-twosided reconstruction) increased correlation to ≈0.66 vs golden. Amplitude differences (dbFS mismatch) did not materially affect correlation.
- Added diagnostic toggles in Swift HAR post-processing:
  - `KOKORO_USE_RAW_PHASE=1` to bypass sin() on phase (corr ≈0.62–0.66 in tests)
  - `KOKORO_PACKING=interleaved` to support alt channel layouts
  - `KOKORO_DISABLE_HALF_SCALE=1` to test mirror-scaling hypothesis
- Conclusion: For V1, prioritize the HAR path. Decoder-only requires exact source parity (custom op or CPU-side source) to reach high correlation.

- HAR correlation stabilized around ~0.66 after fixing 1/N IFFT scaling. Least-squares gain alignment confirmed correlation invariance to amplitude (gain ~0.92). Further improvements likely require exact PyTorch windowing conventions and bin handling parity.

- Spectral log-magnitude correlation between HAR Swift output and golden is higher (~0.73) than raw waveform (~0.66), suggesting phase integration/OLA details are the main remaining gap. Local 50ms correlations range widely (p10≈0.02, p90≈0.90), indicating phase alignment varies across time.

- Matching torch.hann_window(periodic=True) and negative-angle IFFT twiddles in Swift nudged waveform correlation from ~0.661 to ~0.663. Incremental but consistent; remaining gap likely in exact DC/Nyquist treatment or COLA normalization nuances.

- Added HAR network output dump (CSV + meta) toggle (`KOKORO_DUMP_HAR=1`) in the Swift CLI to aid Python-side parity checks between Swift iSTFT and PyTorch `istft`. Use this to compare per-channel [log-mag|phase] frames and verify packing/normalization.

- Implemented fixture-HAR bypass in Swift and a forced HAR path; initial corr is low because packing/normalization requires exact replication from the model’s internal output. Added `KOKORO_DUMP_HAR` to dump the model’s HAR tensor for frame-wise parity with Swift’s inverse.

- Dumping the model’s HAR tensor and reconstructing with numpy (mirror of Swift) yields high corr vs Swift HAR (~0.81) and good corr vs golden (~0.62). Swift HAR corr vs golden remains slightly higher (~0.66). Conclusion: Swift packing/OLA matches the model’s internal format; the remaining delta to golden likely stems from downstream non-linearities beyond pure ISTFT (e.g., post-filtering) present in the golden path.

- Phase non-linearity/scale matters: reconstructing with sin(phase) or 0.5×phase increases correlation vs golden (e.g., ~0.66→~0.67 in Swift). Added `KOKORO_PHASE_SCALE` to sweep this parameter. This suggests the model’s phase channel isn’t a direct angle; a learned nonlinearity or scaling is applied before inverse STFT in the golden path.

- Phase scale sweep across [0.3..0.7] shows a shallow optimum near 0.3–0.4 (corr ≈ 0.6747 at 0.3). Gains are modest but consistent; we will set default `KOKORO_PHASE_SCALE=0.3` while keeping it tunable.

- Current best (5s fixture): corr ≈ 0.6747 (HAR Swift with phase_scale=0.3). Next: validate on additional sentences and 15s/30s buckets to ensure robustness; if stable, encode this into model export or Swift defaults.

- On-device post-filter (Core ML) now runs in Swift path; 5s fixture correlation improved from ~0.675 to ~0.778. Next: expand training data (multi-sentence, 15s/30s buckets) and increase model capacity/epochs to target >0.9.

- After expanding training pairs and epochs, the on-device post-filter lifted correlation to ~0.816 on the 5s fixture. Action items: add multi-sentence/bucket data, modestly increase capacity (e.g., 64 channels, 12 blocks), and add a multi-band STFT loss to target >0.9. Verify performance on-device (ANE/GPU vs CPU) and latency impact.
