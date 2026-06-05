# External Bakeoff Adapters

Disposable adapters for `README/Plans/kokoro-external-bakeoff-v1.md`.

## Runtime Inputs

```bash
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/bakeoff_harness.py prepare-inputs
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/external_bakeoff/prepare_runtime_inputs.py
```

The runtime manifest must contain `3s`, `7s`, `10s`, `15s`, and `30s`, and each
input must route to its named bucket.

## MLX

Use a disposable venv and a pinned clone:

```bash
uv venv /tmp/kokoro-external-bakeoff/mlx-venv
uv pip install --python /tmp/kokoro-external-bakeoff/mlx-venv/bin/python \
  -r scripts/external_bakeoff/requirements_mlx_audio.txt
/tmp/kokoro-external-bakeoff/mlx-venv/bin/python \
  scripts/external_bakeoff/run_mlx_audio.py --machine-id m2-studio \
  --spotcheck-dir outputs/external_bakeoff/spotcheck_wavs/mlx_audio_m2-studio
```

## Soniqo Speech Swift

Clone and pin `soniqo/speech-swift` outside the repo, then run:

```bash
python scripts/external_bakeoff/run_speech_swift_kokoro.py \
  --machine-id m2-studio \
  --speech-swift /tmp/kokoro-external-bakeoff/speech-swift \
  --spotcheck-dir outputs/external_bakeoff/spotcheck_wavs/soniqo_speech_swift_kokoro_m2-studio
```

## Laishere Core ML Backup

Clone and convert `laishere/kokoro-coreml` outside the repo. The converter can
take a long time while Core ML compiles large models; do not interrupt it just
because output is quiet.

```bash
cd /tmp/kokoro-external-bakeoff/laishere-kokoro-coreml
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python convert.py --max-frames 2000
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python \
  /path/to/kokoro-coreml/scripts/external_bakeoff/run_laishere_kokoro_coreml.py \
  --machine-id m2-studio \
  --laishere-repo /tmp/kokoro-external-bakeoff/laishere-kokoro-coreml \
  --manifest /path/to/kokoro-coreml/outputs/external_bakeoff/runtime_input_manifest.json \
  --spotcheck-dir /path/to/kokoro-coreml/outputs/external_bakeoff/spotcheck_wavs/laishere_kokoro_coreml_m2-studio
```

The adapter times the seven-stage Core ML chain only; G2P and feed preparation
are outside the timed calls, matching laishere's public benchmark.

## Config F

Config F uses the existing Swift benchmark binary and prepared Swift inputs:

```bash
cd swift && swift build -c release --product kokoro-bench
cd ..
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/prepare_swift_bench_inputs.py
python scripts/external_bakeoff/run_config_f_reference.py --machine-id m2-studio
```

By default, each adapter writes the last warm output for each successful input
to `outputs/external_bakeoff/spotcheck_wavs/<impl>_<machine_id>/<bucket>.wav`.
Pass `--spotcheck-dir` to use an explicit collection directory.

## Summarize

```bash
python scripts/external_bakeoff/summarize_external.py
```
