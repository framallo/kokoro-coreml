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
  scripts/external_bakeoff/run_mlx_audio.py --machine-id m2-studio
```

## Soniqo Speech Swift

Clone and pin `soniqo/speech-swift` outside the repo, then run:

```bash
python scripts/external_bakeoff/run_speech_swift_kokoro.py \
  --machine-id m2-studio \
  --speech-swift /tmp/kokoro-external-bakeoff/speech-swift
```

## Config F

Config F uses the existing Swift benchmark binary and prepared Swift inputs:

```bash
cd swift && swift build -c release --product kokoro-bench
cd ..
uv run --with-requirements requirements-bakeoff.txt --no-sync \
  python scripts/prepare_swift_bench_inputs.py
python scripts/external_bakeoff/run_config_f_reference.py --machine-id m2-studio
```

## Summarize

```bash
python scripts/external_bakeoff/summarize_external.py
```
