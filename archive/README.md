# Archived export scripts

## `export_vocoder.py`

Legacy **full-decoder** Core ML export producing `coreml/KokoroVocoder.mlpackage` (`VocoderWrapper` around the entire decoder). The repo’s **canonical** flow (see root `README.md`) is:

- **Duration:** `export_duration.py` → `coreml/kokoro_duration.mlpackage`
- **Decoder-only buckets:** `export_synthesizers.py` (package `export_synth/`) → `kokoro_decoder_only_*.mlpackage` / `kokoro_synthesizer_*.mlpackage`

`HybridTTSPipeline` may still load `KokoroVocoder.mlpackage` if present (`kokoro/coreml_pipeline.py`); the app prefers decoder-only buckets.

**Run:** from repository root, `python archive/export_vocoder.py --help`
