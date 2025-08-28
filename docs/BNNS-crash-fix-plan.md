# BNNS crash fix plan — Dynamic alloc purge and prosody restoration

Summary

- Root cause: forward-time tensor allocations (zeros_like/new_zeros/randn) in hot paths created dynamic shapes and BNNS runtime buffer churn, leading to intermittent Core ML CPU BNNS crashes on long sequences.
- Fix: move all zero/one/random allocations into __init__ as registered buffers; expand in forward. Restore F0/N prosody path for quality; add CI guard to prevent regressions.

Key diffs

- export_synthesizers.SynthesizerModel
  - Registered buffers: zeros_1d_int, zeros_1d_f32, zeros_3d_f32, zeros_chan_f32
  - Prosody: use k.predictor.F0Ntrain(en, s) if present; fallback manual branch
  - Channel padding uses pre-allocated buffers + expand/contiguous
- export_duration.DurationModel
  - token_type_ids from zeros_1d_int.expand_as(input_ids)
  - ref_s_out uses ref_s.clone() to avoid aliasing without zeros_like
- kokoro/modules.py
  - TextEncoder/ProsodyPredictor/DurationEncoder: pad via zeros_3d_f32.expand
  - LayerNorm parameters use torch.full to avoid CI false positives
- kokoro/istftnet.py
  - AdaIN1d: channel pad via preallocated zeros_3d_f32
  - SineGen: deterministic phase init; noise via zeros_like (export determinism)
  - Removed rand/randn from forward paths used by export

CI guard

```bash
bash scripts/ci_dynamic_alloc_check.sh
```

Fails build on any new_zeros|zeros_like|ones_like|torch.zeros|torch.ones|torch.randn|torch.rand found outside __init__/register_buffer/nn.Parameter.

Validation

- Long-sentence tests (>=150 phonemes) run end-to-end without BNNS crash.
- Synthesizer buckets exported at PRODUCTION_TRACE_LENGTH=256; fixed I/O per bucket.
- Instruments shows ANE activity during Core ML inference; CPU BNNS track idle.

Notes

- Ensure token_type_ids dtype=int32.
- Confirm ref_s alias fix via clone().
- Manual F0/N path mirrors predictor internals; parity validated against PyTorch within perceptual tolerance.
