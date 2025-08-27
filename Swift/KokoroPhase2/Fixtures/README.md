Fixture JSON may include optional HAR tensors for Decoder_HAR models:
- har_spec: shape [1, C, 1, T]
- har_phase: shape [1, C, 1, T]
If present, pass them when invoking a Decoder_HAR Core ML model; otherwise only decoder-only inputs are used.
