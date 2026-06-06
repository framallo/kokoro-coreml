# Irvine 3s Placement Target

Warmed Irvine M1 3s only. This report turns saved Core ML compute-plan
JSONs into the next implementation target for the remaining laishere
loss.

## Compute Plans

| Plan | Compute units | Ops | Preferred counts | CPU weight | GPU weight | NE weight | Source |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `ours_har_post_3s_cpu_ne` | `cpuAndNeuralEngine` | 2207 | `{"cpu": 1038, "unknown": 1169}` | 0.0% | 0.0% | 0.0% | `outputs/external_bakeoff/compute_plan/ours_har_post_3s_cpu_ne.json` |
| `ours_har_post_3s_cpu_gpu` | `cpuAndGPU` | 2207 | `{"gpu": 1041, "unknown": 1166}` | 0.0% | 100.0% | 0.0% | `outputs/graph_surface/irvine_m1/compute_plan_generator_3s_cpu_gpu.json` |
| `laishere_vocoder_cpu_ne` | `cpuAndNeuralEngine` | 1534 | `{"cpu": 58, "neuralEngine": 597, "unknown": 879}` | 49.6% | 0.0% | 47.5% | `outputs/external_bakeoff/compute_plan/laishere_vocoder_cpu_ne.json` |
| `exact_decoder_vocoder_body_3s_cpu_ne` | `cpuAndNeuralEngine` | 1546 | `{"cpu": 64, "neuralEngine": 599, "unknown": 883}` | 51.3% | 0.0% | 48.7% | `outputs/decoder_vocoder_split/3s_har_cos_rsqrt_native_in_broadcast_ios17/irvine/compute_plan_body_cpu_ne_irvine.json` |

## Decision

Our strict CPU+NE plan has `0` Neural Engine-preferred ops.
laishere's CPU+NE vocoder plan has `597` Neural Engine-preferred ops and `47.5%` estimated cost on Neural Engine.
A compute-unit flag flip is not sufficient.

## Partial-NE Counterexample

`3s_broadcast_adain_native_in_ios17` is a strict-pass surface but measured `318.2 ms` versus `177.1 ms` (`-79.7%` speedup).
That rejects the hypothesis that any partial Neural Engine placement wins.

## Exact Body Placement Trap

The existing strict decoder+vocoder body split already gets `599` Neural Engine-preferred ops and `48.7%` estimated cost on Neural Engine.
That still was not a warmed runtime win, so placement alone is not the target.

## Best Existing Strict Positive Surface

`3s_har28561` saves only `0.7 ms` (`0.4%`).

## Target

Build a laishere-like mixed CPU/Neural Engine body plan that is runtime-positive for the strict Swift HAR/source contract. Existing strict graph surfaces show that merely obtaining partial Neural Engine placement is not enough; the body boundary and operator surface must avoid the observed synchronization penalty.

## Deep Research Request

Design an M1 MLProgram source/STFT/vocoder body that preserves current Swift HAR/source semantics, keeps strict waveform parity, and shifts the expensive conv/add/mul/instance_norm body work into a laishere-like mixed CPU/Neural Engine plan without the existing split-boundary synchronization penalty or 3s warmed regression.

## Residual

Known saved positive estimates leave `6.2 ms` against warmed laishere.
