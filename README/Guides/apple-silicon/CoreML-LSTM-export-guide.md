# Exporting PyTorch LSTMs to Core ML on Apple Silicon

April 16, 2026

> **Scope:** Recurrent models (especially **bidirectional LSTMs**) exported from PyTorch to Core ML `mlprogram`: padding effects, MIL limitations, `pack_padded_sequence` conversion, dynamic shapes vs the ANE, `ct.EnumeratedShapes`, and practical workarounds. Pairs with the compute-unit scheduling guide for runtime verification.

## Related Documentation

- **[Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)**: `MLComputeUnits`, silent ANE fallback, powermetrics, Instruments, and LLDB.
- **[LSTM enumerated shapes and ANE (deep dive)](CoreML-LSTM-Enumerated-Shapes.md)**: TorchScript vs MIL, `RangeDim` vs `EnumeratedShapes`, Netron, profiling, Swift `MLMultiArray` wiring.
- **[PyTorch MPS and Core ML field guide](pytorch-mps.md)**: Training and conversion context on Apple Silicon.
- **[Core ML conversion field manual](../../coreml-conversion-guide.md)** (`README/`): `ct.convert` checklist, ANE pitfalls, profiling cues.
- **Institutional notes:** [debug-notes.md](../../Notes/debug-notes.md) — bidirectional LSTM, right-padding, and mask-aware export experiments for this repo’s duration path.

---

The deployment of sequence-to-sequence architectures—particularly text-to-speech (TTS) acoustic **duration** models built with bidirectional LSTMs (BiLSTMs)—is awkward on edge hardware. Server-side PyTorch can run exact-length batches; Core ML on Apple Silicon prefers static shapes and pushes recurrent graphs toward CPU/GPU unless the graph fits the Apple Neural Engine (ANE) compiler’s constraints.

When variable-length sequences are bucketed and **right-padded** with zeros, unidirectional LSTMs often tolerate padding if downstream layers only consume valid positions. **Bidirectional** LSTMs run forward and backward over the full padded length. The backward direction starts from the **end** of the tensor; padding tokens still participate in the recurrence, so **hidden** and **cell** states pick up bias and nonlinearity effects before the backward pass reaches real tokens. For duration heads, that mismatch can show up as badly wrong timings.

This guide walks through MIL behavior, why packed sequences and dynamic slices often hurt ANE placement, the `ct.EnumeratedShapes` compromise, and when forcing CPU/BNNS or changing architecture is the better move.

## Native Core ML MIL and sequence representation

The first question when moving a PyTorch BiLSTM to Core ML is whether MIL can represent **per-sequence lengths** or **masks** inside the `lstm` op so padded positions do not update state.

When `coremltools` converts PyTorch, it lowers TorchScript to **Model Intermediate Language (MIL)**. MIL is the contract between the graph and Apple’s backends (Espresso on CPU via BNNS, Metal for GPU, ANE runtime for supported ops). The MIL [`lstm`](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html#coremltools.converters.mil.mil.ops.defs.iOS15.recurrent.lstm) operator is essentially a fixed recurrent contraction over a **fixed** time dimension in the converted graph: it expects sequence tensors, initial hidden/cell, and weights—not a `sequence_length` vector or padding mask that skips steps inside the op the way `pack_padded_sequence` does in PyTorch.

Other MIL ops (for example sequence utilities) may accept length-like parameters, but **native MIL `lstm` does not offer** a first-class “valid length per batch row” switch comparable to packed RNNs in PyTorch. So a bucketed BiLSTM lowered to stock MIL `lstm` **still walks every padded timestep** unless you change the model (crop, pack, enumerate shapes, or re-architect).

## The PyTorch packed-sequence conversion reality

In PyTorch, the usual fix for padding is [`torch.nn.utils.rnn.pack_padded_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) / [`pad_packed_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_packed_sequence.html), which removes padding from the recurrence.

`coremltools` can ingest some of these paths, but conversion relies on **TorchScript** (`torch.jit.trace` / script) or newer export APIs. **Tracing** fixes a single execution path to concrete shapes; the sort/pack logic behind packed sequences becomes a pile of tensor ops (`non_zero`, `slice`, `scatter_nd`, reshapes, and so on). That pattern tends to:

- Fragment the graph and break the contiguous access patterns the ANE backend likes.
- Introduce **data-dependent shapes** that push work off the ANE or into long CPU paths.
- Interact badly with quantization and dynamic training features—conversion may raise `NotImplementedError` in edge cases.

So packed sequences may be *correct*, but they are often a poor fit for **fast, ANE-friendly** Core ML—not because MIL forbids them outright, but because the lowered graph is hostile to the accelerators you wanted.

## Dynamic slicing and ANE scheduling

Another approach is: keep a padded bucket, **slice** to the true length before `lstm`, run the RNN on exact length, then **pad** back to the bucket for the rest of the graph. Conceptually that avoids state corruption from padding in the RNN core.

In practice, **value-dependent** slice sizes (length known only at runtime) make it hard for the compiler to treat the following `lstm` as a static-shape island. The ANE stack favors **static** shapes for SRAM allocation; dynamic slicing and scatter-like pad-back patterns frequently land on **CPU or GPU** with extra synchronization. Control-flow ops (`while_loop`, `cond`, tight dynamic loops over length) are also a poor match for ANE placement.

So this route may be **correct** yet still **slow** and hard to keep on the ANE.

## Manual unroll (masked loop) and graph size

Implementing BiLSTM with a Python `for` loop, carrying state, and using `torch.where` with a mask can match PyTorch numerics. Traced into Core ML, that becomes a **large static unroll**: one timestep after another of matmuls, gates, and masks. For long buckets (hundreds or thousands of steps), you risk:

- **Compile-time pain**: very large MIL graphs stress the compiler (long compiles, high memory).
- **Runtime memory**: many intermediates; ANE SRAM pressure and spills; risk of jetsam on mobile.
- **Poor ANE utilization** even if some ops place, because the graph is dominated by serial recurrence and masking.

This path is usually something you **avoid** for production unless the sequence cap is small and you have measured end-to-end behavior.

## `ct.EnumeratedShapes`: practical compromise

Apple documents **enumerated input shapes** in [Flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) (`coremltools`). You supply up to **128** concrete shapes (per tensor); the compiler can treat each as a separate static specialization. Typical pattern for sequence models: enumerate `(batch, time, features)` with many candidate `time` values (for example steps of 16 up to a cap).

Tradeoffs:

- **Pros:** Keeps each specialization **static** from the compiler’s perspective—much friendlier to ANE than fully dynamic length, and avoids huge manual unrolls.
- **Cons:** First-time load may compile many variants; cold start can be noticeable. **Cache** compiled `.mlmodelc` artifacts and plan **when** compilation happens (e.g. onboarding vs hot path). Reuse a single `MLModel` instance where possible.

Example conversion pattern (from Apple’s flexible-inputs documentation, adapted for a tensor input named `x`):

```python
import coremltools as ct
import torch
import torch.nn as nn

class DurationBiLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        return self.projection(lstm_out)


model = DurationBiLSTM()
model.train(False)
max_seq_len = 512
example_input = torch.rand(1, max_seq_len, 256)
traced_model = torch.jit.trace(model, example_input)

valid_sequence_lengths = [16 * i for i in range(1, 33)]
enumerated_shapes = [[1, t, 256] for t in valid_sequence_lengths]
input_shape = ct.EnumeratedShapes(
    shapes=enumerated_shapes,
    default=[1, max_seq_len, 256],
)

mlmodel = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=input_shape, name="x")],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    compute_units=ct.ComputeUnit.ALL,
)
mlmodel.save("DurationBiLSTM_Enumerated.mlpackage")
```

Use `minimum_deployment_target=ct.target.iOS15` (or newer) if your pipeline requires a specific OS baseline. Output tensor names are auto-assigned unless you add explicit `outputs=[ct.TensorType(...)]`; inspect the package in Xcode or Netron for Swift wiring.

## Best practices for variable-length sequence models on Core ML

- Prefer **bounded** recurrence: enumerated shapes, or a small set of fixed exports, over “fully dynamic everywhere” if ANE throughput matters.
- **`MLModelConfiguration.computeUnits`** (Swift) / `compute_units` at convert time in Python is guidance; the compiler may still partition differently. Measure (see [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)).
- For **pure LSTM** stacks that fight the ANE, **`.cpuOnly`** or **`.cpuAndGPU`** is often more predictable than chasing ANE placement that spills or falls back silently.
- Long term, **Conv1d** or **Transformer**-style blocks align better with parallel matmuls and attention masks than vanilla BiLSTMs for many modern TTS and NLP stacks—see [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) (Apple ML Research).

## Comparative strategies

| Strategy | Correctness (BiLSTM padding) | ANE | Notes |
| --- | --- | --- | --- |
| **RangeDim / fully dynamic length** | Good if graph truly sees only valid tokens | Usually poor | Dynamic seq len fights static ANE specialization. |
| **EnumeratedShapes** | Near-good (pad to nearest bucket) | Good potential | Cold compile cost; cache `.mlmodelc`. |
| **Native MIL “mask inside lstm”** | Not available as in PyTorch | N/A | Do not assume mask args exist on MIL `lstm`. |
| **Manual unroll + mask** | Good | Poor | Huge graphs; compile/memory risk. |
| **Dynamic slice + pad back** | Good for RNN core | Often poor | Fallback + sync costs. |
| **Force CPU / BNNS** | Good | Bypassed | Stable for serial RNNs. |
| **Distill to Conv1d / Transformer** | Attention/conv masks | Strong | Architectural fix for many teams. |

## Example: dynamic slice (illustrates ANE tension)

This pattern slices to valid length before the LSTM; conversion may succeed, but placement on the ANE is often poor for the reasons above.

```python
import torch
import torch.nn as nn
import coremltools as ct


class DynamicSliceBiLSTM(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, bidirectional=True, batch_first=True
        )
        self.projection = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, valid_length):
        length_int = int(valid_length.item())
        valid_x = x[:, :length_int, :]
        lstm_out, _ = self.bilstm(valid_x)
        projected = self.projection(lstm_out)
        pad_size = x.size(1) - length_int
        return torch.nn.functional.pad(
            projected, (0, 0, 0, pad_size, 0, 0)
        )


model = DynamicSliceBiLSTM()
model.train(False)
example_input = torch.rand(1, 1000, 256)
example_length = torch.tensor(500, dtype=torch.int32)
traced_model = torch.jit.trace(model, (example_input, example_length))

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="x", shape=(1, 1000, 256)),
        ct.TensorType(name="valid_length", shape=(1,), dtype=torch.int32),
    ],
    convert_to="mlprogram",
)
mlmodel.save("DurationBiLSTM_DynamicSlice.mlpackage")
```

Shape names and dtypes should match your real export; this is a **schematic** example.

## Example: Transformer-style block (architecture direction)

```python
import torch
import torch.nn as nn
import coremltools as ct


class DurationTransformer(nn.Module):
    def __init__(self, input_dim=256, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.projection = nn.Linear(input_dim, 1)

    def forward(self, x, padding_mask):
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.projection(out)


model = DurationTransformer()
model.train(False)
example_input = torch.rand(1, 512, 256)
example_mask = torch.zeros(1, 512, dtype=torch.bool)
example_mask[0, 400:] = True
traced_model = torch.jit.trace(model, (example_input, example_mask))

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(name="x", shape=(1, 512, 256)),
        ct.TensorType(name="padding_mask", shape=(1, 512), dtype=bool),
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
)
mlmodel.save("DurationTransformer_Distilled.mlpackage")
```

Transformer conversion can still hit op coverage issues—validate with your exact module and `coremltools` version.

## Swift inference sketch (enumerated shapes)

Load a **compiled** `.mlmodelc` when possible to reuse the on-disk cache. Align padding to the **nearest enumerated length** your Python export used. Input feature names must match the converted model (here `x`).

```swift
import CoreML

final class DurationModelPredictor {
    private var model: MLModel?
    private let validShapes: [Int] = Array(stride(from: 16, through: 512, by: 16))

    func loadModelFromCache() {
        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            do {
                guard let url = Bundle.main.url(
                    forResource: "DurationBiLSTM_Enumerated",
                    withExtension: "mlmodelc"
                ) else { return }
                let config = MLModelConfiguration()
                config.computeUnits = .cpuOnly
                self?.model = try MLModel(contentsOf: url, configuration: config)
            } catch {
                print("Failed to load model: \(error)")
            }
        }
    }

    func predictDuration(acousticFeatures: [[Float]]) -> [Float]? {
        guard let model = model else { return nil }
        let actualLength = acousticFeatures.count
        guard let targetLength = validShapes.first(where: { $0 >= actualLength }) else {
            return nil
        }
        let shape: [NSNumber] = [
            NSNumber(value: 1),
            NSNumber(value: targetLength),
            NSNumber(value: 256),
        ]
        guard let multiArray = try? MLMultiArray(shape: shape, dataType: .float16) else {
            return nil
        }
        for t in 0..<targetLength {
            for d in 0..<256 {
                let idx = [0, t, d] as [NSNumber]
                if t < actualLength {
                    multiArray[idx] = NSNumber(value: Float(acousticFeatures[t][d]))
                } else {
                    multiArray[idx] = NSNumber(value: 0.0)
                }
            }
        }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["x": multiArray]) else {
            return nil
        }
        guard let prediction = try? model.prediction(from: input) else { return nil }
        let outputName = model.modelDescription.outputDescriptionsByName.keys.first
            ?? "identity"
        guard let outputArray = prediction.featureValue(for: outputName)?.multiArrayValue else {
            return nil
        }
        var out: [Float] = []
        for t in 0..<actualLength {
            let idx = [0, t, 0] as [NSNumber]
            out.append(outputArray[idx].floatValue)
        }
        return out
    }
}
```

Replace indexing with your real rank (e.g. scalar vs last-dim layout) and the actual output feature name from `modelDescription`.

## Verifying execution: ANE, GPU, CPU

Core ML does not always tell you where ops ran. Use Instruments (Time Profiler, [Processor Trace](https://developer.apple.com/documentation/xcode/analyzing-cpu-usage-with-processor-trace) where available), `powermetrics` (ANE vs CPU/GPU power—see [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)), and unified logging, for example:

```bash
sudo log show --predicate '(subsystem IN {"com.apple.espresso","com.apple.coreml"})' --info --debug --last 5m
```

LLDB symbolic breakpoint (illustrative; symbols vary by OS/toolchain):

```text
(lldb) breakpoint set -n "-[_ANEModel program]"
```

## Known issues and version notes

- **Scatter / index validation:** MIL scatter ops gained stricter index semantics on newer OS releases; negative or invalid indices in converted masking/scatter patterns can crash or misbehave—audit indexing after conversion.
- **Rank mismatches:** Bidirectional RNN outputs are 3D; later layers expecting 2D can throw reshape/broadcast errors—often fixed with explicit reshape ops or consistent `TensorType` rank in `ct.convert`.
- **Long compile / timeout:** Huge unrolled graphs or pathological slice/pad patterns can stress the compiler—reduce unroll depth or use enumerated shapes / smaller caps.
- **Quantization + dynamic ops:** Dynamic packing and some training-time quant paths are brittle; expect to fall back to CPU-only conversion or simplify the graph.

## Conclusions

BiLSTMs on Core ML inherit **padding semantics** from PyTorch unless you change the graph. Native MIL `lstm` does not give you PyTorch-style packed RNNs inside one op; packed lowers are often **bad for ANE**; dynamic slicing trades correctness in the RNN for **scheduling** problems; manual unroll explodes graph size.

The usual **pragmatic** compromise is **`ct.EnumeratedShapes`** with a modest set of lengths, plus caching compiled artifacts and careful `MLModel` lifecycle. If the ANE still does not help, **CPU/BNNS** is often the honest answer for serial RNNs. Longer term, **non-recurrent** architectures fit parallel accelerators and masking better.

## References

- [Model Intermediate Language (MIL)](https://apple.github.io/coremltools/docs-guides/source/model-intermediate-language.html) — `coremltools`
- [Flexible input shapes (EnumeratedShapes, RangeDim)](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) — `coremltools`
- [Convert PyTorch models](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html) — `coremltools`
- [MIL op reference (recurrent / lstm)](https://apple.github.io/coremltools/source/coremltools.converters.mil.mil.ops.defs.html) — `coremltools`
- [PackedSequence / `pack_padded_sequence`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) — PyTorch
- [LSTM tutorial (background)](https://www.datacamp.com/de/tutorial/lstm-models) — DataCamp
- [Padding effects in LLMs (general)](https://arxiv.org/html/2510.01238v2) — arXiv
- [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers) — Apple ML Research
- [ANE execution hints (community)](https://github.com/hollance/neural-engine/blob/master/docs/is-model-using-ane.md) — GitHub
