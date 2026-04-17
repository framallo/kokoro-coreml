# Core ML deployment of PyTorch LSTMs: enumerated sequence lengths and ANE utilization

April 16, 2026

> **Scope:** How **TorchScript tracing**, MIL, and **`ct.EnumeratedShapes`** interact for **unidirectional** LSTM exports to `mlprogram`: why **`RangeDim`** often fights the Apple Neural Engine (ANE), how **enumerated shapes** buy specialization, and how to **verify** placement (Netron, powermetrics, Instruments) and wire **Swift** `MLMultiArray` safely. For BiLSTM padding, mask-aware patterns, and repo-specific duration experiments, pair with the LSTM export guide and notes below.

## Related documentation

- **[Exporting PyTorch LSTMs to Core ML](CoreML-LSTM-export-guide.md)**: BiLSTM padding, MIL `lstm` limits, practical `EnumeratedShapes` patterns, and CPU vs ANE tradeoffs.
- **[Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)**: `MLComputeUnits`, silent ANE fallback, powermetrics, and Instruments.
- **[PyTorch MPS and Core ML field guide](pytorch-mps.md)**: Training and conversion context on Apple Silicon.
- **[Core ML conversion field manual](../../coreml-conversion-guide.md)**: `ct.convert` checklist and ANE-oriented pitfalls.
- **Institutional notes:** [debug-notes.md](../../Notes/debug-notes.md) — recurrent export and duration-path experiments in this repo.

---

## The core question

Deploying RNNs on edge accelerators means reconciling **dynamic-length** training code with **statically scheduled** hardware. A single Core ML `mlprogram` *can* accept **multiple exact sequence lengths at runtime** when those lengths are declared up front with **`ct.EnumeratedShapes`**: the converter attaches a finite set of allowed input shapes so MIL can keep a **native recurrent** `lstm` op instead of a giant unroll, and the runtime can specialize compilation for those shapes ([Flexible input shapes](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)).

That capability is still bounded by **coremltools/OS versions**, **ANE compiler heuristics**, **multi-input enumeration rules**, and **Swift** buffer lifetime when you bypass copies.

## PyTorch tracing and MIL

### What tracing fixes

PyTorch runs eagerly; for Core ML you typically produce **TorchScript** via `torch.jit.trace`. You pass a dummy tensor, e.g. shape `(batch_size, seq_len, input_size)` with `batch_first=True`. The tracer records one concrete execution path: the time dimension in the trace is **fixed to the dummy `seq_len`**. For example, tracing with `seq_len == 10` bakes **10** steps into the graph unless you override inputs at conversion time.

Mathematically, an LSTM walks a time index **t** from 1 to **T**. The hidden state **h_t** depends on **h_{t-1}** and the input at step **t**. Tracing does not preserve Python-level “any T” semantics by itself—it records the **observed** **T**.

### What `ct.convert` can override

The TorchScript graph is lowered to **Model Intermediate Language (MIL)**. Recurrent ops are represented as high-level **`lstm`** (and related) nodes, not necessarily as a long chain of matmuls. At **`ct.convert`** time you supply **`ct.TensorType`** with a flexible shape object. **`ct.EnumeratedShapes`** lists up to **128** concrete shapes (per input); the **default** shape is one of those tensors (see [input types](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html)). That lets the converter align the traced graph with **allowed** sequence lengths while keeping a single MIL recurrent op where the stack supports it.

If Netron shows a **deep unroll** (many repeated matmul/gate ops) instead of a compact **`lstm`** node, the stack often failed to map your TorchScript to the native recurrent path—common causes include **bidirectional** quirks, **dynamic slicing** outside the module, or unsupported surrounding ops. See [CoreML-LSTM-export-guide.md](CoreML-LSTM-export-guide.md) for padding and BiLSTM-specific issues.

## Netron checks

Open the `.mlpackage` in [Netron](https://github.com/lutzroeder/netron):

- **Inputs:** The time dimension should appear as part of an **enumerated** or symbolic flexible description—not only a single fixed integer—when you used `EnumeratedShapes`.
- **Graph:** Prefer a **single `lstm` / recurrent** node over a timestep-expanded chain. An unroll usually means you should simplify the TorchScript surface or adjust conversion inputs.

## `RangeDim` vs `EnumeratedShapes`

Core ML tools expose **bounded ranges** (`ct.RangeDim`) and **finite sets** (`ct.EnumeratedShapes`) for flexible dimensions ([flexible inputs guide](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)).

| Aspect | `ct.RangeDim` | `ct.EnumeratedShapes` | Single fixed shape |
| --- | --- | --- | --- |
| **Definition** | Min/max per dimension | Up to 128 listed full shapes | One static shape |
| **ANE** | Often poor: unknown shape at compile time can force fallback or expensive reshaping | Better: finite set enables per-shape specialization | Predictable if op is supported |
| **Waste** | No padding if you pass exact length | No padding if your length is in the set | Padding if you pad to one max length |

**Heuristic:** Fully open ranges interact badly with **static** ANE scheduling. Teams often report CPU/GPU fallback or high **recompilation** costs when shapes vary freely; see discussion in [coremltools#2370](https://github.com/apple/coremltools/issues/2370). **`EnumeratedShapes`** trades flexibility for **known** lengths—closer to what fixed-function accelerators expect.

**Multi-input models:** Enumerated shapes on **several** inputs are paired **by index** across inputs, not as a full Cartesian product. Mismatched combinations fail at runtime; see [coremltools#2271](https://github.com/apple/coremltools/issues/2271).

For **`mlprogram`**, prefer **bounded** ranges if you use `RangeDim` at all; unbounded `-1` style ranges belong to older **neural network** spec workflows ([FAQ / flexible shapes](https://apple.github.io/coremltools/docs-guides/source/faqs.html)).

## Versions and conversion path

- Flexible input work evolved across **coremltools** releases; align your script with the **documented** APIs for your installed version.
- **`mlprogram`** (`.mlpackage`) is the modern target; set **`minimum_deployment_target`** to match your app’s OS floor. Default dtypes for newer targets often favor **float16**, which matches typical ANE paths ([model input/output types](https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html)).
- **`torch.export`**-based flows may introduce different shape metadata than **`torch.jit.trace`**; treat export and trace as separate validation paths.

Historical issues around **dimension order** (sequence vs feature vs batch) show up in older trackers (e.g. [coremltools#880](https://github.com/apple/coremltools/issues/880)); always match **`batch_first`**, traced dummy shapes, and **`EnumeratedShapes`** entries.

## Compilation cache (`.mlmodelc`)

An `.mlpackage` is compiled for the device into a **`.mlmodelc`** cache. **Enumerated** shapes give the compiler a **finite** set of graphs to specialize; **wide-open** ranges do not. That is one reason enumerated inputs correlate with more predictable **ANE** residency *when* the graph is otherwise eligible—still **verify** on hardware ([compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)).

## Runtime profiling

Core ML does not always advertise which unit ran each op. Use:

- **`powermetrics`** (macOS) to sample **ANE vs CPU vs GPU** power/coalitions while you run inference—useful for detecting **silent fallback** (see [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)).
- **Xcode Instruments** with the **Core ML** template for a timeline of backend usage.

Example (one sample; adjust interval/count to your needs):

```bash
sudo powermetrics -n 1 -i 5000 --samplers tasks --show-process-energy --show-process-gpu
```

ANE-heavy runs often show stronger **Neural Engine** energy than CPU-only fallback; interpret numbers in context of your process and OS version.

## Swift: `MLMultiArray` and shapes

If inputs are enumerated, the **`MLMultiArray`** shape must match **one allowed tuple**. Otherwise Core ML fails verification (“shape not in enumerated set”). Production code often **pads** to the **nearest enumerated length** or selects a model variant; see [Stack Overflow discussion](https://stackoverflow.com/questions/57554527/how-to-solve-coreml-failure-verifying-inputs-shape-was-not-in-enumerated-set).

For throughput, **`MLMultiArray`** can wrap existing storage via **`dataPointer`**, but you must keep backing memory alive for async execution—typically **`withExtendedLifetime`** around the buffer for the duration of the `prediction` call.

## Minimal Python conversion sketch

Aligned with repo patterns in [CoreML-LSTM-export-guide.md](CoreML-LSTM-export-guide.md) and Apple’s [enumerated shapes example](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html):

```python
import numpy as np
import torch
import torch.nn as nn
import coremltools as ct


class FlexibleLSTM(nn.Module):
    def __init__(self, input_size=64, hidden_size=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return out


input_size = 64
model = FlexibleLSTM(input_size=input_size).eval()

dummy_seq_len = 10
batch_size = 1
dummy_input = torch.randn(batch_size, dummy_seq_len, input_size)
traced_model = torch.jit.trace(model, dummy_input)

valid_seq_lengths = [10, 20, 30]
enumerated_shapes = [
    [batch_size, t, input_size] for t in valid_seq_lengths
]
flexible_shape = ct.EnumeratedShapes(
    shapes=enumerated_shapes,
    default=[batch_size, 30, input_size],
)

mlmodel = ct.convert(
    traced_model,
    inputs=[
        ct.TensorType(
            name="sequence_input",
            shape=flexible_shape,
            dtype=np.float16,
        )
    ],
    convert_to="mlprogram",
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.iOS16,
)
mlmodel.save("FlexibleLSTM.mlpackage")
```

Inspect output feature names in Xcode or Netron; wire Swift keys to the actual graph outputs.

## Minimal Swift inference sketch

```swift
import CoreML
import Foundation

func runEnumeratedLSTM() throws {
    let config = MLModelConfiguration()
    config.computeUnits = .all

    let url = Bundle.main.url(forResource: "FlexibleLSTM", withExtension: "mlmodelc")!
    let model = try MLModel(contentsOf: url, configuration: config)

    let batch = 1
    let seq = 30
    let features = 64
    var data = [Float](repeating: 0.1, count: batch * seq * features)

    let shape = [batch, seq, features].map { NSNumber(value: $0) }
    let strides = [
        NSNumber(value: seq * features),
        NSNumber(value: features),
        NSNumber(value: 1),
    ]

    let input = try data.withUnsafeMutableBytes { raw -> MLMultiArray in
        let ptr = raw.baseAddress!
        return try MLMultiArray(
            dataPointer: ptr,
            shape: shape,
            dataType: .float32,
            strides: strides,
            deallocator: nil
        )
    }

    let provider = try MLDictionaryFeatureProvider(
        dictionary: ["sequence_input": input]
    )

    try withExtendedLifetime(data) {
        _ = try model.prediction(from: provider)
    }
}
```

Use the **real** output feature name from the converted model instead of guessing.

## Known conversion pitfalls

- **Hidden/weight shape errors** when flexing LSTM inputs: see [coremltools#1032](https://github.com/apple/coremltools/issues/1032) and bidirectional cases [coremltools#824](https://github.com/apple/coremltools/issues/824). Prefer **unidirectional** LSTMs for simpler flex-shape stories; enumerate **time** only with consistent **batch** and **feature** dims.
- **Wrong axis enumerated:** If you attach enumeration to **batch** instead of **time**, the model may compile yet behave catastrophically—double-check tensor layout vs `batch_first`.

## References

- [Flexible input shapes (`EnumeratedShapes`, `RangeDim`)](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html) — coremltools
- [MIL input types (`EnumeratedShapes`)](https://apple.github.io/coremltools/source/coremltools.converters.mil.input_types.html) — API reference
- [Model input and output types](https://apple.github.io/coremltools/docs-guides/source/model-input-and-output-types.html) — coremltools
- [Flexible input shapes on Neural Engine — discussion](https://github.com/apple/coremltools/issues/2370) — GitHub
- [Mixing multiple enumerated shape inputs](https://github.com/apple/coremltools/issues/2271) — GitHub
- [`MLMultiArrayShapeConstraint.enumeratedShapes`](https://developer.apple.com/documentation/coreml/mlmultiarrayshapeconstraint/enumeratedshapes) — Apple Developer
