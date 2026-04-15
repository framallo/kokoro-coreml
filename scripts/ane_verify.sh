#!/usr/bin/env bash
# ANE verification and profiling guide for Kokoro HAR-post Core ML inference.
#
# Run this in one terminal while running the synthesis loop in another.
# Each section below is self-contained — pick the one that matches your goal.
#
# Reference: README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md
#            CLAUDE.md §5 (Validate → Profile → Iterate)

set -euo pipefail

VOICE="${VOICE:-af_heart}"
TEXT="${TEXT:-This is a longer sentence that will test the performance of our pipeline running on the Apple GPU. More text at the end pushes nine seconds.}"
ITERS="${ITERS:-30}"

echo "============================================================"
echo " Kokoro ANE Verification"
echo " voice=$VOICE  iters=$ITERS"
echo "============================================================"
echo ""
echo "STEP 1 — Launch a sustained synthesis loop in ANOTHER terminal:"
echo ""
echo "  uv run python -c \""
echo "  from kokoro.coreml_pipeline import HybridTTSPipeline"
echo "  p = HybridTTSPipeline()"
echo "  for _ in range($ITERS):"
echo "      p.synthesize('$TEXT', '$VOICE', 1.0)"
echo "  print('done')"
echo "  \""
echo ""
echo "------------------------------------------------------------"
echo "STEP 2 — While the loop is running, run ONE of the following:"
echo ""

echo "--- 2A. Quickest: powermetrics ANE power (needs sudo) ---"
echo "sudo powermetrics -i 500 --samplers ane | grep -E 'ANE Power|Neural'"
echo ""
echo "  Interpretation:"
echo "    ANE Power: 0 mW  → silent fallback, not on ANE"
echo "    ANE Power: >0 mW → Neural Engine is active"
echo ""

echo "--- 2B. Full CPU+GPU+ANE power breakdown ---"
echo "sudo powermetrics -i 1000 -n 10 --samplers cpu_power,gpu_power,ane"
echo ""

echo "--- 2C. Instruments (no sudo, visual, most detail) ---"
echo "  1. Open Xcode → Product → Profile (Cmd+I) on a test host app"
echo "     OR: Instruments.app → File → New → Core ML template"
echo "  2. Add 'Neural Engine' and 'GPU' instruments"
echo "  3. Look for H11ANEServicesThread in thread list = ANE active"
echo "  4. Absence of H11ANEServicesThread = fallback to CPU/GPU"
echo ""

echo "--- 2D. LLDB symbolic breakpoints (definitive proof) ---"
echo "  Attach LLDB to the Python process and set:"
echo "    (lldb) breakpoint set -n \"-[_ANEModel program]\""
echo "    (lldb) breakpoint set -n \"Espresso::BNNSEngine::convolution_kernel::__launch\""
echo "  If -[_ANEModel program] fires → ANE is executing"
echo "  If BNNSEngine fires instead  → CPU fallback confirmed"
echo ""

echo "============================================================"
echo "STEP 3 — After collecting data, run stage breakdown:"
echo ""
echo "  uv run python scripts/bench_pipeline_stages.py --preset long --iterations 7"
echo ""
echo "  This shows how much of total latency is CPU vs Core ML."
echo "  If Core ML < 20% of wall time, optimise the CPU pipeline first."
echo "============================================================"
echo ""

echo "STEP 4 — If ANE power stays 0 mW (silent fallback):"
echo ""
echo "  Common causes in Kokoro's GeneratorFromHar:"
echo "    - 3D Conv1d tensors (B,C,T): ANE prefers 4D (B,C,1,T)"
echo "      Fix: convert backbone Conv1d → Conv2d (major change, do AFTER"
echo "           confirming Core ML is actually the bottleneck via STEP 3)"
echo "    - Dynamic sequence dimensions: already fixed via fixed-shape buckets"
echo "    - Orion Constraint 1 (concat): already removed from AdaIN1d"
echo ""
echo "  If ANE IS active but Core ML is still <20% of total wall time:"
echo "    → The bottleneck is the CPU hn-nsf (SineGen/SourceModuleHnNSF)."
echo "    Next steps:"
echo "      a) torch.compile(gen.m_source) — fuse hn-nsf kernels"
echo "      b) Test hn-nsf on MPS (debug-notes.md shows ~0.00 correlation"
echo "         for CoreML FP16 only; MPS at full precision may be fine)"
echo "      c) Profile SineGen sub-ops with torch.profiler to find the hot loop"
echo "============================================================"
