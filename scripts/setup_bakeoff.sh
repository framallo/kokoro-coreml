#!/usr/bin/env bash
# Setup everything needed to run the bakeoff on a fresh machine.
#
# Usage:
#     bash scripts/setup_bakeoff.sh          # full setup
#     bash scripts/setup_bakeoff.sh --skip-download  # skip HF download (models already local)
#
# After this completes, run the bakeoff with:
#     BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \
#     uv run --no-sync python scripts/bakeoff_harness.py run \
#       --configs a,d,e,f --iterations 5 --order-seed 0
#
# Or use the $bakeoff skill.

set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_DOWNLOAD=false
for arg in "$@"; do
    case "$arg" in
        --skip-download) SKIP_DOWNLOAD=true ;;
    esac
done

echo "=== Bakeoff Setup ==="
echo "  Repo: $(pwd)"
echo "  Skip download: $SKIP_DOWNLOAD"
echo

# 1. Python deps
echo "--- Step 1/6: Python dependencies ---"
if command -v uv &>/dev/null; then
    uv sync
    uv pip install -r requirements-bakeoff.txt
else
    echo "uv not found. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

run_python() {
    uv run --no-sync python "$@"
}

run_pytest() {
    uv run --no-sync pytest "$@"
}

# 2. Download base models from HF
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo
    echo "--- Step 2/6: Download CoreML models from Hugging Face ---"
    run_python scripts/download_models.py --coreml
else
    echo
    echo "--- Step 2/6: Skipping download (--skip-download) ---"
fi

# 3. Prepare canonical benchmark inputs before Duration export so
# export_duration.py can build exact-token native Duration packages for the
# frozen bakeoff texts.
echo
echo "--- Step 3/6: Prepare bakeoff inputs ---"
run_python scripts/bakeoff_harness.py prepare-inputs
run_python scripts/prepare_swift_bench_inputs.py

# 4. Export new models
echo
echo "--- Step 4/6: Export Duration models (padded fallback + exact token packages) ---"
run_python export_duration.py

echo
echo "--- Step 4/6: Export F0Ntrain models ---"
run_python export_f0ntrain.py --t-frames 120 280 400 600 1200

echo
echo "--- Step 4/6: Export DecoderPre models ---"
run_python export_decoder_pre.py --buckets 3 7 10 15 30

echo
echo "--- Step 4/6: Export GeneratorFromHar models ---"
run_python -m export_synth.main --mode decoder-har --buckets 3s,7s,10s,15s,30s -o coreml

# 5. Build Swift binary
echo
echo "--- Step 5/6: Build Swift benchmark CLI ---"
cd swift
swift build -c release --product kokoro-bench
cd ..

# 6. Verify
echo
echo "--- Step 6/6: Verify ---"
READY=true

if [ ! -f "coreml/kokoro_duration.mlpackage/Manifest.json" ]; then
    echo "  MISSING: Duration model"
    READY=false
fi

for input_json in outputs/swift_bench_inputs/*.json; do
    [ -e "$input_json" ] || continue
    [ "$(basename "$input_json")" = "hnsf_weights.json" ] && continue
    tokens="$(run_python -c 'import json,sys; print(json.load(open(sys.argv[1]))["num_tokens"])' "$input_json")"
    if [ ! -f "coreml/kokoro_duration_exact_t${tokens}.mlpackage/Manifest.json" ]; then
        echo "  MISSING: exact Duration model T=${tokens}"
        READY=false
    fi
done

for bucket in 3 7 10 15 30; do
    if [ ! -d "coreml/kokoro_decoder_har_post_${bucket}s.mlpackage" ]; then
        echo "  MISSING: GeneratorFromHar ${bucket}s"
        READY=false
    fi
    if [ ! -d "coreml/kokoro_decoder_pre_${bucket}s.mlpackage" ]; then
        echo "  MISSING: DecoderPre ${bucket}s"
        READY=false
    fi
done

for tframes in 120 280 400 600 1200; do
    if [ ! -d "coreml/kokoro_f0ntrain_t${tframes}.mlpackage" ]; then
        echo "  MISSING: F0Ntrain T=${tframes}"
        READY=false
    fi
done

if [ ! -f "swift/.build/release/kokoro-bench" ]; then
    echo "  MISSING: Swift binary"
    READY=false
fi

if [ ! -f "outputs/bakeoff/input_manifest.json" ]; then
    echo "  MISSING: Input manifest"
    READY=false
fi

if [ ! -f "outputs/swift_bench_inputs/hnsf_weights.json" ]; then
    echo "  MISSING: hn-nsf weights"
    READY=false
fi

if [ "$READY" = true ]; then
    echo "  Verifying Core ML package shape contracts..."
    if ! run_pytest -q tests/test_mlpackage_exports.py::test_decoder_har_post_bucket_shape_matches_advertised_duration; then
        READY=false
    fi
fi

if [ "$READY" = true ]; then
    echo "  All prerequisites present."
    echo
    echo "=== Setup complete. Run the bakeoff with: ==="
    echo
    echo "  BAKEOFF_SKIP_SMOKE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 \\"
    echo "  uv run --no-sync python scripts/bakeoff_harness.py run \\"
    echo "    --configs a,d,e,f --iterations 5 --order-seed 0"
    echo
    echo "  On 16 GB machines, split the run if the combined A/D/E/F"
    echo "  process swaps heavily:"
    echo "    --configs a,f   then   --configs d,e"
    echo
else
    echo
    echo "  Some prerequisites missing. Check output above."
    exit 1
fi
