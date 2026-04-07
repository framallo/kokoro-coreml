"""CLI for Kokoro synthesizer Core ML export."""
from __future__ import annotations

import argparse

from .convert import export_synthesizers


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Kokoro Synthesizer to CoreML with bucketing.")
    parser.add_argument("--output_dir", "-o", type=str, default="coreml", help="Output directory for mlpackage files.")
    parser.add_argument("--buckets", type=str, default="3s", help="Comma-separated list of bucket sizes in seconds (e.g., '3s,5s,10s').")
    parser.add_argument("--debug", action="store_true", help="Use smaller trace_length for debugging to avoid memory issues.")
    parser.add_argument("--trace_length", type=int, default=None, help="Override trace length (tokens). Must match duration export.")
    parser.add_argument("--precision", type=str, default=None, help="Core ML precision: 'float16'|'fp16' or 'float32'|'fp32'. Default: float16")
    parser.add_argument("--backend", type=str, default=None, help="Core ML backend: 'mlprogram' (default) or 'neuralnetwork' ('nn')")
    parser.add_argument("--mode", type=str, default="full", help="Export mode: 'full' (default) or 'decoder' for decoder-only model")
    args = parser.parse_args()

    try:
        export_synthesizers(
            args.output_dir,
            args.buckets,
            args.debug,
            trace_length=args.trace_length,
            precision=args.precision,
            backend=args.backend,
            mode=args.mode,
        )
        print("\n\n🎉 Synthesizer export complete. You're ready to ship.")
    except Exception as e:
        print(f"\n❌ An error occurred during export: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
