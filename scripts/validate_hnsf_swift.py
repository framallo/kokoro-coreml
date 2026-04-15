#!/usr/bin/env python3
"""Generate PyTorch hn-nsf reference outputs for Swift validation.

Runs the PyTorch hn-nsf pipeline on test inputs and saves intermediate arrays
as .npy files. The Swift implementation reads the same inputs and compares
its outputs against these references.

Usage::

    uv run python scripts/validate_hnsf_swift.py generate   # save reference arrays
    uv run python scripts/validate_hnsf_swift.py compare     # compare Swift output vs reference

Output directory: outputs/hnsf_validation/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load_generator():
    """Load the Kokoro Generator module."""
    from kokoro import KModel
    kmodel = KModel()
    return kmodel.decoder.generator


def generate_references(output_dir: Path) -> None:
    """Run PyTorch hn-nsf on test inputs and save all intermediates."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Generator...")
    gen = _load_generator()

    # Extract SourceModuleHnNSF learned weights for Swift
    linear_w = gen.m_source.l_linear.weight.detach().numpy().flatten()  # (9,)
    linear_b = gen.m_source.l_linear.bias.detach().numpy().flatten()    # (1,)
    np.save(output_dir / "linear_weights.npy", linear_w)
    np.save(output_dir / "linear_bias.npy", linear_b)
    print(f"  l_linear weights: {linear_w.shape}, bias: {linear_b.shape}")
    print(f"  weights: {linear_w}")
    print(f"  bias: {linear_b}")

    # Save config for Swift
    config = {
        "sample_rate": 24000,
        "upsample_scale": int(round(float(gen.f0_upsamp.scale_factor))),
        "harmonic_num": gen.m_source.l_sin_gen.harmonic_num,
        "sine_amp": float(gen.m_source.sine_amp),
        "noise_std": float(gen.m_source.noise_std),
        "voiced_threshold": float(gen.m_source.l_sin_gen.voiced_threshold),
        "stft_n_fft": gen.stft.filter_length if hasattr(gen.stft, 'filter_length') else 20,
        "stft_hop": gen.stft.hop_length if hasattr(gen.stft, 'hop_length') else 5,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config}")

    # Test inputs — representative F0 curves
    test_cases = {
        "voiced_constant": np.full((1, 80), 200.0, dtype=np.float32),   # 200 Hz constant
        "voiced_sweep": np.linspace(100, 400, 120, dtype=np.float32).reshape(1, -1),  # sweep
        "mixed_uv": np.concatenate([
            np.full(40, 150.0),     # voiced
            np.zeros(20),           # unvoiced
            np.full(60, 250.0),     # voiced
        ]).astype(np.float32).reshape(1, -1),
        "tiny_bakeoff": np.full((1, 62), 180.0, dtype=np.float32),     # ~"Hello world!" length
    }

    for name, f0_pad in test_cases.items():
        print(f"\n--- Test case: {name} (f0_len={f0_pad.shape[-1]}) ---")
        case_dir = output_dir / name
        case_dir.mkdir(exist_ok=True)

        np.save(case_dir / "f0_input.npy", f0_pad)

        # Set deterministic seed for reproducible noise
        torch.manual_seed(42)
        np.random.seed(42)

        with torch.no_grad():
            # Step 1: F0 upsample
            f0_up = gen.f0_upsamp(torch.from_numpy(f0_pad)[:, None]).transpose(1, 2)
            f0_up_np = f0_up.numpy().squeeze()
            np.save(case_dir / "f0_upsampled.npy", f0_up_np)
            print(f"  f0_up shape: {f0_up.shape} -> squeezed: {f0_up_np.shape}")

            # Step 2: SourceModuleHnNSF (includes SineGen)
            har_source, noise, uv = gen.m_source(f0_up)
            har_source_np = har_source.transpose(1, 2).squeeze(1).numpy().squeeze()
            np.save(case_dir / "har_source.npy", har_source_np)
            print(f"  har_source shape: {har_source_np.shape}")

            # Step 3: STFT
            har_source_stft = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = gen.stft.transform(har_source_stft)
            har_spec_np = har_spec.numpy().squeeze()
            har_phase_np = har_phase.numpy().squeeze()
            np.save(case_dir / "har_spec.npy", har_spec_np)
            np.save(case_dir / "har_phase.npy", har_phase_np)
            print(f"  har_spec shape: {har_spec_np.shape}, har_phase shape: {har_phase_np.shape}")

            # Step 4: Concatenated har
            har = torch.cat([har_spec, har_phase], dim=1)
            har_np = har.numpy().squeeze()
            np.save(case_dir / "har_output.npy", har_np)
            print(f"  har output shape: {har_np.shape}")

    print(f"\nAll references saved to: {output_dir}")


def compare_outputs(output_dir: Path, swift_dir: Path) -> None:
    """Compare Swift outputs against PyTorch references."""
    test_cases = [d.name for d in output_dir.iterdir() if d.is_dir()]

    all_pass = True
    for name in sorted(test_cases):
        case_dir = output_dir / name
        swift_case = swift_dir / name

        if not swift_case.exists():
            print(f"SKIP: {name} — no Swift output found at {swift_case}")
            continue

        print(f"\n--- Comparing: {name} ---")

        for stage in ["har_source", "har_spec", "har_phase", "har_output"]:
            ref_path = case_dir / f"{stage}.npy"
            swift_path = swift_case / f"{stage}.npy"

            if not swift_path.exists():
                print(f"  SKIP: {stage} — no Swift output")
                continue

            ref = np.load(ref_path).flatten()
            swift = np.load(swift_path).flatten()

            if ref.shape != swift.shape:
                print(f"  FAIL: {stage} — shape mismatch: ref {ref.shape} vs swift {swift.shape}")
                all_pass = False
                continue

            corr = float(np.corrcoef(ref, swift)[0, 1]) if ref.size > 1 else 1.0
            max_err = float(np.max(np.abs(ref - swift)))
            threshold = 0.999 if stage in ("har_spec", "har_phase") else 0.99

            status = "PASS" if corr > threshold else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"  {status}: {stage} — corr={corr:.6f} (threshold {threshold}), max_err={max_err:.6f}")

    if all_pass:
        print("\nAll comparisons PASS.")
    else:
        print("\nSome comparisons FAILED.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="hn-nsf Swift validation")
    parser.add_argument("action", choices=["generate", "compare"])
    parser.add_argument("--output-dir", type=str, default="outputs/hnsf_validation")
    parser.add_argument("--swift-dir", type=str, default="outputs/hnsf_validation_swift")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.action == "generate":
        generate_references(output_dir)
    else:
        compare_outputs(output_dir, Path(args.swift_dir))


if __name__ == "__main__":
    main()
