#!/usr/bin/env python3
"""Prepare pre-tokenized inputs for the Swift benchmark.

Runs the Python tokenizer on the 4 bakeoff inputs and saves the results
as JSON files that the Swift benchmark can load directly.

Usage::

    uv run python scripts/prepare_swift_bench_inputs.py

Output: outputs/swift_bench_inputs/{key}.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent

BAKEOFF_INPUTS = {
    "tiny": "Hello world!",
    "short": "The quick brown fox jumps over the dog.",
    "medium": (
        "This is a longer sentence designed to test the performance "
        "of our text to speech system running on the Apple GPU."
    ),
    "long": (
        "This is a longer sentence designed to test the performance "
        "of our text to speech system running on modern Apple Silicon "
        "hardware. A few more words added here."
    ),
}
VOICE = "af_heart"
SPEED = 1.0


def main():
    from kokoro import KModel, KPipeline
    from kokoro.pipeline import voice_embedding_for_phoneme_string

    output_dir = _ROOT / "outputs" / "swift_bench_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = KPipeline(lang_code="a")
    kmodel = KModel()

    # Save hn-nsf linear weights
    gen = kmodel.decoder.generator
    linear_w = gen.m_source.l_linear.weight.detach().numpy().flatten().tolist()
    linear_b = float(gen.m_source.l_linear.bias.detach().numpy().flatten()[0])

    hnsf_config = {
        "linear_weights": linear_w,
        "linear_bias": linear_b,
    }
    with open(output_dir / "hnsf_weights.json", "w") as f:
        json.dump(hnsf_config, f)
    print(f"Saved hn-nsf weights: {len(linear_w)} weights, bias={linear_b:.6f}")

    for key, text in BAKEOFF_INPUTS.items():
        print(f"\n--- {key}: {text!r:.60} ---")

        # Tokenize
        voice_pack = pipeline.load_voice(VOICE)
        phonemes = None
        for _, ps, _ in pipeline(text, VOICE, SPEED):
            phonemes = ps
            break

        if not phonemes:
            print(f"  FAILED to tokenize {key}")
            continue

        # Get voice embedding
        ref_s = voice_embedding_for_phoneme_string(voice_pack, phonemes)

        # Build input IDs (padded to 128)
        input_ids = list(filter(lambda i: i is not None,
                               map(lambda p: kmodel.vocab.get(p), phonemes)))
        input_ids = [0] + input_ids + [0]  # BOS/EOS

        # Pad to 128
        padded_ids = input_ids[:128] + [0] * max(0, 128 - len(input_ids))
        attention_mask = [1] * min(len(input_ids), 128) + [0] * max(0, 128 - len(input_ids))

        ref_s_list = ref_s.cpu().numpy().flatten().tolist()

        # Get canonical duration from the bakeoff manifest if available
        manifest_path = _ROOT / "outputs" / "bakeoff" / "input_manifest.json"
        canonical_dur = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if key in manifest.get("inputs", {}):
                canonical_dur = manifest["inputs"][key].get("canonical_duration_s")

        # If no manifest, compute from extract_vocoder_inputs
        if canonical_dur is None:
            from kokoro.coreml_pipeline import HybridTTSPipeline
            pipe = HybridTTSPipeline()
            vi = pipe.extract_vocoder_inputs(text, VOICE, SPEED)
            if vi is not None:
                T_f0 = int(vi["f0_curve"].shape[-1])
                canonical_dur = T_f0 / 80.0

        entry = {
            "key": key,
            "text": text,
            "voice": VOICE,
            "speed": SPEED,
            "phonemes": phonemes,
            "input_ids": padded_ids,
            "attention_mask": attention_mask,
            "ref_s": ref_s_list,
            "canonical_duration_s": canonical_dur,
            "num_tokens": len(input_ids),
        }

        with open(output_dir / f"{key}.json", "w") as f:
            json.dump(entry, f)
        print(f"  Tokens: {len(input_ids)}, phonemes: {phonemes}")

    print(f"\nAll inputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
