#!/usr/bin/env python3
"""
Hybrid ANE-Accelerated TTS pipeline demo (CLI).

Renamed from ``test_ane_pipeline.py`` so repo tooling does not treat it as a
pytest module. Production orchestration lives in ``kokoro.coreml_pipeline.HybridTTSPipeline``.
``run_single.py`` imports only that package—not this file.
"""

import argparse
import time
from pathlib import Path

# Optional imports with fallbacks
try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("ℹ️ soundfile not available - audio saving will be skipped")

from kokoro.coreml_pipeline import (
    COREML_AVAILABLE,
    COREML_MODEL_PATH,
    HybridTTSPipeline,
)


def check_ane_usage():
    """
    Check if the Apple Neural Engine is being used.

    This function provides various methods to verify ANE utilization,
    from simple model inspection to system-level monitoring.
    """
    print("\n🔍 Checking ANE Usage...")

    if not COREML_AVAILABLE:
        print("❌ CoreML model not available - cannot check ANE usage")
        return

    try:
        import coremltools as ct

        model = ct.models.MLModel(COREML_MODEL_PATH)

        # Check compute units configuration
        compute_units = model.compute_units
        print(f"📊 Model compute units: {compute_units}")

        if compute_units == ct.ComputeUnit.ALL:
            print("✅ Model allows ANE usage (compute_units=ALL)")
        elif compute_units == ct.ComputeUnit.CPU_AND_NE:
            print("✅ Model configured for CPU+ANE")
        else:
            print(f"⚠️ Model may not use ANE (compute_units={compute_units})")

        # Print performance recommendations
        print("\n💡 To verify ANE usage during runtime:")
        print("1. Use Instruments with Core ML template")
        print("2. Monitor 'Neural Engine' activity during inference")
        print("3. Run: sudo powermetrics -i 1000 --samplers ane | grep 'ANE Power'")
        print("4. Check for H11ANEServicesThread activity in Activity Monitor")

    except Exception as e:
        print(f"❌ Error checking ANE usage: {e}")


def run_performance_test(pipeline, test_texts):
    """
    Run performance benchmarks comparing different pipeline modes.

    Args:
        pipeline: HybridTTSPipeline instance
        test_texts: List of test texts to synthesize
    """
    print("\n⚡ Running Performance Tests...")

    results = []

    for i, text in enumerate(test_texts):
        print(f"\n📝 Test {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")

        start_time = time.time()
        audio, sample_rate = pipeline.synthesize(text)
        end_time = time.time()

        if audio is not None:
            duration = end_time - start_time
            audio_length = len(audio) / sample_rate
            rtf = duration / audio_length  # Real-time factor

            result = {
                "text": text,
                "synthesis_time": duration,
                "audio_length": audio_length,
                "rtf": rtf,
                "success": True,
            }

            print(f"  ⏱️  Synthesis time: {duration:.3f}s")
            print(f"  🎵 Audio length: {audio_length:.3f}s")
            print(f"  🚀 Real-time factor: {rtf:.3f}x")

        else:
            result = {
                "text": text,
                "success": False,
            }
            print("  ❌ Synthesis failed")

        results.append(result)

    # Summary
    successful_results = [r for r in results if r["success"]]
    if successful_results:
        avg_rtf = sum(r["rtf"] for r in successful_results) / len(successful_results)
        print("\n📊 Performance Summary:")
        print(f"  - Successful syntheses: {len(successful_results)}/{len(results)}")
        print(f"  - Average RTF: {avg_rtf:.3f}x")
        if avg_rtf < 1.0:
            print("  ✅ Pipeline is faster than real-time!")
        else:
            print("  ⚠️ Pipeline is slower than real-time")


def main():
    """Main execution function for the hybrid pipeline test."""
    print("🎯 Hybrid ANE-Accelerated TTS Pipeline Test")
    print("=" * 50)
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["pytorch", "coreml"], default="pytorch")
    args = parser.parse_args()

    # Initialize pipeline
    try:
        pipeline = HybridTTSPipeline(force_engine=args.engine)
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return

    # Check ANE usage capabilities
    check_ane_usage()

    # Test texts of varying complexity
    test_texts = [
        "Hello world!",
        "The quick brown fox jumps over the lazy dog.",
        "This is a longer sentence that will test the performance of our hybrid pipeline architecture.",
        "Kokoro is a high-quality text-to-speech system that can generate natural sounding speech.",
    ]

    # Run performance tests
    run_performance_test(pipeline, test_texts)

    # Generate sample outputs
    print("\n🎵 Generating Sample Audio Files...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    for i, text in enumerate(test_texts[:2]):  # Just first two for samples
        print(f"\n📝 Generating sample {i+1}: '{text}'")

        audio, sample_rate = pipeline.synthesize(text, voice="af_heart", speed=1.0)

        if audio is not None:
            output_path = output_dir / f"sample_{i+1:02d}.wav"
            if SOUNDFILE_AVAILABLE:
                sf.write(output_path, audio, sample_rate)
                print(f"  💾 Saved: {output_path}")
            else:
                print(f"  ⚠️ Would save to: {output_path} (soundfile not available)")
                print(f"  📊 Audio info: {audio.shape}, range [{audio.min():.3f}, {audio.max():.3f}]")
        else:
            print("  ❌ Failed to generate audio")

    print("\n🎉 Pipeline test completed!")
    print(f"📁 Sample audio files saved in: {output_dir}")

    if pipeline.use_coreml:
        print("\n🔥 Next Steps for ANE Usage Verification:")
        print("1. Run this script while monitoring with Instruments")
        print("2. Use 'sudo powermetrics -i 1000 --samplers ane' in another terminal")
        print("3. Look for Neural Engine activity during CoreML inference")


if __name__ == "__main__":
    main()
