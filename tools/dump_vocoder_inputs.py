#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
from test_ane_pipeline import HybridTTSPipeline, Phase1Constants, HybridPipelineConstants

# Dump vocoder inputs for the Phase 2 Swift app
# Saves JSON at Swift/KokoroPhase2/Resources/inputs_vocoder.json

def np_to_list(a: np.ndarray):
    return a.reshape(-1).astype(np.float32).tolist()


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "Swift" / "KokoroPhase2" / "Resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "inputs_vocoder.json"

    pipeline = HybridTTSPipeline(force_engine='coreml')
    text = Phase1Constants.TEST_SENTENCE
    voice = HybridPipelineConstants.BENCHMARK_VOICE_DEFAULT

    vi = pipeline.extract_vocoder_inputs(text, voice=voice, speed=1.0)
    if vi is None:
        raise RuntimeError("Failed to extract vocoder inputs")

    # Shapes expected by KokoroVocoder.mlpackage
    # asr: (1,512,1,asr_len), f0/n: (1,1,1,f0_len), s: (1,128)
    asr = vi['asr']          # (1,512,T_asr)
    f0 = vi['f0_curve']      # (1,T_f0)
    n = vi['n']              # (1,T_f0)
    s = vi['s']              # (1,128)

    asr_len = int(asr.shape[-1])
    f0_len = int(f0.shape[-1])

    payload = {
        "meta": {
            "text": text,
            "voice": voice,
            "asr_len": asr_len,
            "f0_len": f0_len,
            "sample_rate": 24000
        },
        "asr_shape": [1, 512, 1, asr_len],
        "f0_shape": [1, 1, 1, f0_len],
        "n_shape": [1, 1, 1, f0_len],
        "s_shape": [1, 128],
        "asr": np_to_list(asr),
        "f0": np_to_list(f0),
        "n": np_to_list(n),
        "s": np_to_list(s)
    }

    out_json.write_text(json.dumps(payload))
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    main()
