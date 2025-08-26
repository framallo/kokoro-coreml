#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path so imports from sibling modules work when running from tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test_ane_pipeline import HybridTTSPipeline, Phase1Constants, HybridPipelineConstants
from kokoro import KModel

# Dump vocoder inputs for the Phase 2 Swift app
# Saves JSON at Swift/KokoroPhase2/Resources/inputs_vocoder.json

def np_to_list(a: np.ndarray):
    return a.reshape(-1).astype(np.float32).tolist()


def main():
    base = Path(__file__).resolve().parent.parent
    out_dir = base / "Swift" / "KokoroPhase2" / "Resources"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "inputs_vocoder.json"

    # Optional CLI: --seconds {5,15,30}
    seconds = 5
    if len(sys.argv) >= 3 and sys.argv[1] == "--seconds":
        try:
            seconds = int(sys.argv[2])
        except Exception:
            pass
        if seconds not in (5, 15, 30):
            print(f"Unsupported seconds={seconds}; defaulting to 5")
            seconds = 5

    pipeline = HybridTTSPipeline(force_engine='coreml')
    text = Phase1Constants.TEST_SENTENCE
    voice = HybridPipelineConstants.BENCHMARK_VOICE_DEFAULT

    vi = pipeline.extract_vocoder_inputs(text, voice=voice, speed=1.0)
    if vi is None:
        raise RuntimeError("Failed to extract vocoder inputs")

    # Shapes expected by Kokoro models
    # For golden parity, normalize to 5s bucket: asr_len=200, f0/n_len=400
    # asr: (1,512,1,asr_len), f0/n: (1,1,1,f0_len), s: (1,128)
    asr = vi['asr']          # (1,512,T_asr)
    f0 = vi['f0_curve']      # (1,T_f0)
    n = vi['n']              # (1,T_f0)
    s = vi['s']              # (1,128)

    # Bucket lengths (heuristics): ~40 fps for ASR branch, ~80 fps for f0/n branch
    asr_len_target = {5: 200, 15: 600, 30: 1200}[seconds]
    f0_len_target = {5: 400, 15: 1200, 30: 2400}[seconds]

    # Tail-pad/truncate to bucket shapes
    def pad_tail_1d(x: np.ndarray, T: int) -> np.ndarray:
        out = np.zeros((1, T), dtype=np.float32)
        t = min(T, x.shape[-1])
        if t > 0:
            out[:, :t] = x[:, :t]
        return out

    def pad_tail_asr(x: np.ndarray, T: int) -> np.ndarray:
        out = np.zeros((1, 512, T), dtype=np.float32)
        t = min(T, x.shape[-1])
        if t > 0:
            out[:, :, :t] = x[:, :, :t]
        return out

    asr_pad = pad_tail_asr(asr.astype(np.float32), asr_len_target)
    f0_pad = pad_tail_1d(f0.astype(np.float32), f0_len_target)
    n_pad = pad_tail_1d(n.astype(np.float32), f0_len_target)

    asr_len = asr_len_target
    f0_len = f0_len_target

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
        "asr": np_to_list(asr_pad),
        "f0": np_to_list(f0_pad),
        "n": np_to_list(n_pad),
        "s": np_to_list(s)
    }

    # Also compute HAR features via exact PyTorch path for highest quality (Decoder_HAR)
    try:
        model = KModel(disable_complex=True).to('cpu').eval()
        dec = model.decoder
        with torch.no_grad():
            # Use padded f0 to match 5s bucket exactly
            f0_t = torch.from_numpy(f0_pad).float()
            f0_up = dec.generator.f0_upsamp(f0_t[:, None]).transpose(1, 2)
            har_source, _, _ = dec.generator.m_source(f0_up)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = dec.generator.stft.transform(har_source)
        # Shapes to match CoreML Decoder_HAR inputs: (1, F, 1, T)
        har_spec_np = har_spec.detach().cpu().numpy().astype(np.float32)
        har_phase_np = har_phase.detach().cpu().numpy().astype(np.float32)
        payload["har_spec_shape"] = [1, int(har_spec_np.shape[1]), 1, int(har_spec_np.shape[2])]
        payload["har_phase_shape"] = [1, int(har_phase_np.shape[1]), 1, int(har_phase_np.shape[2])]
        payload["har_spec"] = har_spec_np.reshape(-1).tolist()
        payload["har_phase"] = har_phase_np.reshape(-1).tolist()
    except Exception as e:
        print(f"⚠️ Failed to compute HAR features for dump: {e}")

    out_json.write_text(json.dumps(payload))
    print(f"Wrote {out_json}")

if __name__ == "__main__":
    main()
