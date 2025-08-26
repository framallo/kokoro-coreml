#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

# Ensure repository root on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kokoro.custom_stft import CustomSTFT


def write_wav_int16(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    # Clamp to [-1,1], convert to 16-bit PCM, write WAV via wave module
    import wave
    import struct
    x = np.clip(samples.astype(np.float64), -1.0, 1.0)
    ints = (x * 32767.0).astype(np.int16)
    with wave.open(str(path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(int(sample_rate))
        wf.writeframes(ints.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent', required=True, help='Path to latent CSV (C x T)')
    ap.add_argument('--out_wav', required=True, help='Path to output WAV')
    ap.add_argument('--n_fft', type=int, default=20)
    ap.add_argument('--hop', type=int, default=5)
    ap.add_argument('--sr', type=int, default=24000)
    args = ap.parse_args()

    latent_path = Path(args.latent)
    out_wav = Path(args.out_wav)

    # Load latent as C x T
    x = np.loadtxt(str(latent_path), delimiter=',', dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    C, T = x.shape
    freq_bins = args.n_fft // 2 + 1
    assert C >= 2 * freq_bins, f"latent channels {C} < required {2 * freq_bins} for n_fft={args.n_fft}"

    spec_log = x[:freq_bins, :]
    phase_raw = x[freq_bins:freq_bins + freq_bins, :]

    with torch.no_grad():
        stft = CustomSTFT(filter_length=args.n_fft, hop_length=args.hop, win_length=args.n_fft)
        spec_t = torch.exp(torch.from_numpy(spec_log)).unsqueeze(0)
        phase_t = torch.sin(torch.from_numpy(phase_raw)).unsqueeze(0)
        y = stft.inverse(spec_t, phase_t)
        y = y.squeeze(0).cpu().numpy()

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    write_wav_int16(out_wav, y, args.sr)
    print(f"Wrote WAV: {out_wav}")


if __name__ == '__main__':
    main()
