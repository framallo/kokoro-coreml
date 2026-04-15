#!/usr/bin/env python3
"""Compare two decoder-har-post .mlpackage outputs on identical PyTorch-built inputs.

Mirrors tensor prep in ``kokoro.synthesis_backends.decoder_har_post_bucket_impl``.
Uses plan guardrails: Pearson r, SNR, max abs Δ on float32 (see ane-optimization-v1.md).

Example::

    uv run python scripts/compare_decoder_har_post_waveforms.py \\
      --baseline /tmp/kokoro_har_post_baseline_3s.mlpackage \\
      --candidate coreml/kokoro_decoder_har_post_3s.mlpackage \\
      --bucket-sec 3
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch

import coremltools as ct

from kokoro.conv_length import conv1d_output_length_from_module
from kokoro.coreml_pipeline import HybridTTSPipeline


def _predict_waveform(model: ct.models.MLModel, inputs: dict) -> np.ndarray:
    res = model.predict(inputs)
    key = "waveform" if "waveform" in res else list(res.keys())[0]
    return np.asarray(res[key], dtype=np.float32).squeeze()


def _metrics(ref: np.ndarray, cand: np.ndarray):
    ref = ref.astype(np.float64, copy=False)
    cand = cand.astype(np.float64, copy=False)
    n = min(ref.size, cand.size)
    ref = ref.flatten()[:n]
    cand = cand.flatten()[:n]
    max_delta = float(np.max(np.abs(ref - cand)))
    err = ref - cand
    snr = 10.0 * np.log10((np.sum(ref * ref) + 1e-12) / (np.sum(err * err) + 1e-12))
    rms = float(np.sqrt(np.mean(ref * ref)))
    pearson = None
    if rms >= 1e-4:
        pearson = float(np.corrcoef(ref, cand)[0, 1])
    return pearson, float(snr), max_delta, rms


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=str, required=True, help="Path to baseline .mlpackage")
    p.add_argument("--candidate", type=str, required=True, help="Path to candidate .mlpackage")
    p.add_argument("--bucket-sec", type=int, required=True, choices=(3, 10))
    p.add_argument("--text", type=str, default="Hello from Kokoro.")
    p.add_argument("--voice", type=str, default="af_heart")
    p.add_argument("--speed", type=float, default=1.0)
    args = p.parse_args()

    torch.manual_seed(0)
    # Pipeline loads KModel + optional Core ML; we only need PyTorch for x_pre/har.
    pipe = HybridTTSPipeline(force_engine="pytorch")
    sec = args.bucket_sec
    base = ct.models.MLModel(args.baseline)
    cand = ct.models.MLModel(args.candidate)
    spec_b = base.get_spec()
    shapes = {i.name: list(i.type.multiArrayType.shape) for i in spec_b.description.input}
    x_pre_shape = shapes["x_pre"]
    har_shape = shapes["har"]
    asr_len = int(x_pre_shape[-1])
    har_t = int(har_shape[-1])

    vi = pipe.extract_vocoder_inputs(args.text, args.voice, args.speed)
    if vi is None:
        print("extract_vocoder_inputs failed", file=sys.stderr)
        return 1
    T_f0 = int(vi["f0_curve"].shape[-1])
    dec = pipe.pytorch_model.decoder
    gen = dec.generator
    f0_samples_per_step = int(round(float(gen.f0_upsamp.scale_factor)))
    bucket_samples = sec * 24000
    full_f0_len = int(round(bucket_samples / float(f0_samples_per_step)))
    frame_count = conv1d_output_length_from_module(full_f0_len, dec.F0_conv)

    asr = vi["asr"].astype(np.float32)
    f0 = vi["f0_curve"].astype(np.float32)
    n = vi["n"].astype(np.float32)
    ref_s = vi["ref_s"].astype(np.float32)

    asr_pad = np.zeros((1, 512, frame_count), dtype=np.float32)
    t_asr = min(frame_count, asr.shape[-1])
    asr_pad[:, :, :t_asr] = asr[:, :, :t_asr]
    f0_pad = np.zeros((1, full_f0_len), dtype=np.float32)
    n_pad = np.zeros((1, full_f0_len), dtype=np.float32)
    t_f0 = min(full_f0_len, f0.shape[-1])
    f0_pad[:, :t_f0] = f0[:, :t_f0]
    n_pad[:, :t_f0] = n[:, :t_f0]

    with torch.no_grad():
        ref_t = torch.from_numpy(ref_s)
        s = ref_t[:, :128]
        asr_t = torch.from_numpy(asr_pad)
        F0 = dec.F0_conv(torch.from_numpy(f0_pad).unsqueeze(1))
        N = dec.N_conv(torch.from_numpy(n_pad).unsqueeze(1))
        x = torch.cat([asr_t, F0, N], dim=1)
        x = dec.encode(x, s)
        asr_res = dec.asr_res(asr_t)
        res = True
        for block in dec.decode:
            if res:
                x = torch.cat([x, asr_res, F0, N], dim=1)
            x = block(x, s)
            if block.upsample_type != "none":
                res = False
        x_pre = x
        f0_up = gen.f0_upsamp(torch.from_numpy(f0_pad)[:, None]).transpose(1, 2)
        har_source, _, _ = gen.m_source(f0_up)
        har_source = har_source.transpose(1, 2).squeeze(1)
        har_spec, har_phase = gen.stft.transform(har_source)
        har = torch.cat([har_spec, har_phase], dim=1)
        har_np = har.numpy().astype(np.float32)

    x_pre_np = x_pre.cpu().numpy().astype(np.float32)
    if x_pre_np.shape[-1] != asr_len:
        aligned = np.zeros((x_pre_np.shape[0], x_pre_np.shape[1], asr_len), dtype=np.float32)
        c = min(x_pre_np.shape[-1], asr_len)
        aligned[:, :, :c] = x_pre_np[:, :, :c]
        x_pre_np = aligned
    if har_np.shape[-1] != har_t:
        h_new = np.zeros((har_np.shape[0], har_np.shape[1], har_t), dtype=np.float32)
        cpy = min(har_np.shape[-1], har_t)
        h_new[:, :, :cpy] = har_np[:, :, :cpy]
        har_np = h_new

    inputs = {"x_pre": x_pre_np, "ref_s": ref_s, "har": har_np}
    w0 = _predict_waveform(base, inputs)
    w1 = _predict_waveform(cand, inputs)
    target_len = int(round((T_f0 / 80.0) * 24000.0))
    w0 = w0[: min(int(w0.shape[-1]), target_len)]
    w1 = w1[: min(int(w1.shape[-1]), target_len)]

    pearson, snr, max_delta, rms = _metrics(w0, w1)
    print(f"ref_rms={rms:.6e} pearson={pearson} snr_db={snr:.2f} max_abs_delta={max_delta:.6e}")
    ok = max_delta <= 1e-2 and snr >= 40.0
    if pearson is not None:
        ok = ok and pearson > 0.99
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
