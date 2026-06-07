#!/usr/bin/env python3
"""Probe a first-party F0-noise exact-shape generator path.

This experiment tests the speed clue from ``laishere/kokoro-coreml`` without
shipping its packages. The current production generator takes a large Swift
HnSF/HAR tensor, computes source tensors that are later cropped, then runs the
generator body. This probe instead exports:

- ``noise``: ``F0_curve + style_timbre -> x_source_0/x_source_1``
- ``body``: decoder encode/decode plus generator body, emitting a discarded
  anchor and pre-tail activations
- ``tail``: fp32 ``conv_post + exp/sin + iSTFT``

It compares that candidate against the checked-in ``decoder_pre`` +
``GeneratorFromHar`` packages on the same Swift tensor dump. It also records
Core ML candidate vs PyTorch candidate metrics to separate conversion drift
from inherent audio drift against the current HAR path.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_decoder_vocoder_split import _make_decoder_vocoder_module  # noqa: E402
from probe_generator_dual_anchor_split import _make_tail_module, _maybe_palettize, _patch_cos_snake  # noqa: E402
from probe_generator_exact_geometry import _compute_units, _load_kmodel, _metrics  # noqa: E402
from probe_generator_split import _duration_label_from_dump, _precision_arg, _remove_existing_package  # noqa: E402


def _package_version(package: str) -> str | None:
    """Return installed package version for report provenance."""

    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _toolchain_report() -> dict[str, str | None]:
    """Return conversion/runtime package versions."""

    return {
        "coremltools": _package_version("coremltools"),
        "torch": _package_version("torch"),
        "numpy": _package_version("numpy"),
    }


def _np_dtype(name: str) -> type[np.floating[Any]]:
    """Return the NumPy floating dtype requested by a probe CLI option."""

    if name in ("fp16", "float16"):
        return np.float16
    if name in ("fp32", "float32"):
        return np.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _deployment_target(ct: Any, name: str) -> Any:
    """Return a Core ML deployment target from a probe CLI option."""

    targets = {
        "macos13": ct.target.macOS13,
        "ios16": ct.target.iOS16,
        "ios17": ct.target.iOS17,
    }
    try:
        return targets[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported deployment target: {name}") from exc


def _patch_resblock_rsqrt() -> None:
    """Patch decoder AdaIN residual blocks to match laishere's scale form."""

    from export_synth import wrappers

    inv_sqrt2 = 2.0 ** -0.5
    AdainResBlk1d = wrappers.kokoro_modules.AdainResBlk1d

    def _patched_forward(self: Any, x: Any, s: Any) -> Any:
        return (self._residual(x, s) + self._shortcut(x)) * inv_sqrt2

    AdainResBlk1d.forward = _patched_forward


def _patch_native_instance_norm_adain() -> None:
    """Patch AdaIN1d to export native InstanceNorm instead of manual reductions."""

    import torch
    import torch.nn as nn

    from export_synth import wrappers
    from kokoro import istftnet

    def _patched_init(self: Any, style_dim: int, num_features: int) -> None:
        nn.Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-5
        self.norm = nn.InstanceNorm1d(num_features, affine=False, eps=self.eps)
        self.fc = nn.Linear(style_dim, num_features * 2)

    def _patched_forward(self: Any, x: Any, s: Any) -> Any:
        batch, channels, _ = x.shape
        if channels != self.num_features:
            raise AssertionError(f"AdaIN1d channel mismatch: got {channels}, expected {self.num_features}")
        h = self.fc(s).view(batch, 2 * self.num_features, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1.0 + gamma) * self.norm(x) + beta

    for AdaIN1d in {istftnet.AdaIN1d, wrappers.kokoro_istftnet.AdaIN1d}:
        AdaIN1d.__init__ = _patched_init
        AdaIN1d.forward = _patched_forward


def _swift_uniform01(seed: int, count: int) -> np.ndarray:
    """Return Swift-style 24-bit uniform floats in ``[0, 1)``."""

    state = np.uint64(seed)
    values = np.empty(count, dtype=np.uint64)
    for idx in range(count):
        state = np.uint64(state ^ np.uint64(state << np.uint64(13)))
        state = np.uint64(state ^ np.uint64(state >> np.uint64(7)))
        state = np.uint64(state ^ np.uint64(state << np.uint64(17)))
        values[idx] = state
    masked = (values & np.uint64(0xFFFFFF)).astype(np.float32)
    return masked / np.float32(0xFFFFFF)


def _swift_gaussian(seed: int, count: int) -> np.ndarray:
    """Return Gaussian noise matching Swift ``generateGaussianNoise``."""

    pair_count = (count + 1) // 2
    uniforms = _swift_uniform01(seed, pair_count * 2)
    out = np.empty(pair_count * 2, dtype=np.float32)
    tiny = np.finfo(np.float32).eps
    for pair in range(pair_count):
        u1 = max(tiny, float(uniforms[2 * pair]))
        u2 = float(uniforms[2 * pair + 1])
        radius = math.sqrt(-2.0 * math.log(u1))
        theta = 2.0 * math.pi * u2
        out[2 * pair] = np.float32(radius * math.cos(theta))
        out[2 * pair + 1] = np.float32(radius * math.sin(theta))
    return out[:count]


def _make_f0_noise_module(generator: Any, phase_mode: str, source_mode: str, seed: int, f0_len: int):
    """Return ``F0 + style -> x_source_*`` module using first-party weights."""

    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from kokoro.custom_stft import CustomSTFT

    class _CoreMLSineGen(nn.Module):
        """Deterministic Core ML-friendly sine generator."""

        def __init__(self, original: Any):
            super().__init__()
            self.sine_amp = original.sine_amp
            self.noise_std = original.noise_std
            self.harmonic_num = original.harmonic_num
            self.sampling_rate = original.sampling_rate
            self.voiced_threshold = original.voiced_threshold
            self.upsample_scale = original.upsample_scale

        def forward(self, f0: Any):
            harmonics = torch.arange(
                1,
                self.harmonic_num + 2,
                device=f0.device,
                dtype=f0.dtype,
            )
            fn = f0 * harmonics.view(1, 1, -1)
            rad_values = fn / self.sampling_rate
            rv = rad_values.transpose(1, 2)
            rv_down = F.avg_pool1d(
                rv,
                kernel_size=self.upsample_scale,
                stride=self.upsample_scale,
            )
            rad_down = rv_down.transpose(1, 2)
            phase = torch.cumsum(rad_down, dim=1) * (2.0 * math.pi)
            ph = phase.transpose(1, 2) * self.upsample_scale
            ph_up = F.interpolate(
                ph,
                scale_factor=float(self.upsample_scale),
                mode="linear",
                align_corners=False,
            )
            phase = ph_up.transpose(1, 2)
            sines = torch.sin(phase) * self.sine_amp
            uv = (f0 > self.voiced_threshold).to(dtype=f0.dtype)
            noise_amp = uv * self.noise_std + (1.0 - uv) * self.sine_amp / 3.0
            noise = noise_amp * 0.01
            sine_waves = sines * uv + noise
            return sine_waves, uv, noise

    class _CoreMLSourceModule(nn.Module):
        """Source module using deterministic sine generation."""

        def __init__(self, original: Any):
            super().__init__()
            self.sine_amp = original.sine_amp
            self.l_sin_gen = _CoreMLSineGen(original.l_sin_gen)
            self.l_linear = original.l_linear
            self.l_tanh = original.l_tanh

        def forward(self, x: Any):
            sine_wavs, uv, _ = self.l_sin_gen(x)
            sine_merge = self.l_tanh(self.l_linear(sine_wavs))
            noise = torch.zeros_like(uv) * self.sine_amp / 3.0
            return sine_merge, noise, uv

    class _CoreMLSwiftLikeSourceModule(nn.Module):
        """Vectorized Swift HarmonicSource equivalent for fixed-shape export."""

        def __init__(self, original: Any, length: int, fixed_seed: int):
            super().__init__()
            sine = original.l_sin_gen
            self.sine_amp = sine.sine_amp
            self.noise_std = sine.noise_std
            self.sampling_rate = sine.sampling_rate
            self.voiced_threshold = sine.voiced_threshold
            self.upsample_scale = int(sine.upsample_scale)
            self.dim = int(sine.harmonic_num + 1)
            self.length = int(length)
            self.down_len = max(1, (self.length + self.upsample_scale - 1) // self.upsample_scale)
            self.up_len = self.down_len * self.upsample_scale
            self.l_linear = original.l_linear
            self.l_tanh = original.l_tanh
            self.register_buffer(
                "harmonics",
                torch.arange(1, self.dim + 1, dtype=torch.float32).view(1, 1, self.dim),
            )
            initials = np.zeros((self.dim,), dtype=np.float32)
            if self.dim > 1:
                initials[1:] = _swift_uniform01(fixed_seed, self.dim - 1)
            self.register_buffer("initial_phase", torch.from_numpy(initials).view(1, 1, self.dim))
            gaussian = _swift_gaussian(fixed_seed, self.dim * self.length).reshape(self.dim, self.length).T
            self.register_buffer("gaussian", torch.from_numpy(gaussian).view(1, self.length, self.dim))

        def forward(self, f0: Any):
            rad = torch.remainder(f0 * self.harmonics / self.sampling_rate, 1.0)
            rad = rad.clone()
            rad[:, 0:1, :] = rad[:, 0:1, :] + self.initial_phase
            rad_ds = F.interpolate(
                rad.transpose(1, 2),
                size=self.down_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            phase_scaled = torch.cumsum(rad_ds, dim=1) * (2.0 * math.pi * float(self.upsample_scale))
            phase_up = F.interpolate(
                phase_scaled.transpose(1, 2),
                size=self.up_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            sines = torch.sin(phase_up[:, : self.length, :]) * self.sine_amp
            uv = (f0 > self.voiced_threshold).to(dtype=f0.dtype)
            noise_amp = uv * self.noise_std + (1.0 - uv) * self.sine_amp / 3.0
            sine_waves = sines * uv + self.gaussian * noise_amp
            sine_merge = self.l_tanh(self.l_linear(sine_waves))
            return sine_merge, self.gaussian * noise_amp, uv

    class _CoreMLForwardSTFT(nn.Module):
        """Forward STFT via fixed Conv1d kernels."""

        def __init__(self, original_stft: Any, mode: str):
            super().__init__()
            self.phase_mode = mode
            self.center = original_stft.center
            self.n_fft = original_stft.n_fft
            self.pad_mode = original_stft.pad_mode
            self.freq_bins = original_stft.freq_bins
            self.conv_real = nn.Conv1d(
                1,
                self.freq_bins,
                self.n_fft,
                stride=original_stft.hop_length,
                padding=0,
                bias=False,
            )
            self.conv_imag = nn.Conv1d(
                1,
                self.freq_bins,
                self.n_fft,
                stride=original_stft.hop_length,
                padding=0,
                bias=False,
            )
            self.conv_real.weight = nn.Parameter(original_stft.weight_forward_real, requires_grad=False)
            self.conv_imag.weight = nn.Parameter(original_stft.weight_forward_imag, requires_grad=False)

        def transform(self, waveform: Any):
            if self.center:
                pad_len = self.n_fft // 2
                waveform = F.pad(waveform, (pad_len, pad_len), mode=self.pad_mode)
            x = waveform.unsqueeze(1)
            real = self.conv_real(x)
            imag = self.conv_imag(x)
            magnitude = torch.sqrt(real**2 + imag**2 + 1e-14)
            if self.phase_mode == "atan2":
                phase = torch.atan2(imag, real)
            elif self.phase_mode == "acos":
                denom = torch.sqrt(real * real + imag * imag + 1e-14)
                cos_phase = torch.clamp(real / denom, min=-1.0, max=1.0)
                abs_phase = torch.acos(cos_phase)
                sign = torch.where(imag < 0, -torch.ones_like(imag), torch.ones_like(imag))
                phase = abs_phase * sign
            elif self.phase_mode in {"atan_manual", "atan_swift"}:
                eps = torch.tensor(1e-12, dtype=real.dtype, device=real.device)
                safe_real = torch.where(real.abs() < eps, torch.where(real < 0, -eps, eps), real)
                base = torch.atan(imag / safe_real)
                pi = torch.tensor(math.pi, dtype=real.dtype, device=real.device)
                phase = torch.where(
                    real < 0,
                    torch.where(imag >= 0, base + pi, base - pi),
                    base,
                )
                phase = torch.where(
                    (real == 0) & (imag > 0),
                    pi / 2,
                    torch.where((real == 0) & (imag < 0), -pi / 2, phase),
                )
                if self.phase_mode == "atan_swift":
                    phase = torch.where(
                        (imag == 0) & (real < 0),
                        -pi,
                        phase,
                    )
            else:
                raise RuntimeError(f"unsupported phase_mode: {self.phase_mode}")
            return magnitude, phase

    class _F0NoiseModel(nn.Module):
        """Full first-party F0 noise source package."""

        def __init__(self, gen: Any):
            super().__init__()
            self.f0_upsamp = gen.f0_upsamp
            if source_mode == "current":
                self.m_source = _CoreMLSourceModule(gen.m_source)
            elif source_mode == "swift_like":
                self.m_source = _CoreMLSwiftLikeSourceModule(gen.m_source, f0_len * 300, seed)
            else:
                raise RuntimeError(f"unsupported source_mode: {source_mode}")
            fwd_stft = CustomSTFT(
                filter_length=gen.stft.filter_length,
                hop_length=gen.stft.hop_length,
                win_length=gen.stft.win_length,
            )
            self.stft = _CoreMLForwardSTFT(fwd_stft, phase_mode)
            self.noise_convs = gen.noise_convs
            self.noise_res = gen.noise_res

        def forward(self, f0_curve: Any, style_timbre: Any):
            f0 = self.f0_upsamp(f0_curve[:, None]).transpose(1, 2)
            har_source, _, _ = self.m_source(f0)
            har_source = har_source.transpose(1, 2).squeeze(1)
            har_spec, har_phase = self.stft.transform(har_source)
            har = torch.cat([har_spec, har_phase], dim=1)
            outputs = []
            for conv, res in zip(self.noise_convs, self.noise_res):
                x_source = conv(har)
                x_source = res(x_source, style_timbre)
                outputs.append(x_source)
            return tuple(outputs)

    return _F0NoiseModel(generator).eval()


def _select_inputs(tensors: dict[str, np.ndarray], natural_asr: bool) -> dict[str, np.ndarray]:
    """Return candidate inputs from a Swift tensor dump."""

    asr_key = "asr" if natural_asr and "asr" in tensors else "asr_padded"
    asr = tensors[asr_key].astype(np.float32)
    f0 = tensors["f0_padded"].astype(np.float32)
    n_input = tensors["n_padded"].astype(np.float32)
    if natural_asr:
        aligned_f0_len = min(int(f0.shape[-1]), int(asr.shape[-1]) * 2)
        f0 = f0[:, :aligned_f0_len]
        n_input = n_input[:, :aligned_f0_len]
    return {
        "asr": asr,
        "f0": f0,
        "n_input": n_input,
        "ref_s": tensors["ref_s"].astype(np.float32),
        "style_timbre": tensors["ref_s"][:, :128].astype(np.float32),
        "baseline_asr": tensors["asr_padded"].astype(np.float32),
        "baseline_f0": tensors["f0_padded"].astype(np.float32),
        "baseline_n_input": tensors["n_padded"].astype(np.float32),
        "har": tensors["har_padded"].astype(np.float32),
    }


def _export_packages(
    noise_package: Path,
    body_package: Path,
    tail_package: Path,
    tensors: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Export temporary F0-noise/body/tail packages."""

    import coremltools as ct
    import torch

    from export_synth.wrappers import remove_dropout, rewrite_generator_ups_conv_transpose

    if args.cos_snake:
        _patch_cos_snake()
    if args.patch_resblock_scale:
        _patch_resblock_rsqrt()
    if args.native_instance_norm:
        _patch_native_instance_norm_adain()
    deployment_target = _deployment_target(ct, args.deployment_target)

    kmodel = _load_kmodel()
    decoder = kmodel.decoder
    gen = decoder.generator
    rewritten_ups = 0
    if args.rewrite_ups_conv_transpose:
        rewritten_ups = rewrite_generator_ups_conv_transpose(gen)

    inputs = _select_inputs(tensors, args.natural_asr)
    asr_shape = tuple(int(v) for v in inputs["asr"].shape)
    f0_shape = tuple(int(v) for v in inputs["f0"].shape)
    n_shape = tuple(int(v) for v in inputs["n_input"].shape)
    style_shape = tuple(int(v) for v in inputs["style_timbre"].shape)

    asr = torch.zeros(asr_shape, dtype=torch.float32)
    f0 = torch.zeros(f0_shape, dtype=torch.float32)
    n_pred = torch.zeros(n_shape, dtype=torch.float32)
    style = torch.zeros(style_shape, dtype=torch.float32)

    seed = int(args.seed if args.seed is not None else args.seed_from_manifest)
    noise = _make_f0_noise_module(gen, args.phase_mode, args.source_mode, seed, int(f0_shape[-1]))
    noise_removed_dropouts = remove_dropout(noise)
    with torch.no_grad():
        traced_noise = torch.jit.trace(noise, (f0, style), strict=False, check_trace=False)
        sources = tuple(traced_noise(f0, style))
    source_shapes = [tuple(int(v) for v in source.shape) for source in sources]

    noise_model = ct.convert(
        traced_noise,
        inputs=[
            ct.TensorType(name="F0_curve", shape=f0_shape, dtype=np.float32),
            ct.TensorType(name="style_timbre", shape=style_shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name=f"x_source_{idx}") for idx in range(len(source_shapes))],
        convert_to="mlprogram",
        minimum_deployment_target=deployment_target,
        compute_precision=_precision_arg(ct, args.noise_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    noise_model = _maybe_palettize(noise_model, args.palettize_noise)
    noise_package.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_package(noise_package)
    noise_model.save(str(noise_package))

    body = _make_decoder_vocoder_module(decoder, len(source_shapes), args.anchor_mode)
    body_removed_dropouts = remove_dropout(body)
    with torch.no_grad():
        traced_body = torch.jit.trace(
            body,
            (asr, f0, n_pred, style, *sources),
            strict=False,
            check_trace=False,
        )
        anchor, pre_tail = traced_body(asr, f0, n_pred, style, *sources)
    anchor_shape = tuple(int(v) for v in anchor.shape)
    pre_tail_shape = tuple(int(v) for v in pre_tail.shape)

    body_input_dtype = _np_dtype(args.body_input_dtype)
    body_inputs = [
        ct.TensorType(name="asr", shape=asr_shape, dtype=body_input_dtype),
        ct.TensorType(name="F0_curve", shape=f0_shape, dtype=body_input_dtype),
        ct.TensorType(name="N_pred", shape=n_shape, dtype=body_input_dtype),
        ct.TensorType(name="style_timbre", shape=style_shape, dtype=body_input_dtype),
    ]
    for idx, shape in enumerate(source_shapes):
        body_inputs.append(ct.TensorType(name=f"x_source_{idx}", shape=shape, dtype=body_input_dtype))
    body_model = ct.convert(
        traced_body,
        inputs=body_inputs,
        outputs=[ct.TensorType(name="anchor"), ct.TensorType(name="pre_tail")],
        convert_to="mlprogram",
        minimum_deployment_target=deployment_target,
        compute_precision=_precision_arg(ct, args.body_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    body_model = _maybe_palettize(body_model, args.palettize_body)
    _remove_existing_package(body_package)
    body_model.save(str(body_package))

    tail = _make_tail_module(gen)
    tail_removed_dropouts = remove_dropout(tail)
    tail_input = torch.zeros(pre_tail_shape, dtype=torch.float32)
    with torch.no_grad():
        traced_tail = torch.jit.trace(tail, (tail_input,), strict=False, check_trace=False)
        tail_out = traced_tail(tail_input)
    tail_samples = int(tail_out.shape[-1])

    tail_model = ct.convert(
        traced_tail,
        inputs=[ct.TensorType(name="pre_tail", shape=pre_tail_shape, dtype=np.float32)],
        outputs=[ct.TensorType(name="waveform")],
        convert_to="mlprogram",
        minimum_deployment_target=deployment_target,
        compute_precision=_precision_arg(ct, args.tail_precision),
        compute_units=ct.ComputeUnit.ALL,
    )
    _remove_existing_package(tail_package)
    tail_model.save(str(tail_package))

    torch_candidate = None
    if args.include_torch_reference:
        with torch.no_grad():
            t_asr = torch.from_numpy(inputs["asr"].astype(np.float32))
            t_f0 = torch.from_numpy(inputs["f0"].astype(np.float32))
            t_n = torch.from_numpy(inputs["n_input"].astype(np.float32))
            t_style = torch.from_numpy(inputs["style_timbre"].astype(np.float32))
            source_values = tuple(noise(t_f0, t_style))
            _, pre_tail_ref = body(t_asr, t_f0, t_n, t_style, *source_values)
            torch_candidate = tail(pre_tail_ref).detach().cpu().numpy().astype(np.float32)

    return {
        "toolchain": _toolchain_report(),
        "deployment_target": args.deployment_target,
        "phase_mode": args.phase_mode,
        "source_mode": args.source_mode,
        "seed": seed,
        "noise_package": str(noise_package),
        "body_package": str(body_package),
        "tail_package": str(tail_package),
        "natural_asr": bool(args.natural_asr),
        "cos_snake": bool(args.cos_snake),
        "patch_resblock_scale": bool(args.patch_resblock_scale),
        "palettize_noise": bool(args.palettize_noise),
        "palettize_body": bool(args.palettize_body),
        "native_instance_norm": bool(args.native_instance_norm),
        "rewrite_ups_conv_transpose": bool(args.rewrite_ups_conv_transpose),
        "rewritten_upsample_layers": int(rewritten_ups),
        "noise_precision": args.noise_precision,
        "body_precision": args.body_precision,
        "body_input_dtype": args.body_input_dtype,
        "tail_precision": args.tail_precision,
        "asr_shape": list(asr_shape),
        "f0_shape": list(f0_shape),
        "n_shape": list(n_shape),
        "style_shape": list(style_shape),
        "source_shapes": [list(shape) for shape in source_shapes],
        "anchor_shape": list(anchor_shape),
        "pre_tail_shape": list(pre_tail_shape),
        "tail_samples": tail_samples,
        "noise_removed_dropouts": noise_removed_dropouts,
        "body_removed_dropouts": body_removed_dropouts,
        "tail_removed_dropouts": tail_removed_dropouts,
        "torch_candidate": torch_candidate,
    }


def _predict(model: Any, feed: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], float]:
    start = time.perf_counter()
    out = model.predict(feed)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return {key: np.asarray(value) for key, value in out.items()}, elapsed_ms


def _load_models(args: argparse.Namespace, noise_package: Path, body_package: Path, tail_package: Path):
    import coremltools as ct

    decoder_pre = ct.models.MLModel(
        str(args.decoder_pre_package),
        compute_units=_compute_units(ct, args.decoder_pre_compute_units),
    )
    fused = ct.models.MLModel(
        str(args.fused_package),
        compute_units=_compute_units(ct, args.fused_compute_units),
    )
    noise = ct.models.MLModel(
        str(noise_package),
        compute_units=_compute_units(ct, args.noise_compute_units),
    )
    body = ct.models.MLModel(
        str(body_package),
        compute_units=_compute_units(ct, args.body_compute_units),
    )
    tail = ct.models.MLModel(
        str(tail_package),
        compute_units=_compute_units(ct, args.tail_compute_units),
    )
    return decoder_pre, fused, noise, body, tail


def _baseline_predict(decoder_pre: Any, fused: Any, inputs: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    dec_feed = {
        "asr": inputs["baseline_asr"],
        "f0": inputs["baseline_f0"][:, None, :],
        "n_input": inputs["baseline_n_input"][:, None, :],
        "ref_s": inputs["ref_s"],
    }
    dec_out, dec_ms = _predict(decoder_pre, dec_feed)
    x_pre = dec_out["x_pre"].astype(np.float32)
    gen_out, gen_ms = _predict(
        fused,
        {"x_pre": x_pre, "ref_s": inputs["ref_s"], "har": inputs["har"]},
    )
    waveform = gen_out.get("waveform", next(iter(gen_out.values()))).astype(np.float32)
    return waveform, {"decoder_pre_ms": dec_ms, "generator_ms": gen_ms, "total_ms": dec_ms + gen_ms}


def _candidate_predict(
    noise: Any,
    body: Any,
    tail: Any,
    inputs: dict[str, np.ndarray],
    body_input_dtype: type[np.floating[Any]],
) -> tuple[np.ndarray, dict[str, float]]:
    noise_out, noise_ms = _predict(
        noise,
        {"F0_curve": inputs["f0"], "style_timbre": inputs["style_timbre"]},
    )
    body_feed = {
        "asr": inputs["asr"].astype(body_input_dtype),
        "F0_curve": inputs["f0"].astype(body_input_dtype),
        "N_pred": inputs["n_input"].astype(body_input_dtype),
        "style_timbre": inputs["style_timbre"].astype(body_input_dtype),
    }
    for idx in range(len(noise_out)):
        key = f"x_source_{idx}"
        body_feed[key] = noise_out[key].astype(body_input_dtype)
    body_out, body_ms = _predict(body, body_feed)
    pre_tail = body_out["pre_tail"].astype(np.float32)
    tail_out, tail_ms = _predict(tail, {"pre_tail": pre_tail})
    waveform = tail_out.get("waveform", next(iter(tail_out.values()))).astype(np.float32)
    return waveform, {"noise_ms": noise_ms, "body_ms": body_ms, "tail_ms": tail_ms, "total_ms": noise_ms + body_ms + tail_ms}


def _benchmark(
    args: argparse.Namespace,
    tensors: dict[str, np.ndarray],
    noise_package: Path,
    body_package: Path,
    tail_package: Path,
) -> dict[str, Any]:
    inputs = _select_inputs(tensors, args.natural_asr)
    decoder_pre, fused, noise, body, tail = _load_models(args, noise_package, body_package, tail_package)
    body_input_dtype = _np_dtype(args.body_input_dtype)

    baseline_first, baseline_first_times = _baseline_predict(decoder_pre, fused, inputs)
    candidate_first, candidate_first_times = _candidate_predict(noise, body, tail, inputs, body_input_dtype)

    for _ in range(max(0, args.warmup)):
        _baseline_predict(decoder_pre, fused, inputs)
        _candidate_predict(noise, body, tail, inputs, body_input_dtype)

    baseline_decoder_times: list[float] = []
    baseline_generator_times: list[float] = []
    baseline_total_times: list[float] = []
    candidate_noise_times: list[float] = []
    candidate_body_times: list[float] = []
    candidate_tail_times: list[float] = []
    candidate_total_times: list[float] = []
    last_baseline = baseline_first
    last_candidate = candidate_first
    for _ in range(max(1, args.iterations)):
        last_baseline, baseline_times = _baseline_predict(decoder_pre, fused, inputs)
        last_candidate, candidate_times = _candidate_predict(noise, body, tail, inputs, body_input_dtype)
        baseline_decoder_times.append(baseline_times["decoder_pre_ms"])
        baseline_generator_times.append(baseline_times["generator_ms"])
        baseline_total_times.append(baseline_times["total_ms"])
        candidate_noise_times.append(candidate_times["noise_ms"])
        candidate_body_times.append(candidate_times["body_ms"])
        candidate_tail_times.append(candidate_times["tail_ms"])
        candidate_total_times.append(candidate_times["total_ms"])

    trim_len = min(int(tensors["waveform"].size), int(last_candidate.size), int(last_baseline.size))
    baseline_trim = last_baseline.reshape(-1)[:trim_len]
    candidate_trim = last_candidate.reshape(-1)[:trim_len]
    dump_trim = tensors["waveform"].reshape(-1)[:trim_len]

    return {
        "toolchain": _toolchain_report(),
        "decoder_pre_compute_units": args.decoder_pre_compute_units,
        "fused_compute_units": args.fused_compute_units,
        "noise_compute_units": args.noise_compute_units,
        "body_compute_units": args.body_compute_units,
        "tail_compute_units": args.tail_compute_units,
        "warmup": int(max(0, args.warmup)),
        "iterations": int(max(1, args.iterations)),
        "first_predict_ms": {
            "baseline_decoder_pre": float(baseline_first_times["decoder_pre_ms"]),
            "baseline_generator": float(baseline_first_times["generator_ms"]),
            "baseline_total": float(baseline_first_times["total_ms"]),
            "candidate_noise": float(candidate_first_times["noise_ms"]),
            "candidate_body": float(candidate_first_times["body_ms"]),
            "candidate_tail": float(candidate_first_times["tail_ms"]),
            "candidate_total": float(candidate_first_times["total_ms"]),
        },
        "warm_predict_times_ms": {
            "baseline_decoder_pre": baseline_decoder_times,
            "baseline_generator": baseline_generator_times,
            "baseline_total": baseline_total_times,
            "candidate_noise": candidate_noise_times,
            "candidate_body": candidate_body_times,
            "candidate_tail": candidate_tail_times,
            "candidate_total": candidate_total_times,
        },
        "warm_predict_median_ms": {
            "baseline_decoder_pre": float(statistics.median(baseline_decoder_times)),
            "baseline_generator": float(statistics.median(baseline_generator_times)),
            "baseline_total": float(statistics.median(baseline_total_times)),
            "candidate_noise": float(statistics.median(candidate_noise_times)),
            "candidate_body": float(statistics.median(candidate_body_times)),
            "candidate_tail": float(statistics.median(candidate_tail_times)),
            "candidate_total": float(statistics.median(candidate_total_times)),
        },
        "metrics": {
            "baseline_vs_dump_trimmed": _metrics(dump_trim, baseline_trim),
            "candidate_vs_dump_trimmed": _metrics(dump_trim, candidate_trim),
            "candidate_vs_baseline_trimmed": _metrics(baseline_trim, candidate_trim),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest, tensors = load_tensor_dump(args.tensor_dump)
    args.seed_from_manifest = int(manifest.get("metadata", {}).get("seed", 42))
    required = [
        "asr_padded",
        "f0_padded",
        "n_padded",
        "ref_s",
        "har_padded",
        "waveform",
    ]
    if args.natural_asr:
        required.append("asr")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    label = args.label or _duration_label_from_dump(args.tensor_dump, manifest)
    if args.natural_asr:
        label = f"{label}_natural_asr"
    if args.cos_snake:
        label = f"{label}_cos"
    if args.patch_resblock_scale:
        label = f"{label}_rsqrt"
    if args.palettize_noise:
        label = f"{label}_noise_pal"
    if args.palettize_body:
        label = f"{label}_body_pal"
    if args.native_instance_norm:
        label = f"{label}_native_in"
    if args.source_mode != "current":
        label = f"{label}_{args.source_mode}"

    work_dir = args.output_dir / label
    noise_package = work_dir / f"kokoro_f0_noise_{label}.mlpackage"
    body_package = work_dir / f"kokoro_f0_noise_body_{label}.mlpackage"
    tail_package = work_dir / f"kokoro_f0_noise_tail_{label}.mlpackage"
    report_name = Path(args.report_name)
    if report_name.name != str(report_name):
        raise SystemExit(f"--report-name must be a filename, got {args.report_name!r}")
    report_path = work_dir / report_name

    export_report: dict[str, Any] | None = None
    torch_candidate = None
    if args.skip_export:
        missing_packages = [
            str(path)
            for path in (noise_package, body_package, tail_package)
            if not path.is_dir()
        ]
        if missing_packages:
            raise SystemExit(f"--skip-export requested but packages are missing: {missing_packages}")
    else:
        export_report = _export_packages(noise_package, body_package, tail_package, tensors, args)
        torch_candidate = export_report.pop("torch_candidate", None)

    benchmark = _benchmark(args, tensors, noise_package, body_package, tail_package)
    if torch_candidate is not None:
        trim_len = min(int(torch_candidate.size), int(tensors["waveform"].size))
        # Re-run a final candidate cheaply through Core ML would distort timing;
        # use the last benchmark metric only for Core ML vs baseline, and store
        # PyTorch vs dump as the inherent path reference.
        benchmark["metrics"]["torch_candidate_vs_dump_trimmed"] = _metrics(
            tensors["waveform"].reshape(-1)[:trim_len],
            torch_candidate.reshape(-1)[:trim_len],
        )

    metrics = benchmark["metrics"]["candidate_vs_baseline_trimmed"]
    passes = bool(
        metrics["correlation"] is not None
        and metrics["correlation"] >= args.min_corr
        and metrics["snr_db"] >= args.min_snr
        and metrics["max_abs_error"] <= args.max_abs_error
    )
    med = benchmark["warm_predict_median_ms"]
    speedup_vs_baseline_pct = None
    if med["baseline_total"] > 0:
        speedup_vs_baseline_pct = 100.0 * (med["baseline_total"] - med["candidate_total"]) / med["baseline_total"]

    report = {
        "tensor_dump": str(args.tensor_dump),
        "decoder_pre_package": str(args.decoder_pre_package),
        "fused_package": str(args.fused_package),
        "noise_package": str(noise_package),
        "body_package": str(body_package),
        "tail_package": str(tail_package),
        "report": str(report_path),
        "manifest_metadata": manifest.get("metadata", {}),
        "export": export_report,
        "benchmark": benchmark,
        "thresholds": {
            "min_corr": args.min_corr,
            "min_snr": args.min_snr,
            "max_abs_error": args.max_abs_error,
        },
        "speedup_vs_baseline_pct": speedup_vs_baseline_pct,
        "passes": passes,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dump", type=Path, default=Path("outputs/generator_isolation/dumps/3s"))
    parser.add_argument("--decoder-pre-package", type=Path, default=Path("coreml/kokoro_decoder_pre_3s.mlpackage"))
    parser.add_argument("--fused-package", type=Path, default=Path("coreml/kokoro_decoder_har_post_3s.mlpackage"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/f0_noise_exact_shape"))
    parser.add_argument("--label", default=None)
    parser.add_argument("--report-name", default="report.json")
    parser.add_argument("--natural-asr", action="store_true")
    parser.add_argument("--anchor-mode", default="mean", choices=("mean", "slice_mean"))
    parser.add_argument("--cos-snake", action="store_true")
    parser.add_argument("--patch-resblock-scale", action="store_true")
    parser.add_argument("--palettize-noise", action="store_true")
    parser.add_argument("--palettize-body", action="store_true")
    parser.add_argument("--native-instance-norm", action="store_true")
    parser.add_argument("--noise-precision", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--body-precision", default="fp16", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--body-input-dtype", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--tail-precision", default="fp32", choices=("fp16", "float16", "fp32", "float32"))
    parser.add_argument("--deployment-target", default="macos13", choices=("macos13", "ios16", "ios17"))
    parser.add_argument("--phase-mode", default="atan2", choices=("atan2", "acos", "atan_manual", "atan_swift"))
    parser.add_argument("--source-mode", default="current", choices=("current", "swift_like"))
    parser.add_argument("--rewrite-ups-conv-transpose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--decoder-pre-compute-units", default="cpuAndNeuralEngine")
    parser.add_argument("--fused-compute-units", default="cpuAndGPU")
    parser.add_argument("--noise-compute-units", default="all")
    parser.add_argument("--body-compute-units", default="cpuAndGPU")
    parser.add_argument("--tail-compute-units", default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--min-corr", type=float, default=0.99)
    parser.add_argument("--min-snr", type=float, default=35.0)
    parser.add_argument("--max-abs-error", type=float, default=1e-2)
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--include-torch-reference", action="store_true")
    parser.add_argument("--fail-on-difference", action="store_true")
    args = parser.parse_args()

    report = run(args)
    med = report["benchmark"]["warm_predict_median_ms"]
    metrics = report["benchmark"]["metrics"]["candidate_vs_baseline_trimmed"]
    print(
        "f0_noise_exact_shape "
        f"passes={report['passes']} "
        f"label={Path(report['noise_package']).parent.name} "
        f"baseline_median_ms={med['baseline_total']:.3f} "
        f"candidate_median_ms={med['candidate_total']:.3f} "
        f"baseline_decoder_pre_ms={med['baseline_decoder_pre']:.3f} "
        f"baseline_generator_ms={med['baseline_generator']:.3f} "
        f"candidate_noise_ms={med['candidate_noise']:.3f} "
        f"candidate_body_ms={med['candidate_body']:.3f} "
        f"candidate_tail_ms={med['candidate_tail']:.3f} "
        f"speedup_vs_baseline_pct={report['speedup_vs_baseline_pct']:.2f} "
        f"corr={metrics['correlation']} "
        f"snr_db={metrics['snr_db']:.2f} "
        f"max_abs={metrics['max_abs_error']:.6g} "
        f"report={report['report']}"
    )
    if args.fail_on_difference and not report["passes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
