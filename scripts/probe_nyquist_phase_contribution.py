#!/usr/bin/env python3
"""Probe whether HAR Nyquist phase can be removed from the generator contract.

The compact ``har_source -> waveform`` path is fast, but strict parity fails
because the raw Nyquist phase channel uses a different ``+pi/-pi`` branch than
the current Swift dump. This probe stays in PyTorch and tests the next simple
hypothesis: if the generator barely uses that feature, we can zero or fold it
and keep the shorter HAR source boundary.

Outputs are reports under ``outputs/`` only. Shipping Core ML packages are not
modified.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_ROOT))

from audio_parity_tensor_io import load_tensor_dump  # noqa: E402
from probe_generator_exact_geometry import _load_kmodel, _metrics  # noqa: E402

NYQUIST_BIN = 10
NYQUIST_HAR_CHANNEL = 21
SWIFT_NYQUIST_REAL_BASIS = np.array(
    [
        0.000000000000000000000000000000e00,
        -2.447172999382019042968750000000e-02,
        9.549149870872497558593750000000e-02,
        -2.061073780059814453125000000000e-01,
        3.454914689064025878906250000000e-01,
        -4.999999701976776123046875000000e-01,
        6.545085310935974121093750000000e-01,
        -7.938926219940185546875000000000e-01,
        9.045084714889526367187500000000e-01,
        -9.755282402038574218750000000000e-01,
        1.000000000000000000000000000000e00,
        -9.755282402038574218750000000000e-01,
        9.045084714889526367187500000000e-01,
        -7.938927412033081054687500000000e-01,
        6.545085310935974121093750000000e-01,
        -5.000002384185791015625000000000e-01,
        3.454916477203369140625000000000e-01,
        -2.061074674129486083984375000000e-01,
        9.549167752265930175781250000000e-02,
        -2.447181940078735351562500000000e-02,
    ],
    dtype=np.float32,
)
SWIFT_NYQUIST_IMAG_BASIS = np.array(
    [
        -0.000000000000000000000000000000e00,
        -3.695128425462712584703695029020e-09,
        2.883763094985170027939602732658e-08,
        4.915611917510886996751651167870e-09,
        2.086710395587942912243306636810e-07,
        -6.159079930512234568595886230469e-07,
        -3.121974501141266955528408288956e-08,
        -4.605636263477208558470010757446e-07,
        1.092615889319858979433774948120e-06,
        -1.790874080143112223595380783081e-06,
        2.463632199578569270670413970947e-06,
        -1.155139216280076652765274047852e-06,
        -8.628924774711776990443468093872e-08,
        -1.936925627887831069529056549072e-06,
        7.594045428049867041409015655518e-07,
        -1.847725116022047586739063262939e-06,
        8.346846129825280513614416122437e-07,
        -2.342240605912593309767544269562e-07,
        3.506071095671359216794371604919e-07,
        -5.853862106164342549163848161697e-08,
    ],
    dtype=np.float32,
)


def _manual_stft_har(generator: Any, har_source_np: np.ndarray):
    """Recompute HAR magnitude/phase with CoreML-safe manual atan semantics."""

    import torch
    import torch.nn.functional as F

    from kokoro.custom_stft import CustomSTFT

    stft = CustomSTFT(
        filter_length=generator.stft.filter_length,
        hop_length=generator.stft.hop_length,
        win_length=generator.stft.win_length,
    )
    waveform = torch.from_numpy(har_source_np.astype(np.float32))
    if stft.center:
        waveform = F.pad(waveform, (stft.n_fft // 2, stft.n_fft // 2), mode=stft.pad_mode)
    x = waveform.unsqueeze(1)
    real = F.conv1d(x, stft.weight_forward_real, stride=stft.hop_length)
    imag = F.conv1d(x, stft.weight_forward_imag, stride=stft.hop_length)
    magnitude = torch.sqrt(real**2 + imag**2 + 1e-14)

    eps = torch.full_like(real, 1e-12)
    safe_real = torch.where(torch.abs(real) < eps, torch.where(real < 0.0, -eps, eps), real)
    base = torch.atan(imag / safe_real)
    phase = torch.where(
        real < 0.0,
        torch.where(imag >= 0.0, base + torch.pi, base - torch.pi),
        base,
    )
    phase = torch.where(
        torch.abs(real) < eps,
        torch.where(
            imag > 0.0,
            torch.full_like(imag, torch.pi / 2.0),
            torch.where(imag < 0.0, torch.full_like(imag, -torch.pi / 2.0), torch.zeros_like(imag)),
        ),
        phase,
    )
    return torch.cat([magnitude, phase], dim=1)


def _run_generator_from_har(generator: Any, tensors: dict[str, np.ndarray], har: Any):
    """Run the generator body from an already-built HAR tensor."""

    import torch
    import torch.nn.functional as F

    x = torch.from_numpy(tensors["x_pre_padded"].astype(np.float32))
    ref_s = torch.from_numpy(tensors["ref_s"].astype(np.float32))
    s = ref_s[:, :128]

    gen = generator
    for i in range(gen.num_upsamples):
        x = F.leaky_relu(x, negative_slope=0.1)
        x_source = gen.noise_convs[i](har)
        x_source = gen.noise_res[i](x_source, s)
        x = gen.ups[i](x)
        if i == gen.num_upsamples - 1:
            x = gen.reflection_pad(x)
        tx = x.size(2)
        ts = x_source.size(2)
        if ts < tx:
            x_source = F.pad(x_source, (0, tx - ts))
        elif ts > tx:
            x_source = x_source[:, :, :tx]
        x = x + x_source
        xs = None
        for j in range(gen.num_kernels):
            y = gen.resblocks[i * gen.num_kernels + j](x, s)
            xs = y if xs is None else xs + y
        x = xs / gen.num_kernels
    x = F.leaky_relu(x)
    logits = gen.conv_post(x)
    spec = torch.exp(logits[:, : gen.post_n_fft // 2 + 1, :])
    phase = torch.sin(logits[:, gen.post_n_fft // 2 + 1 :, :])
    return gen.stft.inverse(spec, phase)


def _phase_channel_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    raw = _metrics(reference, candidate)
    wrapped_delta = np.angle(np.exp(1j * (reference.astype(np.float64) - candidate.astype(np.float64))))
    raw["wrapped_mean_abs_error"] = float(np.mean(np.abs(wrapped_delta)))
    raw["wrapped_max_abs_error"] = float(np.max(np.abs(wrapped_delta)))
    raw["two_pi_branch_errors"] = int(np.sum(np.abs(reference - candidate) > math.pi))
    return raw


def _swift_basis_nyquist_components(har_source_np: np.ndarray, frame_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the Swift HnSF Nyquist dot products from its Float basis."""

    source = har_source_np.astype(np.float32, copy=False)
    if source.ndim != 2:
        raise ValueError(f"expected har_source shape [B, T], got {source.shape}")
    padded = np.pad(source, ((0, 0), (10, 10)), mode="edge")
    real_out = np.empty((source.shape[0], frame_count), dtype=np.float32)
    imag_out = np.empty((source.shape[0], frame_count), dtype=np.float32)
    for frame_index in range(frame_count):
        start = frame_index * 5
        window = padded[:, start : start + SWIFT_NYQUIST_IMAG_BASIS.shape[0]]
        real_out[:, frame_index] = np.sum(window * SWIFT_NYQUIST_REAL_BASIS, axis=1, dtype=np.float32)
        imag_out[:, frame_index] = np.sum(window * SWIFT_NYQUIST_IMAG_BASIS, axis=1, dtype=np.float32)
    return real_out, imag_out


def _swift_basis_nyquist_branch_phase(har_source_np: np.ndarray, frame_count: int) -> np.ndarray:
    """Predict only the Swift HnSF Nyquist +/-pi branch."""

    _, imag = _swift_basis_nyquist_components(har_source_np, frame_count)
    return np.where(imag >= 0.0, math.pi, -math.pi).astype(np.float32)


def _swift_basis_nyquist_atan2_phase(har_source_np: np.ndarray, frame_count: int) -> np.ndarray:
    """Predict the full Swift HnSF Nyquist atan2 phase from Float residuals."""

    real, imag = _swift_basis_nyquist_components(har_source_np, frame_count)
    return np.arctan2(imag, real).astype(np.float32)


def _weight_stats(generator: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, conv in enumerate(generator.noise_convs):
        weight = conv.weight.detach().cpu().numpy().astype(np.float64)
        nyq = weight[:, NYQUIST_HAR_CHANNEL, :]
        all_abs = float(np.sum(np.abs(weight)))
        all_l2 = float(np.linalg.norm(weight))
        nyq_abs = float(np.sum(np.abs(nyq)))
        nyq_l2 = float(np.linalg.norm(nyq))
        rows.append(
            {
                "noise_conv_index": index,
                "weight_shape": [int(v) for v in weight.shape],
                "nyquist_abs_fraction": nyq_abs / all_abs if all_abs else None,
                "nyquist_l2_fraction": nyq_l2 / all_l2 if all_l2 else None,
                "nyquist_weight_mean": float(np.mean(nyq)),
                "nyquist_weight_abs_mean": float(np.mean(np.abs(nyq))),
                "nyquist_weight_max_abs": float(np.max(np.abs(nyq))),
            }
        )
    return rows


def _pad_or_trim_har(har: Any, target_time: int | None):
    import torch.nn.functional as F

    if target_time is None:
        return har
    current = int(har.size(2))
    if current < target_time:
        return F.pad(har, (0, target_time - current))
    if current > target_time:
        return har[:, :, :target_time]
    return har


def _make_variants(tensors: dict[str, np.ndarray], recomputed_har: Any, pad_har_to: int | None) -> dict[str, Any]:
    import torch

    dumped_har = torch.from_numpy(tensors["har"].astype(np.float32))
    dumped_phase = torch.from_numpy(tensors["har_phase"].astype(np.float32))
    dumped_nyquist = dumped_phase[:, NYQUIST_BIN, :]
    recomputed_nyquist = recomputed_har[:, NYQUIST_HAR_CHANNEL, :]
    mean_nyquist = float(dumped_nyquist.mean().item())

    variants: dict[str, Any] = {}
    variants["dumped_har"] = dumped_har
    if "har_padded" in tensors:
        variants["dumped_har_padded"] = torch.from_numpy(tensors["har_padded"].astype(np.float32))
    variants["dumped_har_zero_nyquist"] = dumped_har.clone()
    variants["dumped_har_zero_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = 0.0
    variants["dumped_har_mean_nyquist"] = dumped_har.clone()
    variants["dumped_har_mean_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = mean_nyquist

    variants["recomputed_manual"] = recomputed_har
    variants["recomputed_manual_dumped_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_dumped_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = dumped_nyquist
    variants["recomputed_manual_zero_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_zero_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = 0.0
    variants["recomputed_manual_mean_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_mean_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = mean_nyquist
    variants["recomputed_manual_pos_pi_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_pos_pi_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = math.pi
    variants["recomputed_manual_neg_pi_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_neg_pi_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = -math.pi

    # Oracle-fitted scalar/affine repairs are not deployable by themselves, but
    # they cheaply falsify the tempting idea that one global calibration can
    # replace the branch-sensitive Nyquist convention.
    x = recomputed_nyquist.detach().cpu().numpy().reshape(-1).astype(np.float64)
    y = dumped_nyquist.detach().cpu().numpy().reshape(-1).astype(np.float64)
    x_var = float(np.var(x))
    scale = float(np.cov(x, y, bias=True)[0, 1] / x_var) if x_var > 0.0 else 0.0
    bias = float(np.mean(y) - scale * np.mean(x))
    variants["recomputed_manual_affine_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_affine_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = (
        recomputed_nyquist * scale + bias
    )
    variants["recomputed_manual_negated_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_negated_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = -recomputed_nyquist
    swift_basis_nyquist = torch.from_numpy(
        _swift_basis_nyquist_branch_phase(tensors["har_source"], int(recomputed_har.size(2)))
    )
    variants["recomputed_manual_swift_basis_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_swift_basis_nyquist"][:, NYQUIST_HAR_CHANNEL, :] = swift_basis_nyquist
    swift_basis_atan2_nyquist = torch.from_numpy(
        _swift_basis_nyquist_atan2_phase(tensors["har_source"], int(recomputed_har.size(2)))
    )
    variants["recomputed_manual_swift_basis_atan2_nyquist"] = recomputed_har.clone()
    variants["recomputed_manual_swift_basis_atan2_nyquist"][
        :, NYQUIST_HAR_CHANNEL, :
    ] = swift_basis_atan2_nyquist
    return {name: _pad_or_trim_har(har, pad_har_to) for name, har in variants.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    manifest, tensors = load_tensor_dump(args.tensor_dump)
    required = ["har_source", "har", "har_phase", "x_pre_padded", "ref_s"]
    missing = [name for name in required if name not in tensors]
    if missing:
        raise SystemExit(f"tensor dump missing required tensors: {missing}")

    generator = _load_kmodel().decoder.generator.eval()
    recomputed_har = _manual_stft_har(generator, tensors["har_source"])
    variants = _make_variants(tensors, recomputed_har, args.pad_har_to)
    reference_waveform_key = "waveform_raw_trimmed" if "waveform_raw_trimmed" in tensors else "waveform"
    if reference_waveform_key not in tensors:
        raise SystemExit("tensor dump missing waveform_raw_trimmed or waveform")
    reference_waveform = tensors[reference_waveform_key].reshape(-1)

    with torch.no_grad():
        waveform_metrics = {}
        outputs = {}
        for name, har in variants.items():
            waveform = _run_generator_from_har(generator, tensors, har).detach().cpu().numpy().astype(np.float32)
            outputs[name] = waveform
            waveform_metrics[name] = _metrics(reference_waveform, waveform.reshape(-1))

    dumped_har = tensors["har"].astype(np.float32)
    recomputed_np = recomputed_har.detach().cpu().numpy().astype(np.float32)
    feature_metrics = {
        "recomputed_har_vs_dumped_har": _metrics(dumped_har, recomputed_np),
        "recomputed_nyquist_phase_vs_dumped": _phase_channel_metrics(
            dumped_har[:, NYQUIST_HAR_CHANNEL, :].reshape(-1),
            recomputed_np[:, NYQUIST_HAR_CHANNEL, :].reshape(-1),
        ),
        "swift_basis_nyquist_phase_vs_dumped": _phase_channel_metrics(
            dumped_har[:, NYQUIST_HAR_CHANNEL, :].reshape(-1),
            _swift_basis_nyquist_branch_phase(tensors["har_source"], dumped_har.shape[2]).reshape(-1),
        ),
        "swift_basis_atan2_nyquist_phase_vs_dumped": _phase_channel_metrics(
            dumped_har[:, NYQUIST_HAR_CHANNEL, :].reshape(-1),
            _swift_basis_nyquist_atan2_phase(tensors["har_source"], dumped_har.shape[2]).reshape(-1),
        ),
    }
    for bin_index in range(NYQUIST_BIN):
        feature_metrics[f"recomputed_phase_bin_{bin_index}_vs_dumped"] = _phase_channel_metrics(
            dumped_har[:, 11 + bin_index, :].reshape(-1),
            recomputed_np[:, 11 + bin_index, :].reshape(-1),
        )

    report = {
        "tensor_dump": str(args.tensor_dump),
        "manifest_metadata": manifest.get("metadata", {}),
        "nyquist_bin": NYQUIST_BIN,
        "nyquist_har_channel": NYQUIST_HAR_CHANNEL,
        "pad_har_to": args.pad_har_to,
        "reference_waveform_key": reference_waveform_key,
        "weight_stats": _weight_stats(generator),
        "feature_metrics": feature_metrics,
        "waveform_metrics_vs_dump": waveform_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tensor_dump", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/nyquist_phase_contribution/report.json"))
    parser.add_argument(
        "--pad-har-to",
        type=int,
        default=None,
        help="Pad or trim every natural HAR variant to this time length before running the generator.",
    )
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["waveform_metrics_vs_dump"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
