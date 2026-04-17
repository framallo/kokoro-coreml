#!/usr/bin/env python3
"""Probe exact-length enumerated Core ML Duration export.

This is an experiment, not the production exporter. The current production
Duration export uses mask-aware manually unrolled BiLSTMs so padded token buckets
match exact PyTorch semantics. This probe tests the alternative architecture:
never pad inside Duration, pass exact token lengths, and let Core ML specialize a
native recurrent graph with ``ct.EnumeratedShapes``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kokoro._export_utils import load_kokoro_for_export

ROOT = Path(__file__).resolve().parents[1]

_, kokoro_modules, kokoro_model = load_kokoro_for_export(
    ROOT, suffix="_duration_exact_enum"
)
KModel = kokoro_model.KModel
AdaLayerNorm = kokoro_modules.AdaLayerNorm


class ExactNativeTextEncoder(nn.Module):
    """Text encoder variant that runs native BiLSTM over exact, unpadded tokens."""

    def __init__(self, original_encoder: nn.Module) -> None:
        super().__init__()
        self.embedding = original_encoder.embedding
        self.cnn = original_encoder.cnn
        self.lstm = original_encoder.lstm

    def forward(
        self, x: torch.Tensor, input_lengths: torch.Tensor, m: torch.Tensor
    ) -> torch.Tensor:
        x = self.embedding(x)
        x = x.transpose(1, 2)
        m = m.unsqueeze(1)
        x.masked_fill_(m, 0.0)
        for c in self.cnn:
            x = c(x)
            x.masked_fill_(m, 0.0)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(-1, -2)
        x.masked_fill_(m, 0.0)
        return x


class ExactNativeDurationEncoder(nn.Module):
    """Duration encoder variant that runs native BiLSTMs over exact tokens."""

    def __init__(self, original_encoder: nn.Module) -> None:
        super().__init__()
        self.lstms = original_encoder.lstms
        self.dropout = original_encoder.dropout

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        text_lengths: torch.Tensor,
        m: torch.Tensor,
    ) -> torch.Tensor:
        masks = m
        x = x.permute(2, 0, 1)
        batch_size = x.shape[1]
        seq_len = x.shape[0]
        style_dim = style.shape[-1]
        s = style.unsqueeze(0).repeat(seq_len, batch_size, 1)
        x = torch.cat([x, s], axis=-1)
        x.masked_fill_(masks.unsqueeze(-1).transpose(0, 1), 0.0)
        x = x.transpose(0, 1)
        x = x.transpose(-1, -2)
        for block in self.lstms:
            if isinstance(block, AdaLayerNorm) or type(block).__name__ == "AdaLayerNorm":
                x = block(x.transpose(-1, -2), style).transpose(-1, -2)
                x = torch.cat([x, s.permute(1, 2, 0)], axis=1)
                x.masked_fill_(masks.unsqueeze(-1).transpose(-1, -2), 0.0)
            else:
                x = x.transpose(-1, -2)
                x, _ = block(x)
                x = F.dropout(x, p=self.dropout, training=False)
                x = x.transpose(-1, -2)
        return x.transpose(-1, -2)


class ExactEnumeratedDurationModel(nn.Module):
    """Duration wrapper for exact token lengths, with no padding mask input."""

    def __init__(self, kmodel: KModel) -> None:
        super().__init__()
        self.kmodel = kmodel
        self.kmodel.text_encoder = ExactNativeTextEncoder(kmodel.text_encoder)
        self.kmodel.predictor.text_encoder = ExactNativeDurationEncoder(
            kmodel.predictor.text_encoder
        )
        if hasattr(self.kmodel.bert.embeddings, "token_type_ids"):
            delattr(self.kmodel.bert.embeddings, "token_type_ids")

    def forward(
        self,
        input_ids: torch.Tensor,
        ref_s: torch.Tensor,
        speed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        k = self.kmodel
        attention_mask = torch.ones_like(input_ids)
        input_lengths = input_ids.new_full((input_ids.shape[0],), input_ids.shape[1])
        input_lengths = input_lengths.to(torch.long)
        text_mask = attention_mask == 0
        token_type_ids = torch.zeros_like(input_ids)

        bert_dur = k.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        d_en = k.bert_encoder(bert_dur).transpose(-1, -2)
        s = ref_s[:, 128:]
        d = k.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = k.predictor.lstm(d)
        duration = k.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1) / speed
        pred_dur = torch.round(duration).clamp(min=1).long()
        t_en = k.text_encoder(input_ids, input_lengths, text_mask)
        ref_s_out = ref_s + torch.zeros_like(ref_s)
        return pred_dur, d, t_en, s, ref_s_out


def _path_is_readable_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def load_kmodel() -> KModel:
    cfg = ROOT / "checkpoints/config.json"
    ckpt = ROOT / "checkpoints/kokoro-v1_0.pth"
    if _path_is_readable_file(cfg) and _path_is_readable_file(ckpt):
        return KModel(config=str(cfg), model=str(ckpt), disable_complex=True)
    if _path_is_readable_file(cfg):
        return KModel(config=str(cfg), disable_complex=True)
    return KModel(disable_complex=True)


def remove_training_ops(model: nn.Module) -> None:
    """Replace training-only modules so tracing stays in inference dialect."""

    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, nn.Identity())
        elif isinstance(module, nn.BatchNorm1d):
            module.eval()
            module.track_running_stats = False
        elif isinstance(module, nn.LSTM):
            module.eval()


def load_prepared_input(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    num_tokens = int(data["num_tokens"])
    input_ids = np.asarray(data["input_ids"][:num_tokens], dtype=np.int32).reshape(
        1, num_tokens
    )
    ref_s = np.asarray(data["ref_s"], dtype=np.float32).reshape(1, 256)
    speed = np.asarray([float(data.get("speed", 1.0))], dtype=np.float32)
    canonical_duration_s = data.get("canonical_duration_s")
    expected_frames = (
        int(round(float(canonical_duration_s) * 40.0))
        if canonical_duration_s is not None
        else None
    )
    return {
        "path": str(path),
        "key": path.stem,
        "num_tokens": num_tokens,
        "input_ids": input_ids,
        "ref_s": ref_s,
        "speed": speed,
        "canonical_duration_s": canonical_duration_s,
        "expected_frames": expected_frames,
    }


def tensor_from_np(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array)


def duration_frame_sum(output: Any) -> int:
    return int(np.asarray(output).reshape(-1).sum())


def collect_op_counts(mlmodel: ct.models.MLModel) -> dict[str, int]:
    """Return MIL op counts for a converted ML Program when available."""

    spec = mlmodel.get_spec()
    if spec.WhichOneof("Type") != "mlProgram":
        return {}

    counts: Counter[str] = Counter()

    def visit_block(block: Any) -> None:
        for op in block.operations:
            counts[op.type] += 1
            for child in getattr(op, "blocks", []):
                visit_block(child)

    for function in spec.mlProgram.functions.values():
        for block in function.block_specializations.values():
            visit_block(block)
    return dict(sorted(counts.items()))


def export_exact_enumerated_model(
    model: nn.Module,
    lengths: list[int],
    output_path: Path,
) -> ct.models.MLModel:
    default_t = max(lengths)
    input_ids = torch.zeros((1, default_t), dtype=torch.int32)
    ref_s = torch.zeros((1, 256), dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(
            model, (input_ids, ref_s, speed), strict=False, check_trace=False
        )

    input_shape = ct.EnumeratedShapes(
        shapes=[[1, t] for t in lengths],
        default=[1, default_t],
    )
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=input_shape, dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="pred_dur"),
            ct.TensorType(name="d"),
            ct.TensorType(name="t_en"),
            ct.TensorType(name="s"),
            ct.TensorType(name="ref_s_out"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return mlmodel


def export_exact_fixed_model(
    model: nn.Module,
    length: int,
    output_path: Path,
) -> ct.models.MLModel:
    input_ids = torch.zeros((1, length), dtype=torch.int32)
    ref_s = torch.zeros((1, 256), dtype=torch.float32)
    speed = torch.tensor([1.0], dtype=torch.float32)

    with torch.no_grad():
        traced = torch.jit.trace(
            model, (input_ids, ref_s, speed), strict=False, check_trace=False
        )

    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="input_ids", shape=(1, length), dtype=np.int32),
            ct.TensorType(name="ref_s", shape=(1, 256), dtype=np.float32),
            ct.TensorType(name="speed", shape=(1,), dtype=np.float32),
        ],
        outputs=[
            ct.TensorType(name="pred_dur"),
            ct.TensorType(name="d"),
            ct.TensorType(name="t_en"),
            ct.TensorType(name="s"),
            ct.TensorType(name="ref_s_out"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS12,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
    )
    if output_path.exists():
        import shutil

        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(output_path))
    return mlmodel


def predict_rows(
    mlmodel: ct.models.MLModel,
    prepared_inputs: list[dict[str, Any]],
    iterations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in prepared_inputs:
        model_input = {
            "input_ids": item["input_ids"],
            "ref_s": item["ref_s"],
            "speed": item["speed"],
        }
        latencies_ms: list[float] = []
        pred_frames = None
        for iteration in range(iterations + 1):
            t0 = time.perf_counter()
            out = mlmodel.predict(model_input)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if iteration > 0:
                latencies_ms.append(elapsed_ms)
            pred_frames = duration_frame_sum(out["pred_dur"])
        rows.append(
            {
                "key": item["key"],
                "tokens": item["num_tokens"],
                "expected_frames": item["expected_frames"],
                "coreml_frames": pred_frames,
                "coreml_delta_vs_expected": (
                    None
                    if item["expected_frames"] is None or pred_frames is None
                    else pred_frames - item["expected_frames"]
                ),
                "predict_ms_median": statistics.median(latencies_ms),
                "predict_ms_min": min(latencies_ms),
                "predict_ms_max": max(latencies_ms),
                "iterations": iterations,
            }
        )
    return rows


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    prepared_inputs = [load_prepared_input(Path(path)) for path in args.inputs]
    lengths = [item["num_tokens"] for item in prepared_inputs]

    kmodel = load_kmodel()
    model = ExactEnumeratedDurationModel(kmodel).eval()
    remove_training_ops(model)
    for module in model.modules():
        module.eval()

    torch_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in prepared_inputs:
            pred_dur, *_ = model(
                tensor_from_np(item["input_ids"]),
                tensor_from_np(item["ref_s"]),
                tensor_from_np(item["speed"]),
            )
            frames = duration_frame_sum(pred_dur.detach().cpu().numpy())
            torch_rows.append(
                {
                    "key": item["key"],
                    "tokens": item["num_tokens"],
                    "expected_frames": item["expected_frames"],
                    "torch_exact_native_frames": frames,
                    "torch_delta_vs_expected": (
                        None
                        if item["expected_frames"] is None
                        else frames - item["expected_frames"]
                    ),
                }
            )

    mlmodel = export_exact_enumerated_model(model, lengths, Path(args.output))
    op_counts = collect_op_counts(mlmodel)

    coreml_rows = predict_rows(mlmodel, prepared_inputs, args.iterations)

    fixed_rows: list[dict[str, Any]] = []
    fixed_op_counts: dict[str, dict[str, int]] = {}
    if args.also_fixed:
        fixed_dir = Path(args.fixed_output_dir)
        for item in prepared_inputs:
            length = item["num_tokens"]
            fixed_output = fixed_dir / f"kokoro_duration_exact_t{length}.mlpackage"
            fixed_model = export_exact_fixed_model(model, length, fixed_output)
            fixed_op_counts[str(length)] = collect_op_counts(fixed_model)
            row = predict_rows(fixed_model, [item], args.iterations)[0]
            row["output"] = str(fixed_output)
            fixed_rows.append(row)

    report = {
        "output": str(Path(args.output)),
        "fixed_output_dir": str(Path(args.fixed_output_dir)),
        "lengths": lengths,
        "coremltools": ct.__version__,
        "torch": torch.__version__,
        "torch_rows": torch_rows,
        "coreml_rows": coreml_rows,
        "op_counts": op_counts,
        "fixed_rows": fixed_rows,
        "fixed_op_counts": fixed_op_counts,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "outputs/swift_bench_inputs/3s.json",
            "outputs/swift_bench_inputs/7s.json",
            "outputs/swift_bench_inputs/15s.json",
            "outputs/swift_bench_inputs/30s.json",
        ],
        help="Prepared Swift bench input JSON files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/duration_exact_enum/kokoro_duration_exact_enum.mlpackage",
        help="Experimental Core ML package path.",
    )
    parser.add_argument(
        "--report",
        default="outputs/duration_exact_enum/report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Warm predict once, then time this many predictions per length.",
    )
    parser.add_argument(
        "--also-fixed",
        action="store_true",
        help="Also export one exact fixed-shape package per input length.",
    )
    parser.add_argument(
        "--fixed-output-dir",
        default="outputs/duration_exact_enum/fixed",
        help="Directory for optional exact fixed-shape packages.",
    )
    return parser.parse_args()


def main() -> None:
    report = run_probe(parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
