#!/usr/bin/env python3
"""Summarize the remaining source-contract frontier for Irvine M1.

This report joins three existing evidence streams:

- F0/source formulation probes, which show whether the source equation itself
  can match Swift.
- Irvine next-target analysis, which shows whether quality-fail source/body
  branches would close warmed laishere rows.
- Generator noise/body split reports, which isolate the body-only counterfactual
  when ``x_source_*`` tensors are treated as already available.

It does not promote a candidate. It records why the next useful work is a
cheaper strict source/HAR contract rather than more package-boundary splitting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_VARIANTS_JSON = Path("outputs/f0_source_variants/summary_3s_7s_10s_15s_30s.json")
DEFAULT_IRVINE_TARGETS_JSON = Path("outputs/external_bakeoff/irvine_next_targets.json")
DEFAULT_M2_BODY_REPORT = Path(
    "outputs/generator_noise_split/3s_native_in_broadcast_ios17/report_native_in_ios17_cpu_gpu.json"
)
DEFAULT_IRVINE_BODY_REPORT = Path(
    "outputs/generator_noise_split/3s_native_in_broadcast_ios17/irvine/report_native_in_ios17_cpu_gpu_irvine.json"
)
DEFAULT_OUTPUT = Path("outputs/external_bakeoff/source_contract_frontier.md")
DEFAULT_JSON_OUTPUT = Path("outputs/external_bakeoff/source_contract_frontier.json")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_db(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f} dB"


def _body_counterfactual(report: dict[str, Any], machine_id: str) -> dict[str, Any]:
    med = report["benchmark"]["warm_predict_median_ms"]
    fused_ms = float(med["fused"])
    body_ms = float(med["split_body"])
    source_ms = float(med["split_noise"])
    total_ms = float(med["split_total"])
    return {
        "machine_id": machine_id,
        "fused_ms": fused_ms,
        "body_only_ms": body_ms,
        "source_noise_ms": source_ms,
        "split_total_ms": total_ms,
        "body_only_saves_ms": fused_ms - body_ms,
        "full_split_delta_ms": fused_ms - total_ms,
        "passes": bool(report.get("passes")),
    }


def summarize_source_contract_frontier(
    source_variants: dict[str, Any],
    irvine_targets: dict[str, Any],
    m2_body_report: dict[str, Any],
    irvine_body_report: dict[str, Any],
) -> dict[str, Any]:
    """Return the source-contract frontier summary."""

    rows = source_variants.get("rows") or []
    min_swift_source_snr = source_variants.get("min_swift_like_source_snr_db")
    max_recomputed_har_snr = source_variants.get("max_dump_source_recomputed_har_snr_db")
    quality_profile_closers = [
        row
        for row in (irvine_targets.get("rows") or [])
        if (row.get("quality_fail_vs_warmed_profile") or {}).get(
            "would_beat_warmed_laishere_profile"
        )
    ]
    source_body_rows = [
        row for row in (irvine_targets.get("rows") or []) if "source/body" in str(row.get("target_class"))
    ]
    return {
        "source_equation_is_solved": bool(source_variants.get("source_equation_is_solved")),
        "recomputed_stft_har_is_solved": bool(source_variants.get("recomputed_stft_har_is_solved")),
        "source_variant_rows": len(rows),
        "min_swift_like_source_snr_db": min_swift_source_snr,
        "max_dump_source_recomputed_har_snr_db": max_recomputed_har_snr,
        "irvine_real_loss_count": int(irvine_targets.get("real_loss_count") or 0),
        "irvine_source_body_loss_count": len(source_body_rows),
        "strict_pass_closers": int(irvine_targets.get("strict_pass_closers") or 0),
        "quality_fail_warmed_profile_closers": len(quality_profile_closers),
        "quality_fail_warmed_profile_buckets": [row.get("input_key") for row in quality_profile_closers],
        "body_counterfactuals": [
            _body_counterfactual(m2_body_report, "m2-studio"),
            _body_counterfactual(irvine_body_report, "irvine-m1"),
        ],
        "decision": (
            "The Swift-like source equation is solved, but recomputed HAR/STFT is not. "
            "The body package is fast if x_source tensors are free, and quality-fail "
            "F0/source branches would close several warmed Irvine profile rows. The "
            "next useful work is a cheaper strict source/HAR contract or listening-"
            "accepted source replacement, not another exact HAR-post split."
        ),
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Source Contract Frontier",
        "",
        "Warmed-inference evidence for the remaining Irvine M1 source/body gap.",
        "",
        "## Summary",
        "",
        f"- Source equation solved: `{str(summary['source_equation_is_solved']).lower()}`.",
        f"- Recomputed HAR/STFT solved: `{str(summary['recomputed_stft_har_is_solved']).lower()}`.",
        f"- Source-variant buckets scanned: `{summary['source_variant_rows']}`.",
        f"- Swift-like source minimum SNR: `{_fmt_db(summary['min_swift_like_source_snr_db'])}`.",
        f"- Dumped source recomputed-HAR maximum SNR: `{_fmt_db(summary['max_dump_source_recomputed_har_snr_db'])}`.",
        f"- Irvine real loss rows: `{summary['irvine_real_loss_count']}`.",
        f"- Irvine source/body loss rows: `{summary['irvine_source_body_loss_count']}`.",
        f"- Saved strict candidates closing Irvine rows: `{summary['strict_pass_closers']}`.",
        f"- Quality-fail source candidates that beat warmed laishere profile: `{summary['quality_fail_warmed_profile_closers']}`.",
        "",
        "## Body Counterfactual",
        "",
        "| Machine | Fused | Body only | Source/noise | Full split | Body-only save | Full split delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["body_counterfactuals"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["machine_id"],
                    _fmt_ms(row["fused_ms"]),
                    _fmt_ms(row["body_only_ms"]),
                    _fmt_ms(row["source_noise_ms"]),
                    _fmt_ms(row["split_total_ms"]),
                    _fmt_ms(row["body_only_saves_ms"]),
                    _fmt_ms(row["full_split_delta_ms"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Quality-Fail Closers",
            "",
            "Quality-fail buckets that would beat the warmed laishere profile if accepted:",
            (
                "`" + "`, `".join(str(v) for v in summary["quality_fail_warmed_profile_buckets"]) + "`."
                if summary["quality_fail_warmed_profile_buckets"]
                else "None."
            ),
            "",
            "## Decision",
            "",
            summary["decision"],
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-variants-json", type=Path, default=DEFAULT_SOURCE_VARIANTS_JSON)
    parser.add_argument("--irvine-targets-json", type=Path, default=DEFAULT_IRVINE_TARGETS_JSON)
    parser.add_argument("--m2-body-report", type=Path, default=DEFAULT_M2_BODY_REPORT)
    parser.add_argument("--irvine-body-report", type=Path, default=DEFAULT_IRVINE_BODY_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = summarize_source_contract_frontier(
        _load_json(args.source_variants_json),
        _load_json(args.irvine_targets_json),
        _load_json(args.m2_body_report),
        _load_json(args.irvine_body_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "source_equation_is_solved": summary["source_equation_is_solved"],
                "recomputed_stft_har_is_solved": summary["recomputed_stft_har_is_solved"],
                "quality_fail_warmed_profile_closers": summary[
                    "quality_fail_warmed_profile_closers"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
