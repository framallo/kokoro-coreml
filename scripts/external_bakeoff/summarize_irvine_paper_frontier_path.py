#!/usr/bin/env python3
"""Summarize the Irvine M1 path to the paper-facing frontier.

This joins the listening-gated source/body candidates with the measured
HAR-post rewrite projection. The purpose is to avoid a false conclusion from
profile-only wins: absolute-fastest claims must beat the stricter
``competitive_frontier`` paper rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_LOWER_END_GATE_JSON = DEFAULT_OUTPUT_DIR / "lower_end_mac_win_gate.json"
DEFAULT_REWRITE_IMPACT_JSON = DEFAULT_OUTPUT_DIR / "rewrite_candidate_impact.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "irvine_paper_frontier_path.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "irvine_paper_frontier_path.json"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _rewrite_save_by_bucket(rewrite_impact: dict[str, Any]) -> dict[str, float]:
    saves: dict[str, float] = {}
    for row in rewrite_impact.get("projection_rows") or []:
        if row.get("machine_id") == "irvine-m1":
            saves[str(row["bucket"])] = float(row["projected_save_ms"])
    return saves


def build_summary(lower_end_gate: dict[str, Any], rewrite_impact: dict[str, Any]) -> dict[str, Any]:
    """Build the Irvine paper-frontier combined-path summary."""

    rewrite_saves = _rewrite_save_by_bucket(rewrite_impact)
    rows: list[dict[str, Any]] = []
    for row in lower_end_gate.get("irvine_rows") or []:
        bucket = str(row["input_key"])
        source_projected_ms = float(row["projected_full_ms"])
        rewrite_save_ms = rewrite_saves.get(bucket, 0.0)
        combined_projected_ms = source_projected_ms - rewrite_save_ms
        paper_ms = float(row["paper_competitor_ms"])
        combined_margin_ms = paper_ms - combined_projected_ms
        rows.append(
            {
                "bucket": bucket,
                "candidate": row["candidate"],
                "human_decision": row.get("human_decision") or "",
                "paper_competitor": row.get("paper_competitor"),
                "paper_competitor_ms": paper_ms,
                "source_projected_ms": source_projected_ms,
                "source_paper_margin_ms": float(row["paper_margin_ms"]),
                "rewrite_save_ms": rewrite_save_ms,
                "combined_projected_ms": combined_projected_ms,
                "combined_paper_margin_ms": combined_margin_ms,
                "would_beat_paper_with_rewrite": combined_margin_ms > 0,
                "additional_save_required_ms": max(0.0, -combined_margin_ms),
                "corr": row.get("corr"),
                "snr_db": row.get("snr_db"),
            }
        )
    rows.sort(key=lambda item: (item["additional_save_required_ms"], item["bucket"]))
    return {
        "rows": rows,
        "paper_rows_closed_by_source_plus_rewrite": sum(
            1 for row in rows if row["would_beat_paper_with_rewrite"]
        ),
        "paper_rows_remaining_after_source_plus_rewrite": sum(
            1 for row in rows if not row["would_beat_paper_with_rewrite"]
        ),
        "decision": (
            "On Irvine M1, source/body plus the HAR-post rewrite is enough for "
            "the 10s paper row if listening accepts the source candidate. The "
            "15s row is close, while 3s and 7s still need larger implementation "
            "wins than the current saved probes provide."
        ),
    }


def _fmt_ms(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f} ms"


def _fmt_num(value: Any, digits: int = 3) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def render_markdown(summary: dict[str, Any]) -> str:
    """Render the combined Irvine paper-frontier path as Markdown."""

    lines = [
        "# Irvine Paper Frontier Path",
        "",
        "Warmed inference only. This report combines the best saved Irvine",
        "source/body candidate per bucket with the measured HAR-post rewrite",
        "projection. It uses the stricter paper-facing `competitive_frontier`",
        "rows, not only newer stage-profile rows.",
        "",
        f"Paper rows closed by source+rewrite: `{summary['paper_rows_closed_by_source_plus_rewrite']}`.",
        f"Paper rows still open after source+rewrite: `{summary['paper_rows_remaining_after_source_plus_rewrite']}`.",
        "",
        "| Bucket | Candidate | Paper frontier | Source projected | Rewrite save | Combined projected | Combined margin | Extra save needed | Quality |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in summary["rows"]:
        margin = row["combined_paper_margin_ms"]
        margin_text = _fmt_ms(margin)
        if margin > 0:
            margin_text = f"+{margin_text}"
        quality = f"corr {_fmt_num(row.get('corr'), 3)}, SNR {_fmt_num(row.get('snr_db'), 2)} dB"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    f"`{row['candidate']}`",
                    f"{row['paper_competitor']}: {_fmt_ms(row['paper_competitor_ms'])}",
                    _fmt_ms(row["source_projected_ms"]),
                    _fmt_ms(row["rewrite_save_ms"]),
                    _fmt_ms(row["combined_projected_ms"]),
                    margin_text,
                    _fmt_ms(row["additional_save_required_ms"]),
                    quality,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            summary["decision"],
            "",
            "## Immediate Work",
            "",
            "- Promote the HAR-post rewrite into any publishable Irvine source/body rerun.",
            "- If listening accepts `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt`, rerun Irvine `10s` on a quiet host because source+rewrite should beat the paper row.",
            "- Find at least another `2.7 ms` for Irvine `15s` after source+rewrite, or prove a quieter rerun changes that margin.",
            "- Treat Irvine `3s` and `7s` as unsolved implementation work; current source+rewrite projections remain far short of the paper frontier.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lower-end-gate-json", type=Path, default=DEFAULT_LOWER_END_GATE_JSON)
    parser.add_argument("--rewrite-impact-json", type=Path, default=DEFAULT_REWRITE_IMPACT_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    summary = build_summary(
        _load_json(args.lower_end_gate_json),
        _load_json(args.rewrite_impact_json),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(summary))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paper_rows_closed_by_source_plus_rewrite": summary[
                    "paper_rows_closed_by_source_plus_rewrite"
                ],
                "paper_rows_remaining_after_source_plus_rewrite": summary[
                    "paper_rows_remaining_after_source_plus_rewrite"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
