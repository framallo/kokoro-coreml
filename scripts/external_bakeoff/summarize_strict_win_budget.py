#!/usr/bin/env python3
"""Summarize the remaining strict-win budget after the rewrite candidate.

The output is intentionally conservative: it uses only strict, warmed evidence
and treats the HAR-post upsample rewrite as a projected generator-stage gain
until quiet Irvine timing proves it. The budget tells future experiments how
much additional strict speed is needed after that candidate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_REWRITE_IMPACT = DEFAULT_OUTPUT_DIR / "rewrite_candidate_impact.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "strict_win_budget_after_rewrite.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "strict_win_budget_after_rewrite.json"


def _load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def build_budget(rewrite_impact: dict[str, Any]) -> dict[str, Any]:
    """Build the remaining strict-win budget from rewrite-impact rows."""

    rows: list[dict[str, Any]] = []
    for row in rewrite_impact.get("projection_rows") or []:
        if row.get("machine_id") != "irvine-m1":
            continue
        generator_after_ms = float(row["config_generator_ms"]) - float(row["projected_save_ms"])
        profile_required_ms = max(0.0, float(row["projected_gap_ms"]))
        frontier_required_ms = max(0.0, float(row["projected_frontier_gap_ms"]))
        rows.append(
            {
                "bucket": str(row["bucket"]),
                "projected_total_after_rewrite_ms": float(row["projected_total_ms"]),
                "laishere_profile_ms": float(row["laishere_total_ms"]),
                "paper_frontier_best_ms": float(row["frontier_best_ms"]),
                "generator_after_rewrite_ms": generator_after_ms,
                "additional_profile_save_required_ms": profile_required_ms,
                "additional_profile_generator_speedup_required_pct": (
                    100.0 * profile_required_ms / generator_after_ms if generator_after_ms > 0 else None
                ),
                "additional_paper_save_required_ms": frontier_required_ms,
                "additional_paper_generator_speedup_required_pct": (
                    100.0 * frontier_required_ms / generator_after_ms if generator_after_ms > 0 else None
                ),
            }
        )
    rows.sort(key=lambda item: int(item["bucket"].rstrip("s")))
    return {
        "rewrite_impact": str(DEFAULT_REWRITE_IMPACT),
        "rows": rows,
        "summary": {
            "irvine_buckets": len(rows),
            "profile_rows_already_closed": sum(
                1 for row in rows if row["additional_profile_save_required_ms"] <= 0
            ),
            "profile_rows_remaining": sum(
                1 for row in rows if row["additional_profile_save_required_ms"] > 0
            ),
            "paper_rows_remaining": sum(1 for row in rows if row["additional_paper_save_required_ms"] > 0),
        },
    }


def _fmt_ms(value: float) -> str:
    """Format milliseconds."""

    return f"{value:.1f} ms"


def _fmt_pct(value: float | None) -> str:
    """Format a percent or ``n/a``."""

    return "n/a" if value is None else f"{value:.2f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the budget as Markdown."""

    lines = [
        "# Strict Win Budget After Rewrite",
        "",
        "This table starts from the measured HAR-post upsample rewrite candidate and",
        "asks what additional strict speed is still required on Irvine M1. It uses",
        "warmed profile medians only. The rewrite itself is still a projection for",
        "Irvine until the host is quiet enough for publishable timing.",
        "",
        "## Profile Target",
        "",
        "Profile target means beating the newer warmed laishere stage-profile row.",
        "",
        "| Bucket | Projected Config F | laishere profile | Extra save needed | Extra generator speedup needed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["projected_total_after_rewrite_ms"]),
                    _fmt_ms(row["laishere_profile_ms"]),
                    _fmt_ms(row["additional_profile_save_required_ms"]),
                    _fmt_pct(row["additional_profile_generator_speedup_required_pct"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Paper Frontier Target",
            "",
            "Paper frontier target means beating the current strict paper-facing row,",
            "which may be stricter than the newer warmed profile row.",
            "",
            "| Bucket | Paper frontier best | Extra save needed | Extra generator speedup needed |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['bucket']}`",
                    _fmt_ms(row["paper_frontier_best_ms"]),
                    _fmt_ms(row["additional_paper_save_required_ms"]),
                    _fmt_pct(row["additional_paper_generator_speedup_required_pct"]),
                ]
            )
            + " |"
        )
    s = payload["summary"]
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Irvine profile rows remaining after rewrite projection: `{s['profile_rows_remaining']}`.",
            f"- Irvine paper rows remaining after rewrite projection: `{s['paper_rows_remaining']}`.",
            "- The next strict candidate must be much larger than another 1-3% local",
            "  generator tweak unless it targets only the nearly closed `15s` row.",
            "- For `3s/7s/10s`, the remaining profile target needs roughly",
            "  `4-20%` additional generator-stage improvement after the rewrite.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rewrite-impact", type=Path, default=DEFAULT_REWRITE_IMPACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    payload = build_budget(_load_json(args.rewrite_impact))
    payload["rewrite_impact"] = str(args.rewrite_impact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload) + "\n")
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_rows_remaining": payload["summary"]["profile_rows_remaining"],
                "paper_rows_remaining": payload["summary"]["paper_rows_remaining"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
