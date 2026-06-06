#!/usr/bin/env python3
"""Summarize known Config F speed candidates and next useful gates.

This ledger is intentionally evidence-driven. It records both successes and
rejections so future optimization passes do not repeat measured dead ends while
the remaining lower-end Mac and iPhone gates are waiting on external state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_frontier_matrix.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "candidate_frontier_matrix.json"
DEFAULT_STRICT_BUDGET = DEFAULT_OUTPUT_DIR / "strict_win_budget_after_rewrite.json"
DEFAULT_IOS_INSTALL = DEFAULT_OUTPUT_DIR / "config_f_ios_manual_install_latest.json"


@dataclass(frozen=True)
class Candidate:
    """One measured optimization family."""

    family: str
    scope: str
    best_signal: str
    quality: str
    strict: bool
    production_ready: bool
    decision: str
    evidence: str
    next_gate: str


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        family="HAR-post upsample ConvT rewrite",
        scope="single-package GeneratorFromHar",
        best_signal="M2 Studio package +4.28% 3s, +3.15% 7s, +3.17% 10s, +2.60% 15s, +2.20% 30s; local E2E +1.22-2.58%",
        quality="strict-like spotchecks: corr >=0.999994895, SNR >=46.51 dB, max abs <=0.008453",
        strict=True,
        production_ready=True,
        decision="keep; promote to quiet Irvine timing before replacing checked-in packages",
        evidence="outputs/external_bakeoff/rewrite_candidate_impact.md",
        next_gate="quiet Irvine M1 warmed run; quiet M2 Air rerun for fresh lower-end rows",
    ),
    Candidate(
        family="Full visible surface rewrite",
        scope="single-package GeneratorFromHar",
        best_signal="local E2E only wins 3s (49.532 ms vs production rewrite 49.669 ms) and noise-ties 10s",
        quality="strict-like spotchecks: corr >=0.999993675, SNR >=48.01 dB, max abs <=0.007263",
        strict=True,
        production_ready=False,
        decision="reject as production replacement; simpler rewrite wins more buckets",
        evidence="outputs/external_bakeoff/results_config_f_reference_m2-studio-local_full_surface_ups_as_conv.json",
        next_gate="none unless a new operator rewrite changes runtime behavior beyond surface matching",
    ),
    Candidate(
        family="Native-IN/broadcast/cos/fp16 fused surface without upsample rewrite",
        scope="single-package GeneratorFromHar",
        best_signal="local 3s roughly +0.08-0.26%; .all/CPU+NE remains harmful",
        quality="strict",
        strict=True,
        production_ready=False,
        decision="reject as too small; graph-surface parity alone is not enough",
        evidence="README/Notes/performance-notes.md",
        next_gate="do not repeat without a new layout, fusion, or partitioning mechanism",
    ),
    Candidate(
        family="Style-specialized fused generator",
        scope="single-package GeneratorFromHar with fixed af_heart projections",
        best_signal="Irvine 3s -3.0 ms; M2 Air 3s -2.2 ms; local native-IN variant only +0.07 ms",
        quality="strict on CPU+GPU; CPU+NE fails quality and is far slower",
        strict=True,
        production_ready=False,
        decision="reject; freezing style is not a speed path",
        evidence="README/Notes/performance-notes.md",
        next_gate="none unless combined with a material new operator rewrite",
    ),
    Candidate(
        family="Exact decoder+vocoder split",
        scope="multi-package exact Swift HAR contract",
        best_signal="Irvine 3s CPU+GPU -24.8 ms; CPU+NE -138.3 ms",
        quality="strict versus fused path",
        strict=True,
        production_ready=False,
        decision="reject; split boundary/sync overhead exceeds body savings",
        evidence="README/Guides/apple-silicon/Kokoro-M1-vocoder-boundary-research-brief.md",
        next_gate="do not repeat broad exact split; only try if a Core ML call boundary is removed",
    ),
    Candidate(
        family="Exact generator noise/body split",
        scope="multi-package exact HAR-post generator split",
        best_signal="Irvine 3s CPU+GPU -11.5 ms; CPU+NE quality/speed failure",
        quality="strict on CPU+GPU; CPU+NE quality fails",
        strict=True,
        production_ready=False,
        decision="reject; tightest strict split still loses",
        evidence="README/Notes/performance-notes.md",
        next_gate="do not repeat unless packaging/synchronization changes",
    ),
    Candidate(
        family="HAR-source fused strict path",
        scope="source/STFT/HAR fused path",
        best_signal="Irvine 3s CPU+GPU -22.9 ms; CPU+NE -163.8 ms",
        quality="strict with padded geometry and dumped Nyquist phase",
        strict=True,
        production_ready=False,
        decision="reject; preserving strict source contract loses the speed edge",
        evidence="outputs/nyquist_phase_contribution/summary.md",
        next_gate="new representation only; do not promote padded strict path",
    ),
    Candidate(
        family="Fast F0/source simplification",
        scope="laishere-like source/body branch",
        best_signal="Irvine 3s +10.9 to +18.7 ms depending branch",
        quality="quality-fail: corr 0.813995-0.931840, SNR 5.08-9.19 dB",
        strict=False,
        production_ready=False,
        decision="not paper-strict; only useful with source recovery or no-ASR listening acceptance",
        evidence="outputs/f0_source_listening/cos_resblock_speed_branch/README.md",
        next_gate="human listening decisions or source/STFT representation repair",
    ),
)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _status_order(candidate: Candidate) -> tuple[int, str]:
    if candidate.production_ready:
        return (0, candidate.family)
    if candidate.strict:
        return (1, candidate.family)
    return (2, candidate.family)


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build the matrix payload."""

    candidates = sorted(CANDIDATES, key=_status_order)
    budget = _load_optional_json(args.strict_budget)
    ios_install = _load_optional_json(args.ios_install)
    profile_rows_remaining = (
        budget.get("summary", {}).get("profile_rows_remaining")
        if isinstance(budget.get("summary"), dict)
        else None
    )
    ios_launch_blocker = ios_install.get("launch_blocker") or "unknown"
    return {
        "strict_budget": str(args.strict_budget),
        "ios_install": str(args.ios_install),
        "summary": {
            "candidate_count": len(candidates),
            "production_ready_strict_candidates": sum(1 for item in candidates if item.production_ready and item.strict),
            "strict_rejected_or_too_small": sum(1 for item in candidates if item.strict and not item.production_ready),
            "quality_fail_speed_candidates": sum(1 for item in candidates if not item.strict),
            "profile_rows_remaining_after_rewrite": profile_rows_remaining,
            "iphone_launch_blocker": ios_launch_blocker,
        },
        "candidates": [asdict(item) for item in candidates],
        "next_actions": [
            "Run scripts/external_bakeoff/check_remote_host_quiet.py before any lower-end Mac promotion run.",
            "Retest the HAR-post upsample rewrite on Irvine M1 and M2 Air only when outputs/external_bakeoff/remote_host_quiet_latest.md reports quiet=yes.",
            "Do not use cold compile/cache timings; every frontier update must use warmed medians.",
            "For a new strict candidate, require a single-package graph or a removed Core ML call boundary before lower-end promotion.",
            "Run the installed Config F iPhone runner only after the physical iPhone is unlocked; current launch blocker is device_locked.",
            "Keep fast F0/source branches separate from strict paper claims unless no-ASR human listening accepts the exact WAVs.",
        ],
    }


def _yes_no(value: bool) -> str:
    return "`yes`" if value else "`no`"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render the matrix as Markdown."""

    summary = payload["summary"]
    lines = [
        "# Candidate Frontier Matrix",
        "",
        "This matrix records the current measured optimization frontier for Config F.",
        "It uses warmed-inference evidence only and separates strict production",
        "candidates from quality-changing speed branches.",
        "",
        "## Summary",
        "",
        f"- Candidates recorded: `{summary['candidate_count']}`.",
        f"- Production-ready strict candidates: `{summary['production_ready_strict_candidates']}`.",
        f"- Strict rejected or too-small candidates: `{summary['strict_rejected_or_too_small']}`.",
        f"- Quality-fail speed candidates: `{summary['quality_fail_speed_candidates']}`.",
        f"- Irvine profile rows remaining after rewrite projection: `{summary['profile_rows_remaining_after_rewrite']}`.",
        f"- iPhone Config F launch blocker: `{summary['iphone_launch_blocker']}`.",
        "",
        "## Matrix",
        "",
        "| Family | Scope | Best signal | Strict | Production-ready | Decision | Next gate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["candidates"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["family"],
                    row["scope"],
                    row["best_signal"],
                    _yes_no(bool(row["strict"])),
                    _yes_no(bool(row["production_ready"])),
                    row["decision"],
                    row["next_gate"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## Evidence Links", ""])
    for row in payload["candidates"]:
        lines.append(f"- {row['family']}: `{row['evidence']}`.")
    lines.extend(["", "## Next Actions", ""])
    for action in payload["next_actions"]:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-budget", type=Path, default=DEFAULT_STRICT_BUDGET)
    parser.add_argument("--ios-install", type=Path, default=DEFAULT_IOS_INSTALL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(payload))
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "candidate_count": payload["summary"]["candidate_count"],
                "iphone_launch_blocker": payload["summary"]["iphone_launch_blocker"],
                "output": str(args.output),
                "profile_rows_remaining_after_rewrite": payload["summary"][
                    "profile_rows_remaining_after_rewrite"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
