#!/usr/bin/env python3
"""Bakeoff summarize mode -- read results and emit tables plus gate answers.

Separated from bakeoff_harness.py per the LOC guard (harness must stay <= 800).
This module has zero coupling to benchmark contexts or model loading.

Usage (standalone)::

    python scripts/bakeoff_summarize.py --results outputs/bakeoff/results_m2_ultra.json

Usage (via harness)::

    python scripts/bakeoff_harness.py summarize --results outputs/bakeoff/results_m2_ultra.json
"""
from __future__ import annotations

import argparse
import glob as globmod
import json
import statistics
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BAKEOFF_DIR = _REPO_ROOT / "outputs" / "bakeoff"

CONFIG_LABELS = {
    "a": "Config A (HAR-post)",
    "b": "Config B (decoder-only .all)",
    "c": "Config C (decoder-only .cpuAndGPU)",
    "d": "Config D (PyTorch MPS)",
    "e": "Config E (PyTorch CPU)",
    "f": "Config F (Swift + CoreML .all)",
    "g": "Config G (Swift + CoreML .cpuAndGPU)",
    "bcpu": "Config Bcpu (decoder-only .cpuOnly)",
}


def _stats(values: list[float]) -> dict[str, float]:
    """Compute summary statistics for a list of values."""
    if not values:
        return {"mean": None, "median": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": round(statistics.mean(values), 6),
        "median": round(statistics.median(values), 6),
        "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "n": len(values),
    }


def _parse_telemetry(path: str) -> dict[str, Any]:
    """Parse a powermetrics ANE output file for median power readings."""
    p = Path(path)
    if not p.exists():
        return {"path": path, "available": False}
    lines = p.read_text().splitlines()
    ane_mw = []
    for line in lines:
        # powermetrics --samplers ane outputs lines like "ANE Power: 123 mW"
        if "ane power" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                try:
                    val = float(parts[1].strip().split()[0])
                    ane_mw.append(val)
                except (ValueError, IndexError):
                    continue
    if not ane_mw:
        return {"path": path, "available": True, "readings": 0}
    # Trim first and last 5s worth of samples (at 100ms interval = 50 samples each).
    trim = min(50, len(ane_mw) // 4)
    steady = ane_mw[trim:-trim] if trim > 0 and len(ane_mw) > 2 * trim else ane_mw
    return {
        "path": path,
        "available": True,
        "readings": len(ane_mw),
        "steady_readings": len(steady),
        "median_mw": round(statistics.median(steady), 2),
        "mean_mw": round(statistics.mean(steady), 2),
    }


def _format_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    """Format a markdown table."""
    if align is None:
        align = ["l"] * len(headers)
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))
    sep_parts = []
    for i, a in enumerate(align):
        if a == "r":
            sep_parts.append("-" * max(1, col_widths[i] - 1) + ":")
        else:
            sep_parts.append("-" * col_widths[i])
    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join(sep_parts) + " |"
    body_lines = []
    for row in rows:
        cells = []
        for i, cell in enumerate(row):
            if align[i] == "r":
                cells.append(cell.rjust(col_widths[i]))
            else:
                cells.append(cell.ljust(col_widths[i]))
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, sep_line] + body_lines)


def _ordered_input_keys(data: dict) -> list[str]:
    """Return manifest input order, falling back to result order for legacy files."""
    keys: list[str] = []
    for key in data.get("inputs", {}):
        if key not in keys:
            keys.append(key)
    for result in data.get("results", []):
        key = result.get("input_key")
        if key and key not in keys:
            keys.append(key)
    return keys


def cmd_summarize(args: argparse.Namespace) -> None:
    """Read results files and emit tables + gate answers."""
    results_paths = args.results
    all_data: list[dict] = []
    for rp in results_paths:
        data = json.loads(Path(rp).read_text())
        all_data.append(data)

    # Merge results across files (for multi-machine).
    all_results: list[dict] = []
    machines: list[str] = []
    for data in all_data:
        machine = data.get("machine", {}).get("cpu_brand", "unknown")
        machines.append(machine)
        for r in data.get("results", []):
            r["_machine"] = machine
            all_results.append(r)

    # Telemetry files from fixed path pattern.
    telemetry_glob = str(_BAKEOFF_DIR / "powermetrics_config_*.txt")
    telemetry_files = sorted(globmod.glob(telemetry_glob))
    telemetry: dict[str, dict] = {}
    for tf in telemetry_files:
        name = Path(tf).stem.replace("powermetrics_config_", "")
        telemetry[name] = _parse_telemetry(tf)

    # Build per-machine, per-config tables.
    lines: list[str] = []
    lines.append("# Kokoro TTS Bakeoff Summary\n")
    lines.append(f"Generated from: {', '.join(results_paths)}\n")

    for data in all_data:
        machine = data.get("machine", {}).get("cpu_brand", "unknown")
        mem_gb = data.get("machine", {}).get("memory_gb", "?")
        git = data.get("git_commit", "?")[:12]
        lines.append(f"\n## Machine: {machine} ({mem_gb} GB)\n")
        lines.append(f"Git: `{git}` | Order seed: {data.get('order_seed', '?')}\n")

        results = [r for r in data.get("results", []) if r.get("status") == "ok"]
        configs_present = sorted(set(r["config"] for r in results))
        input_keys = _ordered_input_keys(data)

        # Wall time table.
        lines.append("\n### Wall Time (seconds)\n")
        headers = ["Config", "Input", "Mean", "Median", "Std", "Min", "Max", "N"]
        align = ["l", "l", "r", "r", "r", "r", "r", "r"]
        rows = []
        for cfg in configs_present:
            for ik in input_keys:
                vals = [r["wall_time_s"] for r in results
                        if r["config"] == cfg and r.get("input_key") == ik and r["wall_time_s"] is not None]
                s = _stats(vals)
                if s["n"] == 0:
                    continue
                rows.append([
                    CONFIG_LABELS.get(cfg, cfg), ik,
                    f"{s['mean']:.4f}", f"{s['median']:.4f}", f"{s['std']:.4f}",
                    f"{s['min']:.4f}", f"{s['max']:.4f}", str(s["n"]),
                ])
        lines.append(_format_table(headers, rows, align) + "\n")

        # RTF table.
        lines.append("\n### RTF (Canonical)\n")
        headers = ["Config", "Input", "Mean", "Median", "Std", "Min", "Max"]
        align = ["l", "l", "r", "r", "r", "r", "r"]
        rows = []
        for cfg in configs_present:
            for ik in input_keys:
                vals = [r["rtf_canonical"] for r in results
                        if r["config"] == cfg and r.get("input_key") == ik and r.get("rtf_canonical") is not None]
                s = _stats(vals)
                if s["n"] == 0:
                    continue
                rows.append([
                    CONFIG_LABELS.get(cfg, cfg), ik,
                    f"{s['mean']:.4f}", f"{s['median']:.4f}", f"{s['std']:.4f}",
                    f"{s['min']:.4f}", f"{s['max']:.4f}",
                ])
        lines.append(_format_table(headers, rows, align) + "\n")

        # Config A stage breakdown.
        a_results = [r for r in results if r["config"] == "a"]
        if a_results:
            lines.append("\n### Config A Stage Breakdown\n")
            stage_keys = [
                ("t_prefix_extract_s", "Prefix extract"),
                ("t_har_builder_cpu_s", "HAR builder (CPU)"),
                ("t_decoder_pre_cpu_s", "Decoder pre (CPU)"),
                ("t_coreml_predict_s", "CoreML predict"),
                ("t_trim_s", "Trim"),
                ("t_orchestration_s", "Orchestration"),
            ]
            headers = ["Stage", "Mean (ms)", "Median (ms)", "% of wall"]
            align = ["l", "r", "r", "r"]
            rows = []
            wall_times = [r["wall_time_s"] for r in a_results if r["wall_time_s"]]
            median_wall = statistics.median(wall_times) if wall_times else 1.0
            for key, label in stage_keys:
                vals = [r.get(key) for r in a_results if r.get(key) is not None]
                if not vals:
                    continue
                s = _stats(vals)
                pct = (s["median"] / median_wall * 100) if median_wall > 0 else 0
                rows.append([
                    label,
                    f"{s['mean'] * 1000:.1f}",
                    f"{s['median'] * 1000:.1f}",
                    f"{pct:.1f}%",
                ])
            lines.append(_format_table(headers, rows, align) + "\n")

    # --- Gate Answers ---
    lines.append("\n---\n")
    lines.append("\n## Gate Answers\n")

    # Collect aggregate stats for gate analysis.
    ok_results = [r for r in all_results if r.get("status") == "ok"]
    gate_input_keys: list[str] = []
    for data in all_data:
        for key in _ordered_input_keys(data):
            if key not in gate_input_keys:
                gate_input_keys.append(key)

    def _config_median(cfg: str, field: str = "wall_time_s") -> float | None:
        vals = [r[field] for r in ok_results if r["config"] == cfg and r.get(field) is not None]
        return statistics.median(vals) if vals else None

    def _config_median_by_input(cfg: str, ik: str, field: str = "wall_time_s") -> float | None:
        vals = [r[field] for r in ok_results if r["config"] == cfg and r.get("input_key") == ik and r.get(field) is not None]
        return statistics.median(vals) if vals else None

    # Gate 1: ANE participation.
    lines.append("\n### Gate 1: Does the naive decoder-only Core ML artifact use ANE under `.all`?\n")
    b_wall = _config_median("b")
    c_wall = _config_median("c")
    b_telemetry = telemetry.get("b_all", {})
    c_telemetry = telemetry.get("c_cpu_and_gpu", {})

    if b_wall is not None and c_wall is not None:
        ratio = b_wall / c_wall if c_wall > 0 else float("inf")
        lines.append(f"- Config B median wall time: {b_wall:.4f}s\n")
        lines.append(f"- Config C median wall time: {c_wall:.4f}s\n")
        lines.append(f"- B/C ratio: {ratio:.3f}\n")
        if b_telemetry.get("median_mw") is not None:
            lines.append(f"- Config B ANE power (steady median): {b_telemetry['median_mw']} mW\n")
        if c_telemetry.get("median_mw") is not None:
            lines.append(f"- Config C ANE power (steady median): {c_telemetry['median_mw']} mW\n")

        # Classification.
        if b_telemetry.get("median_mw") is not None and c_telemetry.get("median_mw") is not None:
            delta = b_telemetry["median_mw"] - c_telemetry["median_mw"]
            if delta > 10:
                lines.append(f"\n**ane_participation: yes** (ANE power delta {delta:.1f} mW > 10 mW threshold)\n")
            elif delta > 0:
                lines.append(f"\n**ane_participation: indeterminate** (ANE power delta {delta:.1f} mW, within 0-10 mW range)\n")
            else:
                lines.append(f"\n**ane_participation: no** (no ANE power increase for .all vs .cpuAndGPU)\n")
        else:
            # No telemetry -- use latency comparison.
            if ratio < 0.85:
                lines.append(f"\n**ane_participation: indeterminate** (B is faster than C by {(1-ratio)*100:.1f}%, "
                             f"suggestive of different scheduling, but no telemetry data for definitive proof)\n")
            elif ratio > 1.15:
                lines.append(f"\n**ane_participation: indeterminate** (B is slower than C, no telemetry data)\n")
            else:
                lines.append(f"\n**ane_participation: indeterminate** (B and C have similar latency, no telemetry data)\n")
    else:
        lines.append("\nInsufficient data for Gate 1 (configs B and/or C not available).\n")

    # Gate 2: Shipping hybrid speedup.
    lines.append("\n### Gate 2: How large is the shipping hybrid speedup versus PyTorch CPU and MPS?\n")
    a_wall = _config_median("a")
    d_wall = _config_median("d")
    e_wall = _config_median("e")
    if a_wall is not None:
        lines.append(f"- Config A median wall time: {a_wall:.4f}s\n")
        if e_wall is not None:
            speedup_e = e_wall / a_wall if a_wall > 0 else 0
            lines.append(f"- Config E (CPU) median wall time: {e_wall:.4f}s\n")
            lines.append(f"- **A vs E speedup: {speedup_e:.1f}x**\n")
        if d_wall is not None:
            speedup_d = d_wall / a_wall if a_wall > 0 else 0
            lines.append(f"- Config D (MPS) median wall time: {d_wall:.4f}s\n")
            lines.append(f"- **A vs D speedup: {speedup_d:.1f}x**\n")
        a_rtf = _config_median("a", "rtf_canonical")
        e_rtf = _config_median("e", "rtf_canonical")
        d_rtf = _config_median("d", "rtf_canonical")
        if a_rtf:
            lines.append(f"- Config A median RTF: {a_rtf:.4f} ({1/a_rtf:.1f}x realtime)\n")
        if e_rtf:
            lines.append(f"- Config E median RTF: {e_rtf:.4f} ({1/e_rtf:.1f}x realtime)\n")
        if d_rtf:
            lines.append(f"- Config D median RTF: {d_rtf:.4f} ({1/d_rtf:.1f}x realtime)\n")
    else:
        lines.append("\nConfig A not available.\n")

    # Gate 3: Scaling with sequence length.
    lines.append("\n### Gate 3: How does the advantage scale with sequence length?\n")
    if a_wall is not None and e_wall is not None:
        headers = ["Input", "Duration (s)", "A wall (s)", "E wall (s)", "Speedup"]
        align = ["l", "r", "r", "r", "r"]
        rows = []
        for ik in gate_input_keys:
            a_ik = _config_median_by_input("a", ik)
            e_ik = _config_median_by_input("e", ik)
            # Get canonical duration from first result.
            dur_vals = [r["canonical_audio_duration_s"] for r in ok_results
                        if r.get("input_key") == ik and r.get("canonical_audio_duration_s") is not None]
            dur = dur_vals[0] if dur_vals else 0
            if a_ik is not None and e_ik is not None:
                sp = e_ik / a_ik if a_ik > 0 else 0
                rows.append([ik, f"{dur:.2f}", f"{a_ik:.4f}", f"{e_ik:.4f}", f"{sp:.1f}x"])
        if rows:
            lines.append(_format_table(headers, rows, align) + "\n")
    else:
        lines.append("\nInsufficient data for Gate 3.\n")

    # Gate 4: CPU-side overhead in Config A.
    lines.append("\n### Gate 4: How much CPU-side overhead remains in Config A?\n")
    a_results_ok = [r for r in ok_results if r["config"] == "a"]
    if a_results_ok:
        stage_keys = [
            ("t_prefix_extract_s", "Prefix extract"),
            ("t_har_builder_cpu_s", "HAR builder (CPU)"),
            ("t_decoder_pre_cpu_s", "Decoder pre (CPU)"),
            ("t_coreml_predict_s", "CoreML predict"),
            ("t_trim_s", "Trim"),
            ("t_orchestration_s", "Orchestration"),
        ]
        wall_times = [r["wall_time_s"] for r in a_results_ok if r["wall_time_s"]]
        median_wall_a = statistics.median(wall_times) if wall_times else 1.0
        lines.append(f"Config A median wall time: {median_wall_a * 1000:.1f} ms\n")
        for key, label in stage_keys:
            vals = [r.get(key) for r in a_results_ok if r.get(key) is not None]
            if vals:
                med = statistics.median(vals)
                pct = med / median_wall_a * 100 if median_wall_a > 0 else 0
                lines.append(f"- {label}: {med * 1000:.1f} ms ({pct:.1f}%)\n")
    else:
        lines.append("\nConfig A not available.\n")

    # Gate 5: M1 Mini footnote.
    lines.append("\n### Gate 5 (conditional footnote): Cross-machine scaling\n")
    m1_results = [r for r in all_results if "m1" in r.get("_machine", "").lower()]
    if m1_results:
        lines.append("M1 Mini data available -- see per-machine tables above.\n")
    else:
        lines.append("M1 Mini data not available for this benchmark run.\n")

    # Gate 6: Swift Core ML ANE ablation.
    lines.append("\n### Gate 6: Does `.all` beat `.cpuAndGPU` for the Swift Core ML pipeline?\n")
    f_wall = _config_median("f")
    g_wall = _config_median("g")
    if f_wall is not None and g_wall is not None:
        headers = ["Input", "Audio (s)", "F .all (ms)", "G .cpuAndGPU (ms)", "G/F"]
        align = ["l", "r", "r", "r", "r"]
        rows = []
        for ik in gate_input_keys:
            f_ik = _config_median_by_input("f", ik)
            g_ik = _config_median_by_input("g", ik)
            dur_vals = [
                r["canonical_audio_duration_s"] for r in ok_results
                if r.get("input_key") == ik and r.get("canonical_audio_duration_s") is not None
            ]
            dur = dur_vals[0] if dur_vals else 0
            if f_ik is not None and g_ik is not None:
                ratio = g_ik / f_ik if f_ik > 0 else 0
                rows.append([
                    ik,
                    f"{dur:.2f}",
                    f"{f_ik * 1000:.1f}",
                    f"{g_ik * 1000:.1f}",
                    f"{ratio:.2f}",
                ])
        if rows:
            lines.append(_format_table(headers, rows, align) + "\n")
        if g_wall < f_wall:
            lines.append(
                f"Config G is faster overall in this run "
                f"(median G/F={g_wall / f_wall:.2f}); `.all` does not isolate a latency win here.\n"
            )
        elif f_wall < g_wall:
            lines.append(
                f"Config F is faster overall in this run "
                f"(median F/G={f_wall / g_wall:.2f}); ANE participation may be helping, pending telemetry.\n"
            )
        else:
            lines.append("Config F and G have identical aggregate medians in this run.\n")
    else:
        lines.append("Insufficient data for Gate 6 (configs F and/or G not available).\n")

    # Write summary.
    summary_text = "\n".join(lines)
    summary_path = _BAKEOFF_DIR / "summary.md"
    summary_path.write_text(summary_text + "\n")
    print(summary_text)
    print(f"\nSummary written to: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Bakeoff Summarize")
    parser.add_argument("--results", required=True, nargs="+", help="Results JSON path(s)")
    args = parser.parse_args()
    cmd_summarize(args)


if __name__ == "__main__":
    main()
