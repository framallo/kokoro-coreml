#!/usr/bin/env python3
"""Check whether remote Mac hosts are quiet enough for publishable timing."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("outputs/external_bakeoff")
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "remote_host_quiet_latest.md"
DEFAULT_JSON_OUTPUT = DEFAULT_OUTPUT_DIR / "remote_host_quiet_latest.json"
DEFAULT_HOSTS = (
    "irvine-m1=mattmireles@irvine-m1.local",
    "m2-air=mattmireles@m2-air.local",
)
NOISY_PROCESS_PATTERNS = (
    "mds",
    "mdworker",
    "mediaanalysisd",
    "mediaanalysisd-access",
    "photoanalysisd",
)


@dataclass(frozen=True)
class ProcessSample:
    """One process row from the remote host."""

    cpu_pct: float
    command: str
    noisy: bool


def _run_ssh(target: str, command: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={timeout_s}",
            target,
            command,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_s + 5,
    )


def _parse_load(uptime: str) -> tuple[float | None, float | None, float | None]:
    match = re.search(r"load averages?:\s*([0-9.]+)[, ]+([0-9.]+)[, ]+([0-9.]+)", uptime)
    if not match:
        return (None, None, None)
    return tuple(float(match.group(index)) for index in (1, 2, 3))  # type: ignore[return-value]


def _parse_processes(ps_output: str) -> list[ProcessSample]:
    rows: list[ProcessSample] = []
    for line in ps_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            cpu_pct = float(parts[0])
        except ValueError:
            continue
        command = parts[1]
        command_lower = command.lower()
        noisy = any(pattern in command_lower for pattern in NOISY_PROCESS_PATTERNS)
        rows.append(ProcessSample(cpu_pct=cpu_pct, command=command, noisy=noisy))
    return rows


def _host_status(
    machine_id: str,
    target: str,
    *,
    max_load_1: float,
    max_noisy_cpu_pct: float,
    timeout_s: int,
) -> dict[str, Any]:
    command = "uptime; ps -Ao pcpu,comm | sort -nr | head -12"
    result = _run_ssh(target, command, timeout_s)
    if result.returncode != 0:
        return {
            "machine_id": machine_id,
            "target": target,
            "ok": False,
            "quiet": False,
            "error": result.stderr.strip() or result.stdout.strip(),
            "returncode": result.returncode,
        }
    lines = result.stdout.splitlines()
    uptime = lines[0] if lines else ""
    ps_output = "\n".join(lines[1:])
    load_1, load_5, load_15 = _parse_load(uptime)
    processes = _parse_processes(ps_output)
    noisy_processes = [row for row in processes if row.noisy and row.cpu_pct >= max_noisy_cpu_pct]
    load_ok = load_1 is not None and load_1 <= max_load_1
    noisy_ok = not noisy_processes
    quiet = load_ok and noisy_ok
    blockers: list[str] = []
    if not load_ok:
        blockers.append(f"load1 {load_1!r} exceeds {max_load_1:.2f}")
    for row in noisy_processes:
        blockers.append(f"{row.command} at {row.cpu_pct:.1f}% CPU")
    return {
        "machine_id": machine_id,
        "target": target,
        "ok": True,
        "quiet": quiet,
        "blockers": blockers,
        "thresholds": {
            "max_load_1": max_load_1,
            "max_noisy_cpu_pct": max_noisy_cpu_pct,
            "noisy_process_patterns": list(NOISY_PROCESS_PATTERNS),
        },
        "uptime": uptime,
        "load": {
            "one_minute": load_1,
            "five_minutes": load_5,
            "fifteen_minutes": load_15,
        },
        "processes": [asdict(row) for row in processes],
    }


def _parse_host(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("host must be MACHINE=ssh-target")
    machine_id, target = value.split("=", 1)
    if not machine_id or not target:
        raise argparse.ArgumentTypeError("host must be MACHINE=ssh-target")
    return machine_id, target


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build quiet-host status for all requested hosts."""

    host_values = args.host if args.host else list(DEFAULT_HOSTS)
    hosts = [_parse_host(value) for value in host_values]
    rows = [
        _host_status(
            machine_id,
            target,
            max_load_1=args.max_load_1,
            max_noisy_cpu_pct=args.max_noisy_cpu_pct,
            timeout_s=args.timeout,
        )
        for machine_id, target in hosts
    ]
    return {
        "checked_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "publishable_timing_allowed": all(row.get("quiet") for row in rows),
        "rows": rows,
    }


def _fmt_load(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(payload: dict[str, Any]) -> str:
    """Render quiet-host status as Markdown."""

    lines = [
        "# Remote Host Quiet Status",
        "",
        f"Checked at local time: `{payload['checked_at_local']}`.",
        "",
        "| Machine | Quiet | Load 1/5/15 | Blockers |",
        "| --- | --- | ---: | --- |",
    ]
    for row in payload["rows"]:
        if not row.get("ok"):
            blockers = row.get("error") or "ssh failed"
            load = "n/a"
        else:
            load_payload = row["load"]
            load = "/".join(
                [
                    _fmt_load(load_payload["one_minute"]),
                    _fmt_load(load_payload["five_minutes"]),
                    _fmt_load(load_payload["fifteen_minutes"]),
                ]
            )
            blockers = "; ".join(row.get("blockers") or []) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['machine_id']}`",
                    "`yes`" if row.get("quiet") else "`no`",
                    load,
                    blockers,
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Publishable lower-end Mac timing is allowed only when every target row is",
            "`quiet=yes`. If a host is noisy, skip warmed frontier promotion and record",
            "the blocker instead.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", action="append", default=None, help="MACHINE=ssh-target")
    parser.add_argument("--max-load-1", type=float, default=1.0)
    parser.add_argument("--max-noisy-cpu-pct", type=float, default=10.0)
    parser.add_argument("--timeout", type=int, default=5)
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
                "output": str(args.output),
                "publishable_timing_allowed": payload["publishable_timing_allowed"],
                "quiet_hosts": sum(1 for row in payload["rows"] if row.get("quiet")),
                "total_hosts": len(payload["rows"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
