# Notes

Use this folder for **time-bound debugging trails**, **investigation writeups**,
and **audit notes**—see the **`write-notes`** skill (`.claude/skills/write-notes/`).

Prefer **one domain file** with multiple issue sections over a new file per
session; see [../Guides/content/notes-consolidation-guide.md](../Guides/content/notes-consolidation-guide.md).

Long-running learnings also appear under `README/learnings.md` and other
top-level `README/*.md` files; link from notes instead of duplicating.

## Index

- [Core ML compute-unit ablation](coreml-compute-unit-ablation.md) - F/G/G-prime/G-double-prime benchmark logic for isolating `.all`, ANE, GPU, and CPU behavior in the Swift pipeline.
- [Restarted Kokoro guide triage](kokoro-restarted-guide-triage-2026-06-06.md) - Draft Deep Research report triage for warmed lower-end Mac optimization, benchmark hygiene, and source/body candidate priority.
- [Kokoro strict source/HAR representation repair prompt](Kokoro-strict-source-HAR-representation-repair-deep-research-prompt.md) - External research brief for strict source/HAR representation repair, first-layer folding, and tiny adapter calibration.
- [iPhone Core ML device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md) - Device setup, foreground policy, and evidence capture for future physical-iPhone benchmark rows.
