# Notes consolidation guide

April 1, 2026

## Purpose

Keep institutional memory in `README/Notes/` **high signal and low sprawl**. Prefer updating an existing domain note over adding a new file whenever the topic fits.

## When to update vs create

| Situation | Action |
| --- | --- |
| Same subsystem, new bug or follow-up | Add a section to the existing domain note |
| New symptom, same root area | Same file, new issue block using [Notes-template](../../Templates/Notes-template.md) |
| Entirely new domain with no home | Create a new topic-named file |

## Consolidation checklist

1. **Search** `README/Notes/` for the subsystem or keywords before writing.
2. **One entry point per domain** (e.g. one file for a pipeline, not one file per incident date).
3. **Active issues first** — keep current problems near the top; move resolved items down or mark resolved clearly.
4. **Link out** to guides for long explanations; keep the note to summary, symptom, fix, verification.

## Anti-patterns

- A new markdown file for every investigation when an existing note fits.
- Duplicating the same procedure in three notes — link to the canonical guide once.
- Notes that belong in `README/Guides/` — move durable how-tos to a guide; keep notes for time-bound debugging trails.

## Related

- [Notes template](../../Templates/Notes-template.md)
- `write-notes` skill (`.claude/skills/write-notes/`; `.cursor/skills` and `.agents/skills` symlink to `.claude/skills`)
