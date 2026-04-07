# Markdown authoring guide

April 1, 2026

## Purpose

Consistent, maintainable markdown across `README/`, guides, plans, and notes.

## Core rules

- Use **real markdown links** `[text](path)` for internal paths; avoid bare URLs in prose when a label helps.
- **Blank lines** around headings, lists, and fenced blocks so diffs and renderers stay predictable.
- **Language-tag** fenced code blocks when the language is known (` ```typescript `, ` ```bash `).
- **One trailing newline** at end of file.
- Prefer **one canonical explanation** — link to it instead of copying paragraphs across files.

## Document families

| Family | Template | Notes |
| --- | --- | --- |
| Plan | [Plans-template](../../Templates/Plans-template.md) | Under `README/Plans/...` |
| Note | [Notes-template](../../Templates/Notes-template.md) | Under `README/Notes/...` |
| Guide | [guide-template](../../Templates/guide-template.md) | Under `README/Guides/...` |

Preserve each document’s existing structure unless the task is an explicit restructure.

## Plans and notes

- Plans: phases, checkboxes, and links to shared contracts stay in sync with implementation.
- Notes: use the issue template sections; keep investigation logs dated and short.

## Verification

If the repo adds a markdown lint command (for example in `package.json` for
`kokoro.js/` or a root config), run it before large doc-only PRs.

## Related

- `markdown` skill (`.claude/skills/markdown/`; `.cursor/skills` and `.agents/skills` symlink to `.claude/skills`)
- [Notes consolidation guide](notes-consolidation-guide.md)
