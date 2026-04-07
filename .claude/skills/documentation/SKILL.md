---
name: documentation
description: Write or review inline code documentation that captures domain knowledge, non-obvious constraints, non-greppable cross-file contracts, and state lifecycle. Use when the task is adding or reviewing Python docstrings, file headers, state docs, or constant rationale. Do not use for README files, markdown docs, plan documents, or code changes where comments are incidental.
---

# Documentation

## Purpose

Enforce the repo's inline code documentation standards. Document what a future
editor cannot safely derive from the code itself.

## Use When

- Adding or reviewing Python docstrings or file headers.
- Documenting state lifecycle, constant rationale, or Core ML / export gotchas.
- A PR review flags missing or low-quality documentation.

## Do Not Use When

- Writing README, plan, or notes documents.
- General code changes where documentation isn't the focus.
- Markdown formatting issues (that's linting, not documentation).

## Procedure

1. Read [references/index.md](references/index.md) first.
2. Inspect the target file and the smallest set of related files needed to
   confirm what context is truly missing.
3. Add or tighten docs only where they capture:
   - domain knowledge
   - non-obvious constraints
   - non-greppable cross-file contracts
   - state lifecycle or constant rationale
4. Prefer short, durable comments over boilerplate:
   - short file headers
   - docstrings that explain why or constraints
   - state docs that explain lifetime and persistence
   - constant comments that explain why the value exists
5. Do not add manual call graphs, line-by-line prose, or comments that are more
   likely to drift than to help.
6. If the missing context actually belongs in a canonical guide, update the
   guide as well instead of burying the whole explanation in code comments.

## References

Read [references/index.md](references/index.md) first.

## Handoff Rules

- Hand off to **`debug`** if the real issue is a runtime bug, not missing docs.
- Hand off to normal refactoring flow if the task is structural change, not
  documentation.
