---
name: audit
description: Triggered when the user’s message includes the word **audit** (primary routing hook). Findings-first review of the kokoro-coreml repo or a scoped slice (paths, diff, commits)—runs pytest (and optional lint when configured) as mechanical signals, optionally delegates readonly subagents by charter when scope or risk warrants it (**when in doubt, parallelize**), merges and dedupes findings, and assigns A–F grades for architecture, correctness risk, and complexity debt. Do not use when the user wants implementation fixes unless they explicitly ask to fix issues after the audit—for plan-phase checklists against an active plan, prefer phase-audit.
---

# Audit

## Purpose

Run a **structured, paranoid-friendly** audit focused on **bugs** and **needless complexity** in this **PyTorch → Core ML** repo. The orchestrator **does not** silently rewrite code; it **surfaces** issues with severity, paths, and letter grades.

**Posture:** use **judgment**. Narrow scope (few files, small diff, localized
change) can be a **single-agent** pass. **Whole-repo**, **large diff**, **high
blast radius**, or **you are unsure** → prefer **multiple readonly subagents in
parallel** (one turn, several `Task` calls), each with a **narrow charter**, then
**merge and dedupe** into one report. **When in doubt, use multiple agents.**

## Use When

- The user’s message includes **`audit`** as a substring (primary trigger)—e.g.
  **audit**, **audit this**, **security audit**, **use the audit skill** (case
  insensitive in normal routing).

## Do Not Use When

- The user wants **implementation** only—unless they ask for fixes **after**
  the audit lands.
- The work is **only** validating an `execute-plan` phase against the plan and
  rubric—use **`phase-audit`** instead (this skill is broader).

## Scope (ask or infer once)

Establish **audit scope** before delegating:

| Kind | How to bound |
| --- | --- |
| **Whole codebase** | Repo root; subagents apportion by area (`kokoro/`, `examples/`, `coreml/`, `kokoro.js/`, export scripts) or by charter only. |
| **Paths** | User-provided globs or directories; all charters focus there. |
| **Git delta** | User provides base (`main`, `origin/main`, tag) or “last N commits”; use `git diff`, `git log -n`, `git diff-tree --name-only`. |
| **Single feature** | User names flows (e.g. “duration export”, “decoder-only 3s”); map to directories from `README.md` and `README/*.md`. |

State the **chosen scope** in the final report header.

## Mechanical signals (orchestrator, once)

From **repository root**, when the audited surface includes **Python** (default for this repo):

1. **`pytest`** — run the test suite when pytest is installed (`python -m pytest` if needed).
2. **Lint** — only if the repo defines a standard lint command (e.g. in `pyproject.toml` or docs); otherwise skip and note “no configured lint.”

Report failures as **P0 / Critical** findings with command output (summarize if
huge). Do not “fix” unless the user later asks.

**Optional:** targeted **`kokoro.js`** checks (`npm test` in `kokoro.js/`) when the
change touches the JS package; failures are **P0** for that surface.

When subagents run, they should assume mechanical checks ran **once**; they
focus on **review**, not on re-running gates unless a charter requires
spot-checking a file.

## Subagent delegation (use judgment)

**When one agent is enough:** tiny or localized scope (e.g. a handful of files,
one worker, or a single focused diff); low coupling; user asked for a quick
pass. The orchestrator performs the full charter coverage **solo** (still hit
architecture, correctness, security/ops, and complexity—just in one pass).

**When to parallelize:** whole-repo or multi-package audit; large `git` range;
security- or money-sensitive paths; queue/cron/webhook/idempotency concerns; or
**any uncertainty** about depth. **When in doubt, launch parallel subagents.**

**If parallelizing:** in **one assistant turn**, launch **up to four**
`Task` subagents with `readonly: true`, **`subagent_type`:** `generalPurpose`
(or `explore` when the request is mostly mapping file layout). Use **`model: fast`**
unless the user asks for maximum depth. You may use **fewer than four** if scope
is medium (e.g. two agents—correctness+security vs architecture+complexity).

Give each subagent:

- The **exact scope** (paths, diff summary, or “whole repo”).
- One **charter** from below (copy the charter text into the task prompt).
- Instruction: **concrete file paths**, **severity** (P0–P3), **one paragraph
  max per finding**, **no generic advice**.

### Charter 1 — Architecture & modules

God modules; separation between **export scripts**, **`kokoro/`** library code,
and **runtime-facing** `coreml/` artifacts; public surfaces; fan-in choke points;
misplaced orchestration vs model math. Flag files approaching **~1000 LOC** or
cramming unrelated responsibilities (**repo norm:** well under 1k LOC per file).

### Charter 2 — Correctness & reliability

Logic bugs; tensor shape mistakes; tracing/export mismatches; numerical drift;
error paths; swallowed exceptions; **async** misuse in JS if in scope; boundary
conditions for **static buckets** and Swift-side alignment; Core ML runtime
assumptions vs actual graphs.

### Charter 3 — Security, privacy, operational

Secrets in logs or committed credentials; unsafe subprocess usage; obvious **path**
or **checkpoint** foot-guns; operational mismatches between **documented I/O**
and code **when relevant to scope**.

### Charter 4 — Complexity, duplication, maintainability

Needless abstraction; configuration or branching explosion; **duplication with
drift** between export paths; dead code; **comment/doc lies** vs `README/` or
`CLAUDE.md`; weak tests; naming that hides behavior.

## Consolidation (orchestrator)

After **subagents return** or after a **solo** review pass:

1. **Dedupe:** merge findings that cite the same root cause or file+theme.
2. **Severity:** **P0** critical (wrongness, security, data loss, build breaks),
   **P1** high, **P2** medium, **P3** low / hygiene.
3. **Grades:** assign three letter grades **A–F** using [Grading rubric](#grading-rubric).
   **Overall grade = worst of the three** (if any dimension is **D**, overall
   cannot be **B** or higher—cap at **D** unless you justify an exception in one
   sentence).
4. **New teammate test:** one line—would a newcomer likely break this area in
   week one? (y/n + why)

## Grading rubric

Assign **Architecture**, **Correctness risk**, and **Complexity debt** separately.

| Grade | Meaning |
| --- | --- |
| **A** | Clear boundaries, hard to misuse, simple where it matters, issues are cosmetic. |
| **B** | Solid with minor debt; a few focused fixes would raise to A. |
| **C** | Meaningful issues; would block merges in a strict shop without a plan. |
| **D** | Serious structural or reliability risk; bounded scope but unsafe. |
| **F** | Unsafe, incomprehensible, or broken; needs pause and redesign or revert. |

**Correctness risk** includes likelihood of **latent bugs** and **failure-mode**
holes—not only existing failing tests.

## Output template

```text
## Audit scope
- ...

## Mechanical checks
- pytest: pass | fail (summary)
- lint (if configured): pass | fail (summary)
- kokoro.js tests (if run): ...

## Grades
- Architecture: ?
- Correctness risk: ?
- Complexity debt: ?
- Overall (worst-of-three): ?

## Findings (severity order)
### P0 — Critical
- ...

### P1 — High
- ...

### P2 — Medium
- ...

### P3 — Low
- ...

## Delegation / overlap notes
- Solo vs N subagents; deduped: ...

## Residual risks / what we did not run
- ...

## New teammate test
- ...
```

## Anti-patterns

- **Whole-repo or high-risk audit in a single shallow pass** when depth was
  needed—**when in doubt, parallelize.**
- **Four subagents for a three-line diff**—wasted latency; judge proportionally.
- **Findings without paths**—every substantive issue should anchor to a file or
  symbol when possible.
- **Auto-implementing** during audit when the user only asked for review.
- **Grade inflation**—if Correctness is **D**, overall is not **B**.
- **Ignoring mechanical failures**—typecheck/lint red is at least **P0** for
  code health.

## Relation to other skills

- **`phase-audit`:** plan-phase completion vs rubric and plan checkboxes.
- **`git-commit`:** post-commit **non-blocking** **`pytest`** + **`HEAD`** diff
  scan only—**not** a substitute for **`audit`** (no full matrix by default, no
  grades, no multi-agent).
- **`deploy`:** defines what “ship” means here (no Cloudflare scripts); audit does
  not replace user-intent release checks.
