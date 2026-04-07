# [Technology/Domain] Developer Field Guide

[Month Day, Year]

## Related Documentation

- **[Related Guide](../Guides/path/to/guide.md)**: One-line description of relationship.
- **[Related Plan](../Plans/path/to/plan.md)**: One-line description of relationship.

## 1. The Environment: Constraints and Mental Model

Frame the problem space before prescribing solutions. What is the developer working inside? What are the hard constraints of the platform, runtime, or protocol? What assumptions will bite them?

This section answers: "What do I need to understand about the world before I write a single line of code?"

### 1.1. [Core Constraint or Platform Reality]

Explain the constraint, why it exists, and what it means for the developer. Lead with the consequence, not the specification.

```javascript
// Show the constraint in action — what breaks, what the error looks like
```

### 1.2. [Another Constraint or Unintuitive Behavior]

Same pattern. Constraint → consequence → example.

## 2. Architecture and Core Concepts

How do the pieces fit together? What are the moving parts? What talks to what?

Keep this section conceptual. Implementation details come later. Use ASCII diagrams for system-level flows.

```
+-----------+     +-----------+     +-----------+
| Component | --> | Component | --> | Component |
+-----------+     +-----------+     +-----------+
```

### 2.1. [Key Concept or Subsystem]

Explain the concept, then show the simplest possible code that demonstrates it.

### 2.2. [Another Key Concept]

Same pattern. Concept → minimal code → gotcha if applicable.

## 3. Implementation Patterns

Battle-tested patterns for the most common tasks. Each pattern follows this structure:

### 3.1. [Pattern Name]

**Problem:** What you're trying to do and why the naive approach fails.

**Solution:**

```javascript
/**
 * Annotate code examples — explain WHY, not just WHAT.
 * Call out non-obvious decisions inline.
 */
function examplePattern() {
  // This timeout exists because [specific reason]
  const DEBOUNCE_MS = 500;
}
```

**Why this works:** One to two sentences connecting the solution back to the constraint from Section 1.

### 3.2. [Another Pattern]

Same structure. Problem → annotated code → why it works.

## 4. Gotchas, Edge Cases, and Failure Modes

The most valuable section of the guide. Developers read this when something breaks at 2 AM.

Organize by severity or frequency. Lead with the symptom (what the developer sees), then explain the cause and fix.

### 4.1. [Gotcha Name]

**Symptom:** What the developer observes (error message, silent failure, unexpected behavior).

**Cause:** Why this happens — connect to an architectural constraint from Section 1 if possible.

**Fix:**

```javascript
// Before (broken)
brokenApproach();

// After (fixed)
correctApproach();
```

**Why this is unintuitive:** One sentence explaining why a reasonable developer would get this wrong.

### 4.2. [Another Gotcha]

Same structure. Symptom → cause → fix → why it's unintuitive.

## 5. What NOT to Do (Anti-Patterns)

Explicitly document approaches that seem reasonable but cause problems. For every anti-pattern, explain what to do instead.

| Anti-Pattern | Why It Fails | Do This Instead |
| --- | --- | --- |
| [Naive approach] | [What breaks] | [Correct approach] |
| [Common mistake] | [Consequence] | [Alternative] |

## 6. Decision Framework

When multiple valid approaches exist, provide a comparison table so the developer can choose based on their constraints.

| Approach | Pros | Cons | Use When |
| --- | --- | --- | --- |
| [Option A] | [Strengths] | [Weaknesses] | [Scenario] |
| [Option B] | [Strengths] | [Weaknesses] | [Scenario] |

## 7. Debugging Playbook

Step-by-step diagnostic procedures for common failure scenarios.

### When [symptom occurs]

1. Check [first thing] — rules out [cause A].
2. Look at [second thing] — if [condition], then [cause B].
3. Try [diagnostic command or code] — confirms [cause C].

```bash
# Diagnostic command with explanation
```

## 8. Production Hardening

Patterns for reliability, monitoring, and graceful degradation in production. Skip this section for guides that cover development-only topics.

### 8.1. [Reliability Pattern]

What to monitor, what thresholds to set, what to do when things degrade.

## Works Cited

Numbered references for claims, specifications, or external documentation used in this guide.

1. [Source Name](https://example.com) — brief description of what it covers.
2. [Source Name](https://example.com) — brief description of what it covers.

---

<!--
USAGE NOTES:

1. This template structures deep-research output into a field guide. It is
   designed to be filled by an AI researcher (Gemini Deep Research, Claude,
   etc.) given a prompt like:

   "Create an advanced developer field guide for [topic]. What are the best
   practices? Known limitations? Common bugs? Unintuitive design patterns?
   Weird edge cases? Write it like a cheat sheet for an experienced developer.
   Go light on theory, heavy on examples with code. For every problem,
   propose a solution."

2. Section order matters. Constraints first (Section 1), architecture second
   (Section 2), implementation third (Section 3), failure modes fourth
   (Section 4). This mirrors how developers actually learn a domain.

3. Cross-references in "Related Documentation" are added manually AFTER the
   guide is generated. The deep research tool does not have access to our
   codebase, so leave this section as a placeholder during generation and
   populate it when integrating the guide.

4. Optional sections: Delete sections that don't apply. A guide about a
   simple API integration may not need "Production Hardening" or a full
   "Decision Framework." A debugging-focused guide may skip "Architecture."

5. Code examples: Annotate every code block. Explain WHY, not just WHAT.
   Before/after pairs for gotchas. Named constants instead of magic numbers.
   This is an LLM-first codebase — every comment is a prompt.

6. Anti-patterns (Section 5): Always include. This is what separates a field
   guide from a tutorial. Experienced developers need to know what NOT to do.

7. Gotchas (Section 4): This is the highest-value section. When in doubt,
   add more gotchas. Developers consult field guides when something breaks.

8. Target length: 400-800 lines. Under 400 suggests missing depth. Over 800
   suggests the guide should be split into focused sub-guides.

9. Filename convention: [topic]-guide.md (lowercase, hyphenated). Place in
   the most specific subdirectory under README/Guides/.
-->
