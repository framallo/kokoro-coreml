#!/usr/bin/env bash
# Recreate project skill symlinks for Cursor and Agents (Codex-style) layouts.
#
# Canonical tree: .claude/skills/<skill-name>/SKILL.md
# Mirrors (same content via symlink):
#   .cursor/skills  -> ../.claude/skills
#   .agents/skills  -> ../.claude/skills
#
# Edit skills only under .claude/skills/. Do not duplicate SKILL.md trees.
# Usage: from repo root, ./scripts/ensure_repo_skill_symlinks.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TARGET="../.claude/skills"
SKILLS_DIR="$ROOT/.claude/skills"

if [[ ! -d "$SKILLS_DIR" ]]; then
  echo "error: missing $SKILLS_DIR (clone or create .claude/skills first)" >&2
  exit 1
fi

ensure_link() {
  local parent="$1"
  mkdir -p "$parent"
  local link="$parent/skills"
  if [[ -e "$link" && ! -L "$link" ]]; then
    echo "error: $link exists and is not a symlink; remove the duplicate tree and re-run" >&2
    exit 1
  fi
  rm -f "$link"
  ln -s "$TARGET" "$link"
  echo "ok: $link -> $TARGET"
}

ensure_link ".cursor"
ensure_link ".agents"
