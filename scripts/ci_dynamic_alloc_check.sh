#!/bin/bash
# scripts/ci_dynamic_alloc_check.sh
set -euo pipefail

# Fail on dynamic tensor creation in non-__init__ code paths
# Allowed occurrences:
#  - inside "def __init__(" methods
#  - as register_buffer initializers
#  - as nn.Parameter initializers
#  - comments and empty files are ignored

PATTERN='new_zeros\|zeros_like\|ones_like\|torch\.zeros\|torch\.ones\|torch\.randn\|torch\.rand'

VIOLATIONS=$(grep -R "${PATTERN}" --include="*.py" kokoro/ export_*.py \
  | grep -v "__init__" \
  | grep -v "register_buffer" \
  | grep -v "nn.Parameter" \
  | grep -v "^#" || true)

if [ -n "$VIOLATIONS" ]; then
  echo "ERROR: Dynamic tensor creation found in non-__init__ methods:"
  echo "$VIOLATIONS"
  exit 1
fi

echo "OK: No dynamic tensor allocations detected outside __init__"
