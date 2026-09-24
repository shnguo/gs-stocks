#!/bin/bash
# Manual feature audit, matched training comparisons and report generation.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
exec /usr/bin/env PYTHONPATH="$task_root/src:$task_root/scripts" \
  "$task_root/artifacts/kronos-comparison-20260910-v1/venv/bin/python" \
  "$task_root/scripts/token_features_run.py" "$@"
