#!/bin/bash
# Manual comparison/review using a completed daily run; no data refresh or training.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
exec /usr/bin/env PYTHONPATH="$task_root/src:$task_root/scripts" \
  "$task_root/artifacts/kronos-comparison-20260910-v1/venv/bin/python" \
  "$task_root/scripts/token_ranking_shadow.py" "$@"
