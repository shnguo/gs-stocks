#!/bin/bash
# Manual training, matched forecasts, independent verification and frozen-Top20 report.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
task_python="$task_root/artifacts/kronos-comparison-20260910-v1/venv/bin/python"
export PYTHONPATH="$task_root/src:$task_root/scripts"
"$task_python" "$task_root/scripts/token_ranking_run.py" "$@"
exec "$task_python" "$task_root/scripts/finalize_token_ranking.py" "$@"
