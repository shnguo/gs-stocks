#!/bin/bash
# Manual isolated matched update and fixed-universe forecast validation.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root/src:$task_root/scripts"
task_python="$task_root/artifacts/kronos-comparison-20260910-v1/venv/bin/python"
"$task_python" "$task_root/scripts/token_ranking_incremental.py" "$@"
exec "$task_python" "$task_root/scripts/verify_token_ranking_incremental.py" "$@"
