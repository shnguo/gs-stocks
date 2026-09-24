#!/bin/bash
# Continue matched models using existing verified daily inputs; manual only.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
export PYTHONPATH="$task_root/src:$task_root/scripts"
exec "$task_root/artifacts/kronos-comparison-20260910-v1/venv/bin/python" "$task_root/scripts/token_ranking_daily.py" "$@"
