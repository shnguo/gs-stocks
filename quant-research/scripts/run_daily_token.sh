#!/bin/bash
# Manual entry point: validate closed data, incrementally train, forecast and publish.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
exec /usr/bin/env PYTHONPATH="$task_root/src:$task_root/scripts" \
  "$task_root/.venv/bin/python" \
  "$task_root/scripts/daily_token_cycle.py" \
  --config "$task_root/configs/daily-token-v2.json" \
  --r2-retain "$@"
