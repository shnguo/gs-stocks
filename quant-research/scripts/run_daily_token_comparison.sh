#!/bin/bash
# Manual close validation, main incremental update, main report and frozen candidate comparison.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$task_root/scripts/run_daily_token.sh" \
  --shadow-config "$task_root/artifacts/token-ranking-20260916-v2-shadow.json" "$@"
