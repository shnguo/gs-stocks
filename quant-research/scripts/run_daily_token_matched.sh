#!/bin/bash
# Manual data validation, main update/report, matched continuation and outcome review.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "$task_root/scripts/run_daily_token.sh" \
  --matched-config "$task_root/configs/token-ranking-daily-v1.json" \
  --calibration-config "$task_root/configs/token-return-calibration-v1.json" "$@"
