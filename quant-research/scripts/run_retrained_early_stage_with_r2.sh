#!/bin/sh
set -eu

task_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$task_root"

PYTHONPATH=scripts:src .venv/bin/python scripts/token_history_run.py \
  --stage dense \
  --data artifacts/token-history-data-rebuilt-20260922-v1 \
  --output artifacts/token-history-retrained-20260922-v1

PYTHONPATH=scripts:src .venv/bin/python scripts/retrained_early_stage_compare.py \
  --config configs/retrained-early-stage-v1.json

PYTHONPATH=scripts:src .venv/bin/python scripts/archive_research_artifact.py \
  --archive-id token-history-retrained-20260922-v1 \
  --source artifacts/token-history-retrained-20260922-v1 \
  --output artifacts/r2-retention/token-history-retrained-20260922-v1

PYTHONPATH=scripts:src .venv/bin/python scripts/archive_research_artifact.py \
  --archive-id retrained-early-stage-20260922-v1 \
  --source artifacts/retrained-early-stage-20260922-v1 \
  --output artifacts/r2-retention/retrained-early-stage-20260922-v1
