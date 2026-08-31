#!/usr/bin/env bash

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$ROOT/outputs/position_feature_eda_final_20260731/llm_semantic_smoke"
LOG_PATH="$OUTPUT_DIR/full_run.log"

mkdir -p "$OUTPUT_DIR"
cd "$ROOT"
exec >>"$LOG_PATH" 2>&1

while true; do
  printf '\n[%s] Starting or resuming supervised full Gemini run.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  .venv/bin/python -u scripts/v2_run_gemini_position_feature_smoke.py \
    --all-urls \
    --execute-live-gemini
  exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    printf '[%s] Full Gemini run completed successfully.\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    exit 0
  fi
  printf '[%s] Runner exited with code %s; resuming from chunk cache in 15 seconds.\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$exit_code"
  sleep 15
done
