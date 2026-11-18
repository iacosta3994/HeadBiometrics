#!/usr/bin/env bash
# Async job helper: enqueue then poll until succeeded|failed.
# Usage:
#   ./examples/measure_async.sh [video_path] [scale_mode]
# Env: BASE_URL, API_KEY, POLL_INTERVAL_SEC (default 0.5)

set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
VIDEO="${1:-Video_Tests/Self.mp4}"
SCALE_MODE="${2:-mm_per_pixel}"
POLL_INTERVAL_SEC="${POLL_INTERVAL_SEC:-0.5}"

AUTH_ARGS=()
if [[ -n "${API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "X-API-Key: ${API_KEY}")
fi

EXTRA_ARGS=()
if [[ "$SCALE_MODE" == "mm_per_pixel" ]]; then
  EXTRA_ARGS=(-F "mm_per_pixel=0.25")
fi

JOB_JSON=$(curl -sS -X POST "${BASE_URL}/v1/measure/jobs" \
  "${AUTH_ARGS[@]}" \
  -F "video=@${VIDEO};type=video/mp4" \
  -F "clockwise=false" \
  -F "scale_mode=${SCALE_MODE}" \
  "${EXTRA_ARGS[@]}")

echo "$JOB_JSON"
JOB_ID=$(echo "$JOB_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

while true; do
  STATUS_JSON=$(curl -sS "${AUTH_ARGS[@]}" "${BASE_URL}/v1/measure/jobs/${JOB_ID}")
  STATUS=$(echo "$STATUS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
  echo "$STATUS_JSON"
  if [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]]; then
    break
  fi
  sleep "$POLL_INTERVAL_SEC"
done
