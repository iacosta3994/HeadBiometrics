#!/usr/bin/env bash
# Sync measure helper. Usage:
#   ./examples/measure.sh [video_path] [scale_mode] [extra -F fields...]
# Env: BASE_URL (default http://127.0.0.1:8000), API_KEY (optional)

set -euo pipefail
BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
VIDEO="${1:-Video_Tests/Self.mp4}"
SCALE_MODE="${2:-mm_per_pixel}"
shift $(( $# >= 2 ? 2 : $# )) || true

AUTH_ARGS=()
if [[ -n "${API_KEY:-}" ]]; then
  AUTH_ARGS=(-H "X-API-Key: ${API_KEY}")
fi

EXTRA_ARGS=()
if [[ "$SCALE_MODE" == "mm_per_pixel" ]]; then
  EXTRA_ARGS=(-F "mm_per_pixel=0.25")
fi

curl -sS -X POST "${BASE_URL}/v1/measure" \
  "${AUTH_ARGS[@]}" \
  -F "video=@${VIDEO};type=video/mp4" \
  -F "clockwise=false" \
  -F "scale_mode=${SCALE_MODE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
echo
