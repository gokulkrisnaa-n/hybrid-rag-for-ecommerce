#!/usr/bin/env bash
# End-to-end check of the Docker setup: build, start, and exercise the running app.
#
#   scripts/smoke_test.sh                 # includes one real chat turn (~$0.001 of OpenAI)
#   SMOKE_CHAT=0 scripts/smoke_test.sh    # everything except the chat turn
#
# Leaves the containers running; `docker compose down` stops them.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${BASE_URL:-http://localhost:8000}"
pass() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1" >&2; exit 1; }

echo "→ building and starting (waits for both health checks)"
docker compose up -d --build --wait

echo "→ checks against $BASE"
curl -fsS "$BASE/health/ready" | grep -q '"status":"ok"' \
  && pass "API ready: database reachable, vocabulary loaded" || fail "/health/ready is not ok"

curl -fsS "$BASE/" | grep -q '<div id="root">' \
  && pass "chat UI served at /" || fail "the UI is not served at /"

status=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/v1/does-not-exist")
[ "$status" = 404 ] && pass "unknown paths get the API's 404" || fail "expected 404, got $status"

if [ "${SMOKE_CHAT:-1}" != 0 ]; then
  out=$(curl -fsS -N --max-time 90 -X POST "$BASE/v1/chat/stream" \
          -H 'Content-Type: application/json' \
          -d '{"message": "black hoodie under $200"}')
  echo "$out" | grep -q '^event: delta' && pass "answer streamed" || fail "no streamed text"
  echo "$out" | grep -q '^event: answer' && pass "final answer received" \
    || fail "no answer event:
$(echo "$out" | tail -5)"
fi

echo "All checks passed. Open $BASE"