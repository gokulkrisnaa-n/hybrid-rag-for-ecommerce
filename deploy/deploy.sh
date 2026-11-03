#!/usr/bin/env bash
# Deploy the latest code on the server: pull, rebuild, restart, check.
#
#   deploy/deploy.sh                  # includes one real chat turn (~$0.001)
#   SMOKE_CHAT=0 deploy/deploy.sh     # skip the chat turn
#
# Runs on the server, in the repo. .env must set COMPOSE_FILE (both compose files)
# and SITE_ADDRESS; see the deployment guide.

set -euo pipefail
cd "$(dirname "$0")/.."

[ -f .env ] || { echo "No .env here: create it first (see the deployment guide)." >&2; exit 1; }
SITE=$(grep -E '^SITE_ADDRESS=' .env | cut -d= -f2- | tr -d '"' || true)
[ -n "$SITE" ] || { echo "Set SITE_ADDRESS in .env." >&2; exit 1; }
grep -q '^COMPOSE_FILE=.*docker-compose.prod.yml' .env \
  || { echo "Set COMPOSE_FILE=docker-compose.yml:docker-compose.prod.yml in .env." >&2; exit 1; }

echo "→ pulling the latest code"
git pull --ff-only

echo "→ building and restarting (the database keeps running)"
docker compose up -d --build --wait --remove-orphans

echo "→ removing old images"
docker image prune -f >/dev/null

echo "→ checking https://$SITE"
BASE_URL="https://$SITE" scripts/smoke_test.sh