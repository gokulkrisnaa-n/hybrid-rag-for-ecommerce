#!/usr/bin/env bash
# Writes requirements.lock: every runtime dependency (direct and indirect) pinned to
# the version installed in your working venv, resolved for Linux and Python 3.13
# (the Docker image). Run it with the venv active; commit the result.
#
#   scripts/lock_requirements.sh
#
# To upgrade later: upgrade in the venv, run the tests, run this again, commit.
# Dev tools (pytest) are not in the lock: the image doesn't need them.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${VIRTUAL_ENV:-}" ]; then
  echo "Activate the project's venv first (source .venv/bin/activate)." >&2
  exit 1
fi
python -m uv --version >/dev/null 2>&1 || python -m pip install --quiet uv

constraints=$(mktemp)
trap 'rm -f "$constraints"' EXIT
# What's installed now: the versions you have tested with.
python -m pip freeze --exclude-editable | grep -v '^-e ' > "$constraints"

python -m uv pip compile pyproject.toml \
  --constraint "$constraints" \
  --python-version 3.13 \
  --python-platform linux \
  --no-header \
  --no-annotate \
  --output-file requirements.lock

echo "Wrote requirements.lock ($(grep -c '==' requirements.lock) packages)."