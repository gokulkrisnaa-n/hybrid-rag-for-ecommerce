#!/usr/bin/env bash
# Load a catalog dump into the Docker database, replacing what's there.
#
#   scripts/restore_db.sh backups/<file>
#
# Accepts a custom-format dump (pg_dump -Fc, any extension), a plain .sql file or a
# .sql.gz. The database is dropped and recreated first, so the result is exactly the
# dump: no leftovers from the schema the container created on its first start.
# The app is stopped while this runs and started again at the end.

set -euo pipefail
cd "$(dirname "$0")/.."

DUMP="${1:-}"
if [ -z "$DUMP" ] || [ ! -f "$DUMP" ]; then
  echo "usage: scripts/restore_db.sh <dump file>   (got: '${DUMP}')" >&2
  exit 1
fi

# Runs a command in the db container, where POSTGRES_USER / POSTGRES_DB are set.
# stdin is passed through (the dump); commands that read nothing get /dev/null.
in_db() { docker compose exec -T db sh -c "$1"; }

echo "→ starting the database"
docker compose up -d --wait db

app_was_running=$(docker compose ps --status running --services 2>/dev/null | grep -qx app && echo yes || echo no)
if [ "$app_was_running" = yes ]; then
  echo "→ stopping the app while the database is replaced"
  docker compose stop app >/dev/null
fi

echo "→ recreating an empty database"
# From template0, not the default template1: the ParadeDB image installs its extensions
# (and their schemas, e.g. "paradedb") into template1, and the dump creates them itself.
in_db 'dropdb -U "$POSTGRES_USER" --if-exists --force "$POSTGRES_DB" && createdb -U "$POSTGRES_USER" -T template0 "$POSTGRES_DB"' < /dev/null

# Custom-format dumps start with the bytes "PGDMP".
if [ "$(head -c 5 "$DUMP")" = "PGDMP" ]; then
  echo "→ restoring custom-format dump $DUMP"
  in_db 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-privileges --exit-on-error' < "$DUMP"
elif gzip -t "$DUMP" 2>/dev/null; then
  echo "→ restoring gzipped SQL $DUMP"
  gunzip -c "$DUMP" | in_db 'psql -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null'
else
  echo "→ restoring SQL $DUMP"
  in_db 'psql -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null' < "$DUMP"
fi

echo "→ checking what was loaded"
in_db 'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tA -F " " -c "
  SELECT count(*) FILTER (WHERE is_active), count(embedding) FILTER (WHERE is_active)
  FROM clean.products"' < /dev/null | {
  read -r active embedded
  echo "   active products: ${active}, with embeddings: ${embedded}"
  if [ "${active:-0}" -eq 0 ]; then
    echo "The dump loaded no active products: is it the right file?" >&2
    exit 1
  fi
}

if [ "$app_was_running" = yes ]; then
  echo "→ starting the app again"
  docker compose up -d app
fi
echo "Done."