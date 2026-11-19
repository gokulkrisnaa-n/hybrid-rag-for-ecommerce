#!/usr/bin/env bash
# Set up the vector-index benchmark databases (docker-compose.bench.yml).
#
#   scripts/bench_setup.sh up                      start both databases
#   scripts/bench_setup.sh restore <dump>          load the same dump into both
#   scripts/bench_setup.sh index pgvector          0.27: BM25 index + pgvector HNSW   (variant B)
#   scripts/bench_setup.sh index paradedb [opts]   0.27: BM25 + vector in ONE ParadeDB index,
#                                                  HNSW dropped                         (variant C)
#   scripts/bench_setup.sh status                  versions, indexes and their sizes
#   scripts/bench_setup.sh explain                 0.27 query plans: is the vector search in the index?
#   scripts/bench_setup.sh down                    stop (data kept; `docker compose -f
#                                                  docker-compose.bench.yml down -v` deletes it)
#
# The variants:
#   A  pdb026 (ParadeDB 0.26.1) as restored: BM25 + HNSW, what production runs
#   B  pdb027 (ParadeDB 0.27.0) as restored: the same indexes on the new version
#   C  pdb027 after `index paradedb`: the native vector index instead of HNSW
# A restore leaves pdb027 at B; `index pgvector` goes back to B from C.
#
# `opts` are passed to the ParadeDB index's WITH (...), e.g. to try another layout:
#   scripts/bench_setup.sh index paradedb "target_leaf_size=200, replication=3"
#
# Every index build is timed and its size recorded in evals/results/vector_bench_builds.tsv,
# which the benchmark report reads. Only the bench databases are touched, never the app's.

set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE_FILE=docker-compose.bench.yml
BUILDS=evals/results/vector_bench_builds.tsv

compose() { docker compose -f "$COMPOSE_FILE" "$@"; }

# Runs a shell command in a bench container, where POSTGRES_USER / POSTGRES_DB are set.
# stdin is passed through (a dump or SQL).
in_db() { local svc=$1; shift; compose exec -T "$svc" sh -c "$*"; }

# psql in a bench container; SQL comes on stdin, extra psql flags as arguments.
# The flags go through `sh -c` as one string, so keep them to simple ones like -tA or -q
# (no quoted values: do formatting in the SQL instead).
psql_in() { local svc=$1; shift; in_db "$svc" 'psql -X -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" '"$*"; }

# One value from a query.
scalar() { printf '%s\n' "$2" | psql_in "$1" -tA; }

# The columns of the BM25 index, exactly as in db/schema.sql (section 4a). Both
# variants use this list, so the only difference between them is the vector field.
BM25_COLUMNS="
    product_id,
    (search_text::pdb.simple('stemmer=english', 'stopwords_language=english')),
    (title::pdb.simple('stemmer=english', 'stopwords_language=english')),
    brand,
    collaborator,
    model,
    colorway,
    (category::pdb.literal),
    (subcategory::pdb.literal),
    (gender::pdb.literal),
    (clean_tags::pdb.literal),
    price_min,
    price_max,
    on_sale,
    in_stock,
    is_active"

# Runs one statement with \timing on, prints how long it took, records it in $BUILDS.
#   timed_build <service> <variant> <index name> <statement>
timed_build() {
  local svc=$1 variant=$2 index=$3 statement=$4 out ms secs bytes
  echo "   building $index …"
  # ParadeDB prints an ivf_build NOTICE per index segment; warnings and errors still show.
  out=$(printf 'SET client_min_messages = warning;\n\\timing on\n%s\n' "$statement" | psql_in "$svc")
  ms=$(printf '%s\n' "$out" | sed -n 's/^Time: \([0-9.]*\) ms.*/\1/p' | tail -1)
  secs=$(awk -v ms="${ms:-0}" 'BEGIN { printf "%.1f", ms / 1000 }')
  bytes=$(scalar "$svc" "SELECT pg_relation_size('clean.${index}')")
  echo "   $index: ${secs} s, $(awk -v b="$bytes" 'BEGIN { printf "%.1f MB", b / 1048576 }')"

  [ -f "$BUILDS" ] || printf 'built_at\tserver\tpg_search\tvariant\tindex\tseconds\tbytes\n' > "$BUILDS"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$svc" \
    "$(scalar "$svc" "SELECT extversion FROM pg_extension WHERE extname = 'pg_search'")" \
    "$variant" "$index" "$secs" "$bytes" >> "$BUILDS"
}

drop_indexes() {
  psql_in "$1" -q <<'SQL'
SET client_min_messages = warning;
DROP INDEX IF EXISTS clean.products_search_idx;
DROP INDEX IF EXISTS clean.products_embedding_hnsw_idx;
SQL
}

cmd_up() {
  echo "→ starting the bench databases"
  compose up -d --wait
  cmd_status || echo "   (no catalog yet: run scripts/bench_setup.sh restore <dump>)"
}

restore_one() {
  local svc=$1 dump=$2 started
  echo "→ $svc: recreating an empty database"
  # From template0: the ParadeDB image installs its extensions into template1, and the
  # dump creates them itself (see scripts/restore_db.sh).
  in_db "$svc" 'dropdb -U "$POSTGRES_USER" --if-exists --force "$POSTGRES_DB" && createdb -U "$POSTGRES_USER" -T template0 "$POSTGRES_DB"' < /dev/null

  started=$(date +%s)
  if [ "$(head -c 5 "$dump")" = "PGDMP" ]; then
    echo "→ $svc: restoring custom-format dump (indexes are built fresh by this version)"
    in_db "$svc" 'pg_restore -U "$POSTGRES_USER" -d "$POSTGRES_DB" --no-owner --no-privileges --exit-on-error' < "$dump"
  elif gzip -t "$dump" 2>/dev/null; then
    echo "→ $svc: restoring gzipped SQL"
    gunzip -c "$dump" | in_db "$svc" 'psql -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null'
  else
    echo "→ $svc: restoring SQL"
    in_db "$svc" 'psql -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null' < "$dump"
  fi

  # A restore leaves no planner statistics; without them the plans (and timings) are arbitrary.
  echo "→ $svc: vacuum + analyze"
  in_db "$svc" 'vacuumdb -U "$POSTGRES_USER" -d "$POSTGRES_DB" --analyze --quiet' < /dev/null
  echo "   restored in $(( $(date +%s) - started )) s"
}

cmd_restore() {
  local dump=${1:-}
  if [ -z "$dump" ] || [ ! -f "$dump" ]; then
    echo "usage: scripts/bench_setup.sh restore <dump file>   (got: '${dump}')" >&2
    exit 1
  fi
  compose up -d --wait
  restore_one pdb026 "$dump"
  restore_one pdb027 "$dump"
  cmd_status
}

cmd_index() {
  local kind=${1:-} opts=${2:-}
  case "$kind" in
    pgvector)
      echo "→ pdb027: BM25 index + pgvector HNSW (variant B)"
      drop_indexes pdb027
      timed_build pdb027 pgvector-hnsw products_search_idx \
        "CREATE INDEX products_search_idx ON clean.products USING paradedb (${BM25_COLUMNS});"
      # m / ef_construction as in db/schema.sql.
      timed_build pdb027 pgvector-hnsw products_embedding_hnsw_idx \
        "CREATE INDEX products_embedding_hnsw_idx ON clean.products USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);"
      ;;
    paradedb)
      local with="" variant="paradedb-vector"
      if [ -n "$opts" ]; then
        with=" WITH (${opts})"
        variant="paradedb-vector(${opts})"
      fi
      echo "→ pdb027: one ParadeDB index with BM25 fields + embedding (variant C)${opts:+, $opts}"
      drop_indexes pdb027
      # A table can have only one ParadeDB index, so the vector field joins the BM25 one.
      timed_build pdb027 "$variant" products_search_idx \
        "CREATE INDEX products_search_idx ON clean.products USING paradedb (${BM25_COLUMNS},
    embedding vector_cosine_ops)${with};"
      ;;
    *)
      echo "usage: scripts/bench_setup.sh index pgvector | paradedb [\"opts\"]" >&2
      exit 1
      ;;
  esac
  printf 'ANALYZE clean.products;\n' | psql_in pdb027 -q
  cmd_status
}

# Query plans on pdb027 for the three query shapes the benchmark needs. With the native
# index (after `index paradedb`), each should show "Custom Scan (ParadeDB Base Scan)" with
# "Exec Method: TopKScanExecState"; a filter inside "Tantivy Query" ran in the index.
cmd_explain() {
  psql_in pdb027 -q <<'SQL'
SET client_min_messages = warning;
-- A real product's vector and category, so the plans use realistic values.
SELECT embedding AS v, category AS c
  FROM clean.products WHERE embedding IS NOT NULL AND category IS NOT NULL
 ORDER BY product_id LIMIT 1 \gset
\echo
\echo '== 1. unfiltered: pdb.all() + ORDER BY distance + LIMIT'
EXPLAIN (COSTS OFF)
SELECT product_id FROM clean.products
 WHERE product_id @@@ pdb.all()
 ORDER BY embedding <=> :'v'::vector, product_id
 LIMIT 10;
\echo
\echo '== 2. filter with a ParadeDB operator (category === ...)'
EXPLAIN (COSTS OFF)
SELECT product_id FROM clean.products
 WHERE category === :'c'
 ORDER BY embedding <=> :'v'::vector, product_id
 LIMIT 10;
\echo
\echo '== 3. plain SQL filters on indexed columns (what filters.py writes today)'
EXPLAIN (COSTS OFF)
SELECT product_id FROM clean.products
 WHERE product_id @@@ pdb.all()
   AND is_active AND in_stock AND price_min <= 200
 ORDER BY embedding <=> :'v'::vector, product_id
 LIMIT 10;
SQL
}

cmd_status() {
  local svc
  for svc in pdb026 pdb027; do
    echo
    echo "== $svc =="
    psql_in "$svc" -q -tA <<'SQL'
SET paradedb.planner_warnings = 'off';
SELECT 'postgres ' || current_setting('server_version');
SELECT 'extension ' || extname || ' ' || extversion FROM pg_extension
 WHERE extname IN ('pg_search', 'vector') ORDER BY extname;
SELECT 'products ' || count(*) FILTER (WHERE is_active) || ' active, '
       || count(embedding) FILTER (WHERE is_active) || ' with embeddings'
  FROM clean.products;
SELECT 'index ' || c.relname || ' (' || am.amname || ') ' || pg_size_pretty(pg_relation_size(c.oid))
  FROM pg_index i
  JOIN pg_class c ON c.oid = i.indexrelid
  JOIN pg_am am ON am.oid = c.relam
 WHERE i.indrelid = 'clean.products'::regclass
   AND am.amname IN ('paradedb', 'bm25', 'hnsw', 'ivfflat')
 ORDER BY c.relname;
SQL
  done
}

case "${1:-}" in
  up)      cmd_up ;;
  restore) shift; cmd_restore "$@" ;;
  index)   shift; cmd_index "$@" ;;
  status)  cmd_status ;;
  explain) cmd_explain ;;
  down)    compose down ;;
  *)
    sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
