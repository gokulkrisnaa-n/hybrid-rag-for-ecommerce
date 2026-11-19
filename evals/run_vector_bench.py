"""
Vector-index benchmark: pgvector HNSW vs ParadeDB's native vector index (0.27).

    docker compose -f docker-compose.bench.yml up -d --wait     # once
    scripts/bench_setup.sh restore backups/<dump>               # once
    python -m evals.bench.embed_queries --check                 # vectors up to date?
    python -m evals.run_vector_bench                            # about 10 minutes
    python -m evals.run_vector_bench --reps 3 --label smoke     # quick try

No OpenAI calls: every variant searches with the vectors in
evals/bench/query_vectors.jsonl. The runner switches pdb027's indexes itself
(scripts/bench_setup.sh index ...) and leaves it on the native index.

Variants
  A  ParadeDB 0.26.1 + pgvector HNSW          pdb026, what production runs
  B  ParadeDB 0.27.0 + pgvector HNSW          pdb027; plus an hnsw.ef_search sweep
  C  ParadeDB 0.27.0 native vector index      pdb027; plus a vector_cluster_max_probe sweep (D)
  E  C, with BM25 + vector + RRF in ONE SQL statement

Measured
  quality     dense nDCG@10 / recall@10 / MRR for every configuration; hybrid for A, B, C,
              through the app's own hybrid_search with only the vector search swapped
  ANN recall  share of the exact nearest neighbours found (brute force, same filters), @10 and @k
  latency     the vector query alone, client-side, warm, `--reps` passes over all queries.
              Session settings (ef_search, max_probe) are set once, outside the timing, so
              both engines are timed on one SQL statement each
  filtered    the same under typical filter sets: latency, recall, under-filled results
  plans       EXPLAIN evidence; the run stops if ParadeDB isn't doing the Top K scan

Writes evals/results/vector_bench_<timestamp>.md (commit it) and a .csv with one row
per configuration, filter set and query.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from collections.abc import Awaitable, Callable

from psycopg import AsyncConnection, sql
from psycopg.conninfo import make_conninfo
from psycopg_pool import AsyncConnectionPool

from app.core import llm
from app.core.config import get_settings
from app.core.db import create_pool
from app.retrieval import dense as app_dense
from app.retrieval import hybrid as app_hybrid
from app.retrieval.filters import build_where
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.sparse import search_sparse
from app.schemas.query_models import SearchFilters
from evals.bench.embed_queries import StaleVectorsError, load_query_vectors
from evals.bench.paradedb_dense import (
    Plan,
    dense_query,
    explain,
    probe_pushdown,
    search_dense_paradedb,
    vector_literal,
)
from evals.dataset import EvalCase, load_cases
from evals.metrics import mrr_at_k, ndcg_at_k, percentile, recall_at_k

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evals" / "results"
BUILDS_TSV = RESULTS_DIR / "vector_bench_builds.tsv"
PORTS = {"pdb026": 5434, "pdb027": 5435}

EVAL = "eval set filters"      # the filter set each case carries in ground_truth.jsonl

# Filter sets for the filtered-search section: each replaces the case's own filters.
FILTER_SETS: dict[str, SearchFilters] = {
    "in stock (default)": SearchFilters(),
    "under $100": SearchFilters(max_price=Decimal(100)),
    "womens / unisex": SearchFilters(genders=["womens", "unisex"]),
    "footwear": SearchFilters(categories=["footwear"]),
    "brand nike": SearchFilters(brands=["nike"]),
    "size 10 (not indexed)": SearchFilters(sizes=["10"]),
    "footwear, mens, under $150": SearchFilters(categories=["footwear"], genders=["mens"],
                                                max_price=Decimal(150)),
}


# ---------------------------------------------------------------------------
# Engines: one SQL statement per vector search, settings outside the timing
# ---------------------------------------------------------------------------

@dataclass
class Engine:
    label: str                       # e.g. "B  pgvector HNSW, ef_search 100"
    variant: str                     # A, B, C, D
    server: str                      # pdb026 / pdb027
    kind: str                        # pgvector / paradedb
    session: dict[str, str]          # set once per connection for the direct measurements
    default: bool = True             # the configuration production would use

    def query(self, vec: list[float], filters: SearchFilters, k: int) -> tuple[sql.Composable, dict]:
        if self.kind == "paradedb":
            return dense_query(vec, filters, k)
        where = build_where(filters)
        return (app_dense._DENSE_SQL.format(where=where.clause),        # production's own SQL
                {"qvec": app_dense._vector_literal(vec), "k": k, **where.params})

    async def configure(self, conn: AsyncConnection) -> None:
        for name, value in self.session.items():
            await conn.execute("SELECT set_config(%s, %s, false)", (name, value))

    async def run(self, conn: AsyncConnection, vec: list[float], filters: SearchFilters, k: int) -> list[int]:
        query, params = self.query(vec, filters, k)
        cur = await conn.execute(query, params)
        return [r["product_id"] for r in await cur.fetchall()]


def pgvector_engine(variant: str, server: str, ef: int, k: int, default: bool) -> Engine:
    s = get_settings()
    session = {"hnsw.ef_search": str(max(ef, k))}       # as search_dense: never below k
    if s.hnsw_iterative_scan != "off":
        session["hnsw.iterative_scan"] = s.hnsw_iterative_scan
    version = "0.26.1" if server == "pdb026" else "0.27.0"
    return Engine(f"{variant}  {version} pgvector HNSW, ef_search {max(ef, k)}", variant, server,
                  "pgvector", session, default)


def paradedb_engine(max_probe: int | None) -> Engine:
    if max_probe is None:
        return Engine("C  0.27.0 ParadeDB vector, max_probe default (100)", "C", "pdb027", "paradedb", {})
    return Engine(f"D  0.27.0 ParadeDB vector, max_probe {max_probe}", "D", "pdb027", "paradedb",
                  {"paradedb.vector_cluster_max_probe": str(max_probe)}, default=False)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

@dataclass
class EngineResult:
    engine: Engine
    ids: dict[tuple[str, str], list[int]] = field(default_factory=dict)       # (filter set, case) -> ranked ids
    ms: dict[tuple[str, str], list[float]] = field(default_factory=dict)      # (filter set, case) -> timings
    hybrid: dict[str, list[int]] = field(default_factory=dict)                # case -> hybrid top k


def progress(msg: str) -> None:
    print(f"\r  {msg}".ljust(70), end="", file=sys.stderr, flush=True)


async def measure(engine: Engine, pool: AsyncConnectionPool, cases: list[EvalCase],
                  vectors: dict[str, list[float]], kc: int, reps: int) -> EngineResult:
    """Results (first pass, which also warms the caches), then `reps` timed passes."""
    res = EngineResult(engine)
    sets = [(EVAL, None)] + list(FILTER_SETS.items())
    async with pool.connection() as conn:
        await engine.configure(conn)
        try:
            for name, filters in sets:
                for case in cases:
                    res.ids[(name, case.id)] = await engine.run(conn, vectors[case.id], filters or case.filters, kc)
            for rep in range(1, reps + 1):
                progress(f"{engine.label}: pass {rep}/{reps}")
                for name, filters in sets:
                    for case in cases:
                        start = time.perf_counter()
                        await engine.run(conn, vectors[case.id], filters or case.filters, kc)
                        res.ms.setdefault((name, case.id), []).append((time.perf_counter() - start) * 1000)
        finally:
            await conn.execute("RESET ALL")
    return res


async def exact_neighbours(pool: AsyncConnectionPool, cases: list[EvalCase],
                           vectors: dict[str, list[float]], kc: int) -> dict[tuple[str, str], list[int]]:
    """
    The true nearest neighbours: `(distance) + 0` can't use any index, so Postgres
    computes every distance and sorts. Same filters, same tie-break as the engines.
    """
    out = {}
    sets = [(EVAL, None)] + list(FILTER_SETS.items())
    async with pool.connection() as conn:
        for name, filters in sets:
            progress(f"exact neighbours: {name}")
            for case in cases:
                where = build_where(filters or case.filters)
                q = sql.SQL("""
                    SELECT product_id FROM clean.products
                    WHERE embedding IS NOT NULL AND {where}
                    ORDER BY (embedding <=> %(qvec)s::vector) + 0, product_id
                    LIMIT %(k)s""").format(where=where.clause)
                cur = await conn.execute(q, {"qvec": vector_literal(vectors[case.id]), "k": kc, **where.params})
                out[(name, case.id)] = [r["product_id"] for r in await cur.fetchall()]
    return out


DenseFn = Callable[[AsyncConnection, list[float], SearchFilters, int], Awaitable[list]]


async def hybrid_quality(pool: AsyncConnectionPool, cases: list[EvalCase], k: int, dense_fn: DenseFn) -> dict[str, list[int]]:
    """The app's hybrid search (code routing, fuzzy fallback, coverage weighting), vector search swapped."""
    original = app_hybrid.search_dense
    app_hybrid.search_dense = dense_fn
    try:
        out = {}
        for case in cases:
            res = await app_hybrid.hybrid_search(pool, case.to_search_query(limit=k), "hybrid")
            out[case.id] = [r.product.product_id for r in res.results]
        return out
    finally:
        app_hybrid.search_dense = original


# ---------------------------------------------------------------------------
# E: BM25 + vector + RRF in one statement
# ---------------------------------------------------------------------------

# Each branch collects its top k ids into an array; unnest ... WITH ORDINALITY turns
# the array position into the rank. Same scoring and tie-break as app/retrieval/fusion.py:
# sum of weight / (rrf_k + rank), then best single rank, then product id.
_ONE_QUERY_SQL = sql.SQL("""
WITH
bm25 AS (
    SELECT array(
        SELECT product_id FROM clean.products
        WHERE search_text ||| %(q)s AND {where}
        ORDER BY pdb.score(product_id) DESC, product_id
        LIMIT %(k)s) AS ids
),
dense AS (
    SELECT array(
        SELECT product_id FROM clean.products
        WHERE product_id @@@ pdb.all() AND {where}
        ORDER BY embedding <=> %(qvec)s::vector, product_id
        LIMIT %(k)s) AS ids
),
ranked AS (
    SELECT u.id, u.r FROM bm25, unnest(bm25.ids) WITH ORDINALITY AS u(id, r)
    UNION ALL
    SELECT u.id, u.r FROM dense, unnest(dense.ids) WITH ORDINALITY AS u(id, r)
)
SELECT id AS product_id, sum(1.0::float8 / (%(rrf_k)s::int + r)) AS score, min(r) AS best
FROM ranked
GROUP BY id
ORDER BY score DESC, best, product_id
LIMIT %(limit)s
""")


def one_query(text: str, vec: list[float], filters: SearchFilters, kc: int, limit: int) -> tuple[sql.Composable, dict]:
    where = build_where(filters)
    return (_ONE_QUERY_SQL.format(where=where.clause),
            {"q": text, "qvec": vector_literal(vec), "k": kc, "limit": limit,
             "rrf_k": get_settings().rrf_k, **where.params})


@dataclass
class OneQueryResult:
    ids: dict[str, list[int]] = field(default_factory=dict)          # case -> fused top k (one statement)
    two_ids: dict[str, list[int]] = field(default_factory=dict)      # case -> same fusion, two queries + Python
    one_ms: dict[str, list[float]] = field(default_factory=dict)
    two_ms: dict[str, list[float]] = field(default_factory=dict)
    plan: Plan | None = None


async def bench_one_query(pool: AsyncConnectionPool, cases: list[EvalCase], vectors: dict[str, list[float]],
                          kc: int, k: int, reps: int) -> OneQueryResult:
    """
    The single statement against the way the app does it: BM25 and vector search on two
    pooled connections at the same time, fused in Python. Plain RRF in both (weights 1,
    no fuzzy retry or coverage weighting), so the two must return the same products.
    """
    s = get_settings()
    out = OneQueryResult()

    async def two_queries(case: EvalCase) -> list[int]:
        async def sparse() -> list[int]:
            async with pool.connection() as c:
                return [h.product_id for h in await search_sparse(c, case.query, case.filters, kc)]

        async def vector() -> list[int]:
            async with pool.connection() as c:
                return [h.product_id for h in await search_dense_paradedb(c, vectors[case.id], case.filters, kc)]

        bm25, dense = await asyncio.gather(sparse(), vector())
        fused = reciprocal_rank_fusion({"bm25": bm25, "dense": dense}, k=s.rrf_k, limit=k)
        return [f.product_id for f in fused]

    async def single(conn: AsyncConnection, case: EvalCase) -> list[int]:
        query, params = one_query(case.query, vectors[case.id], case.filters, kc, k)
        cur = await conn.execute(query, params)
        return [r["product_id"] for r in await cur.fetchall()]

    async with pool.connection() as conn:
        first = cases[0]
        query, params = one_query(first.query, vectors[first.id], first.filters, kc, k)
        out.plan = await explain(conn, query, params)
        for case in cases:
            out.ids[case.id] = await single(conn, case)
            out.two_ids[case.id] = await two_queries(case)
        for rep in range(1, reps + 1):
            progress(f"E  one statement vs two queries: pass {rep}/{reps}")
            for case in cases:
                start = time.perf_counter()
                await single(conn, case)
                out.one_ms.setdefault(case.id, []).append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                await two_queries(case)
                out.two_ms.setdefault(case.id, []).append((time.perf_counter() - start) * 1000)
    return out


# ---------------------------------------------------------------------------
# Setup and server facts
# ---------------------------------------------------------------------------

def switch_index(kind: str) -> None:
    print(f"\n→ pdb027: scripts/bench_setup.sh index {kind}", file=sys.stderr)
    done = subprocess.run(["scripts/bench_setup.sh", "index", kind], cwd=ROOT, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        sys.exit(f"bench_setup.sh index {kind} failed:\n{done.stdout}\n{done.stderr}")


async def server_facts(pool: AsyncConnectionPool) -> dict:
    async with pool.connection() as conn:
        cur = await conn.execute("""
            SELECT current_setting('server_version') AS postgres,
                   (SELECT extversion FROM pg_extension WHERE extname = 'pg_search') AS pg_search,
                   (SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector""")
        facts = dict(await cur.fetchone())
        cur = await conn.execute("""
            SELECT c.relname AS index, am.amname AS method, pg_relation_size(c.oid) AS bytes
            FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid JOIN pg_am am ON am.oid = c.relam
            WHERE i.indrelid = 'clean.products'::regclass AND am.amname IN ('paradedb', 'bm25', 'hnsw')
            ORDER BY c.relname""")
        facts["indexes"] = [dict(r) for r in await cur.fetchall()]
    return facts


async def selectivity(pool: AsyncConnectionPool) -> dict[str, int]:
    """How many products each filter set lets through."""
    out = {}
    async with pool.connection() as conn:
        for name, filters in FILTER_SETS.items():
            where = build_where(filters)
            cur = await conn.execute(sql.SQL("SELECT count(*) AS n FROM clean.products WHERE {}").format(where.clause),
                                     where.params)
            out[name] = (await cur.fetchone())["n"]
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def mean(xs: list[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def overlap(found: list[int], exact: list[int], n: int) -> float:
    truth = exact[:n]
    return len(set(found[:n]) & set(truth)) / len(truth) if truth else 1.0


def report(args, cases: list[EvalCase], facts: dict, sel: dict[str, int], results: list[EngineResult],
           exact: dict, pushdown, one: OneQueryResult, plans: dict[str, Plan]) -> str:
    k, kc = args.k, args.candidates
    s = get_settings()
    total = facts["pdb026"]["active"]
    lines = [
        f"# Vector index benchmark, {datetime.now():%Y-%m-%d %H:%M}" + (f" ({args.label})" if args.label else ""),
        "",
        "pgvector HNSW against ParadeDB's native vector index, on the Kith catalog "
        f"({total:,} products, {s.embedding_model} at {s.embedding_dimensions} dims) and the "
        f"{len(cases)} labelled queries in evals/ground_truth.jsonl.",
        "",
        f"- Servers: pdb026 = Postgres {facts['pdb026']['postgres']}, pg_search {facts['pdb026']['pg_search']}, "
        f"pgvector {facts['pdb026']['pgvector']}; pdb027 = Postgres {facts['pdb027']['postgres']}, "
        f"pg_search {facts['pdb027']['pg_search']}, pgvector {facts['pdb027']['pgvector']}. "
        "Docker, shared_buffers 512MB, same settings on both.",
        f"- Machine: {platform.platform()}, {os.cpu_count()} CPUs (Docker Desktop's VM limits apply).",
        f"- Each search asks for {kc} candidates (what hybrid search fuses); quality is scored @{k}.",
        f"- Latency: client-side, one connection, warm, {args.reps} passes over every query; "
        "one SQL statement per search, session settings set beforehand.",
        "- Query vectors are precomputed (evals/bench/query_vectors.jsonl): no OpenAI calls, identical input.",
        "",
        "## Quality on the labelled queries",
        "",
        f"| Configuration | Dense nDCG@{k} | Dense recall@{k} | Dense MRR | Hybrid nDCG@{k} | Hybrid recall@{k} | Hybrid MRR |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ranked = {c.id: r.ids[(EVAL, c.id)][:k] for c in cases}
        dn = [ndcg_at_k(ranked[c.id], c.relevant, k) for c in cases]
        dr = [recall_at_k(ranked[c.id], c.relevant, k) for c in cases]
        dm = [mrr_at_k(ranked[c.id], c.relevant, k) for c in cases]
        if r.hybrid:
            hn = f"{mean([ndcg_at_k(r.hybrid[c.id], c.relevant, k) for c in cases]):.3f}"
            hr = f"{mean([recall_at_k(r.hybrid[c.id], c.relevant, k) for c in cases]):.3f}"
            hm = f"{mean([mrr_at_k(r.hybrid[c.id], c.relevant, k) for c in cases]):.3f}"
        else:
            hn = hr = hm = "-"
        lines.append(f"| {r.engine.label} | {mean(dn):.3f} | {mean(dr):.3f} | {mean(dm):.3f} | {hn} | {hr} | {hm} |")
    lines += ["", "Hybrid = the app's full hybrid search (code routing, fuzzy retry, coverage-weighted RRF) "
              "with only the vector search swapped; run for the default configuration of each variant.", ""]

    lines += [
        "## Approximate vs exact nearest neighbours, and latency",
        "",
        f"Recall vs exact = share of the true nearest neighbours (brute-force search, same filters) the index "
        f"returned. Latency is the vector query alone, over {len(cases)} queries x {args.reps} passes, "
        "with each case's own filters.",
        "",
        f"| Configuration | Recall@{k} vs exact | Recall@{kc} vs exact | p50 ms | p95 ms | mean ms |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        r10 = [overlap(r.ids[(EVAL, c.id)], exact[(EVAL, c.id)], k) for c in cases]
        rk = [overlap(r.ids[(EVAL, c.id)], exact[(EVAL, c.id)], kc) for c in cases]
        lat = [t for c in cases for t in r.ms[(EVAL, c.id)]]
        name = f"**{r.engine.label}**" if r.engine.default else r.engine.label
        lines.append(f"| {name} | {mean(r10):.3f} | {mean(rk):.3f} | {percentile(lat, 50):.2f} | "
                     f"{percentile(lat, 95):.2f} | {mean(lat):.2f} |")
    lines += ["", "Bold = the configuration production uses (pgvector) or ParadeDB's default.", ""]

    defaults = [r for r in results if r.engine.default]
    lines += [
        "## Filtered vector search",
        "",
        "The same queries under typical filter sets (replacing each case's own filters). "
        f"Under-filled = searches that returned fewer of the {kc} candidates than exist under the filter: "
        "the sign of a filter applied after the index picked its top k.",
        "",
        "| Filter set | Products | Configuration | Recall@" f"{k} vs exact | Under-filled | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in FILTER_SETS:
        for r in defaults:
            r10 = [overlap(r.ids[(name, c.id)], exact[(name, c.id)], k) for c in cases]
            under = sum(len(r.ids[(name, c.id)]) < len(exact[(name, c.id)]) for c in cases)
            lat = [t for c in cases for t in r.ms[(name, c.id)]]
            lines.append(f"| {name} | {sel[name]:,} ({sel[name] / total:.0%}) | {r.engine.label} | {mean(r10):.3f} | "
                         f"{under}/{len(cases)} | {percentile(lat, 50):.2f} | {percentile(lat, 95):.2f} |")
    lines.append("")

    lines += [
        "## Which filters run inside the ParadeDB index",
        "",
        "EXPLAIN of the ParadeDB vector search with each filter alone, written by the app's "
        "filters.build_where exactly as search writes it.",
        "",
        "| Filter | Plan | Postgres filter after the scan |",
        "|---|---|---|",
    ]
    for p in pushdown:
        post = f"`{p.plan.post_filter}`" if p.plan.post_filter else "-"
        lines.append(f"| {p.filter} | {p.plan.verdict} | {post} |")
    lines.append("")

    same = sum(one.ids[c.id] == one.two_ids[c.id] for c in cases)
    one_lat = [t for c in cases for t in one.one_ms[c.id]]
    two_lat = [t for c in cases for t in one.two_ms[c.id]]
    c_hybrid = next(r for r in results if r.engine.variant == "C")
    lines += [
        "## Hybrid in one SQL statement (E)",
        "",
        "BM25 (`|||`, pdb.score) and the vector search (`<=>`) as two CTEs over the same ParadeDB index, "
        "fused with RRF in SQL, against the app's way: the two searches on two connections at the same "
        f"time, fused in Python. Plain RRF in both (weights 1, k = {s.rrf_k}, {kc} candidates per branch), "
        "same filters in both branches.",
        "",
        f"- Same top {k} from both: **{same}/{len(cases)}** queries.",
        f"- One statement: p50 {percentile(one_lat, 50):.2f} ms, p95 {percentile(one_lat, 95):.2f} ms. "
        f"Two queries + Python: p50 {percentile(two_lat, 50):.2f} ms, p95 {percentile(two_lat, 95):.2f} ms.",
        f"- nDCG@{k}: one statement {mean([ndcg_at_k(one.ids[c.id], c.relevant, k) for c in cases]):.3f}; "
        f"the app's hybrid on C {mean([ndcg_at_k(c_hybrid.hybrid[c.id], c.relevant, k) for c in cases]):.3f} "
        "(its extra rules: style codes to BM25 alone, fuzzy retry, BM25 weight by query coverage).",
        f"- Vector branch in the index's Top K scan: {'yes' if one.plan and one.plan.top_k else '**no**'}.",
        "",
    ]

    lines += ["## Index builds (pdb027)", "", "| Variant | Index | Seconds | Size |", "|---|---|---|---|"]
    for row in latest_builds():
        lines.append(f"| {row['variant']} | {row['index']} | {row['seconds']} | {int(row['bytes']) / 1048576:.1f} MB |")
    lines += ["", "On pdb026 the indexes were built by the restore: "
              + ", ".join(f"{i['index']} ({i['method']}) {i['bytes'] / 1048576:.1f} MB" for i in facts["pdb026"]["indexes"])
              + ".", ""]

    lines += ["## Plans", ""]
    for name, plan in plans.items():
        lines += [f"**{name}**", "", "```", plan.text(), "```", ""]
    if one.plan:
        lines += ["**E, one statement**", "", "```", one.plan.text(), "```", ""]
    return "\n".join(lines).rstrip() + "\n"


def latest_builds() -> list[dict]:
    """The most recent build of each (variant, index) on pdb027."""
    if not BUILDS_TSV.exists():
        return []
    with BUILDS_TSV.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f, delimiter="\t") if r["server"] == "pdb027"]
    latest = {(r["variant"], r["index"]): r for r in rows}
    return sorted(latest.values(), key=lambda r: (r["variant"], r["index"]))


def write_csv(path: Path, cases: list[EvalCase], results: list[EngineResult], exact: dict, k: int, kc: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "filter_set", "case_id", f"recall@{k}_vs_exact", f"recall@{kc}_vs_exact",
                    "returned", "exact_available", f"ndcg@{k}", "p50_ms", "p95_ms"])
        for r in results:
            for name in [EVAL, *FILTER_SETS]:
                for c in cases:
                    ids, truth, lat = r.ids[(name, c.id)], exact[(name, c.id)], r.ms[(name, c.id)]
                    w.writerow([r.engine.label, name, c.id, f"{overlap(ids, truth, k):.3f}",
                                f"{overlap(ids, truth, kc):.3f}", len(ids), len(truth),
                                f"{ndcg_at_k(ids[:k], c.relevant, k):.4f}" if name == EVAL else "",
                                f"{percentile(lat, 50):.3f}", f"{percentile(lat, 95):.3f}"])


# ---------------------------------------------------------------------------

async def main() -> None:
    s = get_settings()
    ap = argparse.ArgumentParser(description="Benchmark pgvector HNSW against ParadeDB's native vector index.")
    ap.add_argument("--reps", type=int, default=10, help="timed passes over all queries (default 10)")
    ap.add_argument("--k", type=int, default=10, help="cutoff for quality metrics (default 10)")
    ap.add_argument("--candidates", type=int, default=s.retrieval_candidates,
                    help=f"results per vector search (default {s.retrieval_candidates}, as hybrid search)")
    ap.add_argument("--ef", default="50,100,200,400", help="hnsw.ef_search values for B")
    ap.add_argument("--probes", default="10,25,400", help="vector_cluster_max_probe values for D (default is 100)")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    s.tracing_enabled = False                 # 55 x 3 hybrid searches don't belong in Langfuse
    cases = load_cases()
    try:
        vectors = load_query_vectors(cases)
    except StaleVectorsError as e:
        sys.exit(str(e))

    by_text = {c.query: vectors[c.id] for c in cases}

    async def embed_query(text: str) -> list[float]:
        try:
            return by_text[text]
        except KeyError:
            raise RuntimeError(f"no precomputed vector for {text!r}") from None   # never call OpenAI here
    llm.embed_query = embed_query             # hybrid_search embeds through this

    kc, k = args.candidates, args.k
    efs = [int(x) for x in args.ef.split(",") if x]
    probes = [int(x) for x in args.probes.split(",") if x]
    default_ef = s.hnsw_ef_search

    urls = {name: make_conninfo(s.database_url, host="localhost", port=str(port)) for name, port in PORTS.items()}
    async with create_pool(urls["pdb026"], max_size=4) as p026, create_pool(urls["pdb027"], max_size=4) as p027:
        pools = {"pdb026": p026, "pdb027": p027}
        switch_index("pgvector")
        facts = {name: await server_facts(pool) for name, pool in pools.items()}
        async with p026.connection() as conn:
            cur = await conn.execute("SELECT count(*) AS n FROM clean.products WHERE is_active")
            facts["pdb026"]["active"] = (await cur.fetchone())["n"]
        if not facts["pdb027"]["pg_search"].startswith("0.27"):
            sys.exit(f"pdb027 runs pg_search {facts['pdb027']['pg_search']}, expected 0.27.x")

        exact = await exact_neighbours(p026, cases, vectors, kc)
        sel = await selectivity(p026)
        results: list[EngineResult] = []
        plans: dict[str, Plan] = {}

        # A and B: pgvector HNSW on both versions, production's SQL and settings.
        engines = [pgvector_engine("A", "pdb026", default_ef, kc, True)]
        engines += [pgvector_engine("B", "pdb027", ef, kc, ef == default_ef) for ef in sorted(set(efs) | {default_ef})]
        for e in engines:
            r = await measure(e, pools[e.server], cases, vectors, kc, args.reps)
            if e.default:
                r.hybrid = await hybrid_quality(pools[e.server], cases, k, app_dense.search_dense)
            results.append(r)

        # C and D: the native vector index.
        switch_index("paradedb")
        async with p027.connection() as conn:
            first = cases[0]
            query, params = dense_query(vectors[first.id], first.filters, kc)
            plans["C, vector search"] = plan = await explain(conn, query, params)
            if not plan.top_k:
                sys.exit("ParadeDB isn't running the vector search as a Top K scan:\n" + plan.text())
            pushdown = await probe_pushdown(conn, vectors[first.id], kc)

        for e in [paradedb_engine(None)] + [paradedb_engine(p) for p in probes]:
            r = await measure(e, p027, cases, vectors, kc, args.reps)
            if e.default:
                r.hybrid = await hybrid_quality(p027, cases, k, search_dense_paradedb)
            results.append(r)

        # E: hybrid in one statement.
        one = await bench_one_query(p027, cases, vectors, kc, k, args.reps)
        facts["pdb027"] = {**await server_facts(p027), "active": facts["pdb026"]["active"]}
    print(file=sys.stderr)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    suffix = f"_{args.label}" if args.label else ""
    md = RESULTS_DIR / f"vector_bench_{stamp}{suffix}.md"
    md.write_text(report(args, cases, facts, sel, results, exact, pushdown, one, plans), encoding="utf-8")
    write_csv(md.with_suffix(".csv"), cases, results, exact, k, kc)
    print(md.read_text(encoding="utf-8"))
    print(f"wrote {md.relative_to(ROOT)} and {md.with_suffix('.csv').name}", file=sys.stderr)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
