# Vector index benchmark, 2026-11-19 15:43 (smoke)

pgvector HNSW against ParadeDB's native vector index, on the Kith catalog (25,000 products, text-embedding-3-small at 768 dims) and the 55 labelled queries in evals/ground_truth.jsonl.

- Servers: pdb026 = Postgres 18.6 (Debian 18.6-1.pgdg13+2), pg_search 0.26.1, pgvector 0.8.6; pdb027 = Postgres 18.7 (Debian 18.7-1.pgdg13+2), pg_search 0.27.0, pgvector 0.8.6. Docker, shared_buffers 512MB, same settings on both.
- Machine: macOS-27.2.1-arm64-arm-64bit-Mach-O, 12 CPUs (Docker Desktop's VM limits apply).
- Each search asks for 50 candidates (what hybrid search fuses); quality is scored @10.
- Latency: client-side, one connection, warm, 3 passes over every query; one SQL statement per search, session settings set beforehand.
- Query vectors are precomputed (evals/bench/query_vectors.jsonl): no OpenAI calls, identical input.

## Quality on the labelled queries

| Configuration | Dense nDCG@10 | Dense recall@10 | Dense MRR | Hybrid nDCG@10 | Hybrid recall@10 | Hybrid MRR |
|---|---|---|---|---|---|---|
| A  0.26.1 pgvector HNSW, ef_search 100 | 0.679 | 0.309 | 0.772 | 0.756 | 0.381 | 0.868 |
| B  0.27.0 pgvector HNSW, ef_search 50 | 0.696 | 0.318 | 0.790 | - | - | - |
| B  0.27.0 pgvector HNSW, ef_search 100 | 0.697 | 0.318 | 0.790 | 0.764 | 0.386 | 0.869 |
| B  0.27.0 pgvector HNSW, ef_search 200 | 0.696 | 0.318 | 0.790 | - | - | - |
| B  0.27.0 pgvector HNSW, ef_search 400 | 0.722 | 0.346 | 0.816 | - | - | - |
| C  0.27.0 ParadeDB vector, max_probe default (100) | 0.722 | 0.346 | 0.816 | 0.778 | 0.403 | 0.894 |
| D  0.27.0 ParadeDB vector, max_probe 10 | 0.696 | 0.324 | 0.798 | - | - | - |
| D  0.27.0 ParadeDB vector, max_probe 25 | 0.703 | 0.326 | 0.798 | - | - | - |
| D  0.27.0 ParadeDB vector, max_probe 400 | 0.722 | 0.346 | 0.816 | - | - | - |

Hybrid = the app's full hybrid search (code routing, fuzzy retry, coverage-weighted RRF) with only the vector search swapped; run for the default configuration of each variant.

## Approximate vs exact nearest neighbours, and latency

Recall vs exact = share of the true nearest neighbours (brute-force search, same filters) the index returned. Latency is the vector query alone, over 55 queries x 3 passes, with each case's own filters.

| Configuration | Recall@10 vs exact | Recall@50 vs exact | p50 ms | p95 ms | mean ms |
|---|---|---|---|---|---|
| **A  0.26.1 pgvector HNSW, ef_search 100** | 0.916 | 0.894 | 2.51 | 8.49 | 3.25 |
| B  0.27.0 pgvector HNSW, ef_search 50 | 0.895 | 0.865 | 2.43 | 8.86 | 3.23 |
| **B  0.27.0 pgvector HNSW, ef_search 100** | 0.927 | 0.901 | 2.71 | 9.26 | 3.40 |
| B  0.27.0 pgvector HNSW, ef_search 200 | 0.945 | 0.932 | 3.35 | 12.70 | 4.62 |
| B  0.27.0 pgvector HNSW, ef_search 400 | 1.000 | 1.000 | 20.07 | 34.54 | 20.21 |
| **C  0.27.0 ParadeDB vector, max_probe default (100)** | 1.000 | 1.000 | 3.41 | 4.49 | 3.52 |
| D  0.27.0 ParadeDB vector, max_probe 10 | 0.907 | 0.889 | 2.60 | 4.12 | 2.75 |
| D  0.27.0 ParadeDB vector, max_probe 25 | 0.973 | 0.970 | 2.80 | 4.48 | 2.99 |
| D  0.27.0 ParadeDB vector, max_probe 400 | 1.000 | 1.000 | 3.70 | 4.92 | 3.84 |

Bold = the configuration production uses (pgvector) or ParadeDB's default.

## Filtered vector search

The same queries under typical filter sets (replacing each case's own filters). Under-filled = searches that returned fewer of the 50 candidates than exist under the filter: the sign of a filter applied after the index picked its top k.

| Filter set | Products | Configuration | Recall@10 vs exact | Under-filled | p50 ms | p95 ms |
|---|---|---|---|---|---|---|
| in stock (default) | 9,694 (39%) | A  0.26.1 pgvector HNSW, ef_search 100 | 0.915 | 0/55 | 2.49 | 3.54 |
| in stock (default) | 9,694 (39%) | B  0.27.0 pgvector HNSW, ef_search 100 | 0.933 | 0/55 | 2.69 | 3.99 |
| in stock (default) | 9,694 (39%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 1.000 | 0/55 | 3.48 | 4.61 |
| under $100 | 2,627 (11%) | A  0.26.1 pgvector HNSW, ef_search 100 | 0.965 | 0/55 | 5.39 | 11.34 |
| under $100 | 2,627 (11%) | B  0.27.0 pgvector HNSW, ef_search 100 | 0.976 | 0/55 | 5.56 | 10.77 |
| under $100 | 2,627 (11%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 0.995 | 0/55 | 3.06 | 4.01 |
| womens / unisex | 3,634 (15%) | A  0.26.1 pgvector HNSW, ef_search 100 | 0.982 | 0/55 | 5.18 | 8.43 |
| womens / unisex | 3,634 (15%) | B  0.27.0 pgvector HNSW, ef_search 100 | 0.980 | 0/55 | 5.08 | 8.61 |
| womens / unisex | 3,634 (15%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 0.998 | 0/55 | 3.01 | 4.27 |
| footwear | 2,010 (8%) | A  0.26.1 pgvector HNSW, ef_search 100 | 0.949 | 0/55 | 7.23 | 25.94 |
| footwear | 2,010 (8%) | B  0.27.0 pgvector HNSW, ef_search 100 | 0.969 | 0/55 | 6.98 | 24.92 |
| footwear | 2,010 (8%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 1.000 | 0/55 | 3.05 | 4.13 |
| brand nike | 257 (1%) | A  0.26.1 pgvector HNSW, ef_search 100 | 1.000 | 0/55 | 4.99 | 5.99 |
| brand nike | 257 (1%) | B  0.27.0 pgvector HNSW, ef_search 100 | 1.000 | 0/55 | 5.19 | 6.51 |
| brand nike | 257 (1%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 1.000 | 0/55 | 20.60 | 23.92 |
| size 10 (not indexed) | 885 (4%) | A  0.26.1 pgvector HNSW, ef_search 100 | 1.000 | 0/55 | 7.64 | 9.16 |
| size 10 (not indexed) | 885 (4%) | B  0.27.0 pgvector HNSW, ef_search 100 | 1.000 | 0/55 | 7.34 | 8.85 |
| size 10 (not indexed) | 885 (4%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 1.000 | 0/55 | 5.90 | 7.18 |
| footwear, mens, under $150 | 375 (2%) | A  0.26.1 pgvector HNSW, ef_search 100 | 1.000 | 0/55 | 2.25 | 3.79 |
| footwear, mens, under $150 | 375 (2%) | B  0.27.0 pgvector HNSW, ef_search 100 | 1.000 | 0/55 | 2.67 | 3.83 |
| footwear, mens, under $150 | 375 (2%) | C  0.27.0 ParadeDB vector, max_probe default (100) | 1.000 | 0/55 | 2.72 | 3.88 |

## Which filters run inside the ParadeDB index

EXPLAIN of the ParadeDB vector search with each filter alone, written by the app's filters.build_where exactly as search writes it.

| Filter | Plan | Postgres filter after the scan |
|---|---|---|
| is_active, in_stock (default) | in the index | - |
| sold out included | in the index | - |
| on_sale | in the index | - |
| price_min <= x (max price) | in the index | - |
| price_max >= x (min price) | in the index | - |
| gender = ANY(...) | in the index | - |
| NOT gender = ANY(...) (exclude) | in the index | - |
| category = ANY(...) | in the index | - |
| lower(brand) = ANY(...) OR collaborator | in the index | - |
| NOT lower(coalesce(brand)) ... (exclude) | in the index | - |
| sizes_available && ... (not indexed) | in the index | - |
| subcategory = ANY(...) | in the index | - |

## Hybrid in one SQL statement (E)

BM25 (`|||`, pdb.score) and the vector search (`<=>`) as two CTEs over the same ParadeDB index, fused with RRF in SQL, against the app's way: the two searches on two connections at the same time, fused in Python. Plain RRF in both (weights 1, k = 60, 50 candidates per branch), same filters in both branches.

- Same top 10 from both: **55/55** queries.
- One statement: p50 9.05 ms, p95 10.21 ms. Two queries + Python: p50 6.76 ms, p95 7.58 ms.
- nDCG@10: one statement 0.736; the app's hybrid on C 0.778 (its extra rules: style codes to BM25 alone, fuzzy retry, BM25 weight by query coverage).
- Vector branch in the index's Top K scan: yes.

## Index builds (pdb027)

| Variant | Index | Seconds | Size |
|---|---|---|---|
| paradedb-vector | products_search_idx | 1.1 | 93.2 MB |
| pgvector-hnsw | products_embedding_hnsw_idx | 1.2 | 97.7 MB |
| pgvector-hnsw | products_search_idx | 0.1 | 9.2 MB |

On pdb026 the indexes were built by the restore: products_embedding_hnsw_idx (hnsw) 97.7 MB, products_search_idx (paradedb) 13.9 MB.

## Plans

**C, vector search**

```
Limit
  ->  Custom Scan (ParadeDB Base Scan) on products
        Table: products
        Index: products_search_idx
        Exec Method: TopKScanExecState
        Scores: false
           TopK Order By: embedding <=> vector asc, product_id asc
           TopK Limit: 50
        Tantivy Query: {"boolean":{"must":[{"with_index":{"query":{"all":{"field":"product_id"}}}},{"term":{"field":"is_active","value":true}},{"term":{"field":"in_stock","value":true}}]}}
```

**E, one statement**

```
Limit
  ->  Sort
        Sort Key: (sum(('1'::double precision / ((60 + u.r))::double precision))) DESC, (min(u.r)), u.id
        ->  GroupAggregate
              Group Key: u.id
              ->  Sort
                    Sort Key: u.id
                    ->  Append
                          ->  Function Scan on unnest u
                                InitPlan 1
                                  ->  Limit
                                        ->  Gather Merge
                                              Workers Planned: 5
                                              ->  Parallel Custom Scan (ParadeDB Base Scan) on products
                                                    Table: products
                                                    Index: products_search_idx
                                                    Exec Method: TopKScanExecState
                                                    Scores: true
                                                       TopK Order By: pdb.score() desc, product_id asc
                                                       TopK Limit: 50
                                                    Tantivy Query: {"boolean":{"must":[{"with_index":{"query":{"match":{"field":"search_text","value":"samba espadrille","tokenizer":null,"distance":null,"transposition_cost_one":null,"prefix":null,"conjunction_mode":false}}}},{"term":{"field":"is_active","value":true}},{"term":{"field":"in_stock","value":true}}]}}
                          ->  Function Scan on unnest u_1
                                InitPlan 2
                                  ->  Limit
                                        ->  Custom Scan (ParadeDB Base Scan) on products products_1
                                              Table: products
                                              Index: products_search_idx
                                              Exec Method: TopKScanExecState
                                              Scores: false
                                                 TopK Order By: embedding <=> vector asc, product_id asc
                                                 TopK Limit: 50
                                              Tantivy Query: {"boolean":{"must":[{"with_index":{"query":{"all":{"field":"product_id"}}}},{"term":{"field":"is_active","value":true}},{"term":{"field":"in_stock","value":true}}]}}
```
