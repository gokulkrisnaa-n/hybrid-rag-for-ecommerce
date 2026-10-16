# Retrieval eval, 2026-10-15 23:37 (round2-qu)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on. Query understanding: gpt-6-luna, prompt qu-v1, dense weight specific 0.75 / explore 1.5.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.367 | 0.808 | 0.696 | 18 | 37 |
| dense | 0.326 | 0.790 | 0.703 | 318 | 493 |
| hybrid | 0.397 | 0.891 | 0.777 | 304 | 465 |
| hybrid+qu | 0.383 | 0.891 | 0.763 | 2810 | 4101 |

## nDCG@10 by query type

| Tag | Queries | bm25 | dense | hybrid | hybrid+qu |
|---|---|---|---|---|---|
| attribute | 6 | 0.827 | 0.734 | **0.833** | 0.724 |
| brand | 4 | 0.854 | **0.989** | 0.911 | 0.984 |
| collab | 7 | **0.933** | 0.843 | 0.883 | 0.913 |
| exact-name | 8 | **0.954** | 0.683 | 0.876 | 0.939 |
| filtered | 6 | 0.849 | 0.816 | 0.870 | **0.926** |
| kids | 4 | 0.769 | **1.000** | 0.976 | 0.657 |
| model-family | 1 | 0.870 | **0.958** | 0.936 | 0.936 |
| natural-language | 3 | 0.484 | 0.791 | **0.814** | 0.505 |
| no-good-match | 2 | 0.000 | 0.370 | 0.370 | 0.322 |
| style-code | 4 | 0.871 | 0.000 | 0.871 | 0.871 |
| typo | 4 | 0.387 | 0.573 | 0.533 | **0.742** |
| vague-intent | 8 | 0.170 | **0.697** | 0.496 | 0.459 |

## Query understanding

- LLM used on 51/55 queries (skipped for style codes: 4; failed and fell back: 0).
- Out of scope: 0; inferred filters relaxed: 9; catalog values dropped: 2.
- Tokens: 58,867 in / 7,385 out in total (1,154 / 144 per query).
- LLM latency: p50 2530 ms, p95 3474 ms.

## Weakest queries per mode (lowest nDCG@10)

**bm25**: `accessories-to-keep-warm` (0.00), `nomatch-tuxedo` (0.00), `nomatch-womens-winter-parka` (0.00), `rain-jacket` (0.00), `typo-nupste-puffer` (0.00)

**dense**: `chuck-70` (0.00), `clot` (0.00), `court-jacket` (0.00), `nomatch-womens-winter-parka` (0.00), `style-code-aaki4323` (0.00)

**hybrid**: `nomatch-womens-winter-parka` (0.00), `typo-nupste-puffer` (0.00), `warm-jacket-for-winter` (0.00), `comfortable-walking-shoes` (0.08), `rain-jacket` (0.21)

**hybrid+qu**: `accessories-to-keep-warm` (0.00), `comfortable-walking-shoes` (0.00), `nl-black-sneakers-daily` (0.00), `nomatch-womens-winter-parka` (0.00), `toddler-samba` (0.00)
