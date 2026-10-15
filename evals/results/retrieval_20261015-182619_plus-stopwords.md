# Retrieval eval, 2026-10-15 18:26 (plus-stopwords)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| bm25 | 0.366 | 0.809 | 0.695 | 20 | 27 |
| dense | 0.326 | 0.790 | 0.703 | 309 | 510 |
| hybrid | 0.397 | 0.891 | 0.777 | 307 | 504 |

## nDCG@10 by query type

| Tag | Queries | bm25 | dense | hybrid |
|---|---|---|---|---|
| attribute | 5 | 0.939 | 0.783 | **0.943** |
| brand | 4 | 0.854 | **0.989** | 0.911 |
| collab | 7 | **0.933** | 0.843 | 0.883 |
| exact-name | 8 | **0.954** | 0.683 | 0.876 |
| filtered | 6 | 0.849 | 0.816 | **0.870** |
| kids | 4 | 0.769 | **1.000** | 0.976 |
| model-family | 1 | 0.870 | **0.958** | 0.936 |
| natural-language | 3 | 0.466 | 0.791 | **0.807** |
| no-good-match | 3 | 0.088 | **0.410** | 0.341 |
| style-code | 4 | 0.871 | 0.000 | 0.871 |
| typo | 4 | 0.387 | **0.573** | 0.533 |
| vague-intent | 8 | 0.170 | **0.697** | 0.496 |

## Weakest queries per mode (lowest nDCG@10)

**bm25**: `accessories-to-keep-warm` (0.00), `nomatch-tuxedo` (0.00), `nomatch-womens-winter-parka` (0.00), `rain-jacket` (0.00), `typo-nupste-puffer` (0.00)

**dense**: `chuck-70` (0.00), `clot` (0.00), `court-jacket` (0.00), `nomatch-womens-winter-parka` (0.00), `style-code-aaki4323` (0.00)

**hybrid**: `nomatch-womens-winter-parka` (0.00), `typo-nupste-puffer` (0.00), `warm-jacket-for-winter` (0.00), `comfortable-walking-shoes` (0.08), `rain-jacket` (0.21)
