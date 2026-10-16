# Retrieval eval, 2026-10-16 00:09 (qu-v3-dense-original)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on. Query understanding: gpt-6-luna, prompt qu-v2, dense weight specific 0.75 / explore 1.5; subcategory filters off; dense embeds the original text.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| hybrid | 0.397 | 0.891 | 0.777 | 302 | 562 |
| hybrid+qu | 0.404 | 0.926 | 0.805 | 3065 | 4416 |

## nDCG@10 by query type

| Tag | Queries | hybrid | hybrid+qu |
|---|---|---|---|
| attribute | 6 | 0.833 | **0.868** |
| brand | 4 | 0.911 | **0.984** |
| collab | 7 | 0.883 | **0.923** |
| exact-name | 8 | 0.876 | **0.901** |
| filtered | 6 | 0.870 | **0.931** |
| kids | 4 | **0.976** | 0.890 |
| model-family | 1 | 0.936 | 0.936 |
| natural-language | 3 | **0.814** | 0.747 |
| no-good-match | 2 | **0.370** | 0.322 |
| style-code | 4 | 0.871 | 0.871 |
| typo | 4 | 0.533 | **0.672** |
| vague-intent | 8 | 0.496 | **0.522** |

## Query understanding

- LLM used on 51/55 queries (skipped for style codes: 4; failed and fell back: 0).
- Out of scope: 0; inferred filters relaxed: 0; catalog values dropped: 2.
- Tokens: 59,938 in / 7,896 out in total (1,175 / 154 per query).
- LLM latency: p50 2765 ms, p95 4074 ms.

## Weakest queries per mode (lowest nDCG@10)

**hybrid**: `nomatch-womens-winter-parka` (0.00), `typo-nupste-puffer` (0.00), `warm-jacket-for-winter` (0.00), `comfortable-walking-shoes` (0.08), `rain-jacket` (0.21)

**hybrid+qu**: `nomatch-womens-winter-parka` (0.00), `comfortable-walking-shoes` (0.05), `warm-jacket-for-winter` (0.09), `typo-nupste-puffer` (0.20), `accessories-to-keep-warm` (0.34)
