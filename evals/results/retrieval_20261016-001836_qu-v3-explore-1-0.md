# Retrieval eval, 2026-10-16 00:18 (qu-v3-explore-1.0)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on. Query understanding: gpt-6-luna, prompt qu-v2, dense weight specific 0.75 / explore 1; subcategory filters off; dense embeds the original text.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| hybrid+qu | 0.419 | 0.932 | 0.814 | 2763 | 4193 |

## nDCG@10 by query type

| Tag | Queries | hybrid+qu |
|---|---|---|
| attribute | 6 | 0.825 |
| brand | 4 | 0.984 |
| collab | 7 | 0.923 |
| exact-name | 8 | 0.941 |
| filtered | 6 | 0.941 |
| kids | 4 | 0.901 |
| model-family | 1 | 0.936 |
| natural-language | 3 | 0.742 |
| no-good-match | 2 | 0.390 |
| style-code | 4 | 0.871 |
| typo | 4 | 0.822 |
| vague-intent | 8 | 0.469 |

## Query understanding

- LLM used on 51/55 queries (skipped for style codes: 4; failed and fell back: 0).
- Out of scope: 0; inferred filters relaxed: 0; catalog values dropped: 2.
- Tokens: 59,938 in / 7,979 out in total (1,175 / 156 per query).
- LLM latency: p50 2454 ms, p95 3893 ms.

## Weakest queries per mode (lowest nDCG@10)

**hybrid+qu**: `nomatch-womens-winter-parka` (0.00), `comfortable-walking-shoes` (0.05), `warm-jacket-for-winter` (0.14), `nomatch-hiking-boots` (0.35), `rain-jacket` (0.36)
