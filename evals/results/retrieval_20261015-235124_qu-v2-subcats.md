# Retrieval eval, 2026-10-15 23:51 (qu-v2-subcats)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on. Query understanding: gpt-6-luna, prompt qu-v2, dense weight specific 0.75 / explore 1.5; subcategory filters on.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| hybrid+qu | 0.381 | 0.886 | 0.759 | 3096 | 5020 |

## nDCG@10 by query type

| Tag | Queries | hybrid+qu |
|---|---|---|
| attribute | 6 | 0.785 |
| brand | 4 | 0.984 |
| collab | 7 | 0.913 |
| exact-name | 8 | 0.942 |
| filtered | 6 | 0.760 |
| kids | 4 | 0.879 |
| model-family | 1 | 0.936 |
| natural-language | 3 | 0.505 |
| no-good-match | 2 | 0.225 |
| style-code | 4 | 0.871 |
| typo | 4 | 0.516 |
| vague-intent | 8 | 0.511 |

## Query understanding

- LLM used on 51/55 queries (skipped for style codes: 4; failed and fell back: 0).
- Out of scope: 0; inferred filters relaxed: 11; catalog values dropped: 2.
- Tokens: 59,938 in / 7,637 out in total (1,175 / 149 per query).
- LLM latency: p50 2846 ms, p95 4634 ms.

## Weakest queries per mode (lowest nDCG@10)

**hybrid+qu**: `alias-chuck-taylor-70s` (0.00), `comfortable-walking-shoes` (0.00), `nl-black-sneakers-daily` (0.00), `nomatch-womens-winter-parka` (0.00), `samba-size-10` (0.00)
