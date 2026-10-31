# Retrieval eval, 2026-10-31 12:56 (qu-none)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on. Query understanding: gpt-6-luna, prompt qu-v6, dense weight specific 0.75 / explore 1.5; subcategory filters off; dense embeds the original text; reasoning effort none.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| hybrid+qu | 0.409 | 0.938 | 0.814 | 1542 | 2051 |

## nDCG@10 by query type

| Tag | Queries | hybrid+qu |
|---|---|---|
| attribute | 6 | 0.861 |
| brand | 4 | 0.984 |
| collab | 7 | 0.923 |
| exact-name | 8 | 0.911 |
| filtered | 6 | 0.900 |
| kids | 4 | 0.874 |
| model-family | 1 | 0.936 |
| natural-language | 3 | 0.743 |
| no-good-match | 2 | 0.178 |
| style-code | 4 | 0.871 |
| typo | 4 | 0.822 |
| vague-intent | 8 | 0.565 |

## Query understanding

- LLM used on 51/55 queries (skipped for style codes: 4; failed and fell back: 0).
- Out of scope: 0; inferred filters relaxed: 0; catalog values dropped: 2.
- Tokens: 87,121 in / 4,615 out in total (1,708 / 90 per query).
- LLM latency: p50 1492 ms, p95 2002 ms.

## Weakest queries per mode (lowest nDCG@10)

**hybrid+qu**: `nomatch-womens-winter-parka` (0.00), `comfortable-walking-shoes` (0.05), `warm-jacket-for-winter` (0.09), `nomatch-tuxedo` (0.36), `nl-black-sneakers-daily` (0.44)
