# Retrieval eval, 2026-10-31 12:46 (qu-minimal)

55 queries, k = 10. Embeddings: text-embedding-3-small (768 dims); RRF k = 60; 50 candidates per method.

RRF weights bm25 1 / dense 1; code routing on; fuzzy fallback on; coverage weighting on. Query understanding: gpt-6-luna, prompt qu-v6, dense weight specific 0.75 / explore 1.5; subcategory filters off; dense embeds the original text; reasoning effort minimal.

| Mode | Recall@10 | MRR | nDCG@10 | p50 ms | p95 ms |
|---|---|---|---|---|---|
| hybrid+qu | 0.397 | 0.891 | 0.777 | 221 | 334 |

## nDCG@10 by query type

| Tag | Queries | hybrid+qu |
|---|---|---|
| attribute | 6 | 0.833 |
| brand | 4 | 0.911 |
| collab | 7 | 0.883 |
| exact-name | 8 | 0.876 |
| filtered | 6 | 0.871 |
| kids | 4 | 0.976 |
| model-family | 1 | 0.936 |
| natural-language | 3 | 0.814 |
| no-good-match | 2 | 0.370 |
| style-code | 4 | 0.871 |
| typo | 4 | 0.533 |
| vague-intent | 8 | 0.496 |

## Query understanding

- LLM used on 0/55 queries (skipped for style codes: 4; failed and fell back: 51).
- Out of scope: 0; inferred filters relaxed: 0; catalog values dropped: 0.
- Tokens: 0 in / 0 out in total (0 / 0 per query).
- LLM latency: p50 223 ms, p95 334 ms.
- First error: `BadRequestError: Error code: 400 - {'error': {'message': "Unsupported value: 'minimal' is not supported with the 'gpt-6-luna' model. Supported values are: 'none', 'low', 'medium', 'high', 'xhigh', and 'max'.", 'type': 'invalid_request_error', 'param': 'reasoning.effort', 'code': 'unsupported_value'}`

## Weakest queries per mode (lowest nDCG@10)

**hybrid+qu**: `nomatch-womens-winter-parka` (0.00), `typo-nupste-puffer` (0.00), `warm-jacket-for-winter` (0.00), `comfortable-walking-shoes` (0.08), `rain-jacket` (0.21)
