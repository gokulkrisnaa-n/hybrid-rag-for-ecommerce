# Latency benchmark, 2026-10-31 12:47 (qu-minimal)

37 turns (0 failed). Streamed; plan and embedding caches cleared before each turn. Query understanding gpt-6-luna (qu-v6, reasoning minimal); generation gpt-6-luna (gen-v1, reasoning low); embedding prefetch on; 8 candidates.

| Step | Turns | p50 ms | p95 ms | Max ms |
|---|---|---|---|---|
| understand | 37 | 251 | 417 | 521 |
| embed (wait) | 37 | 85 | 265 | 290 |
| bm25 | 37 | 38 | 67 | 73 |
| dense | 37 | 7 | 17 | 20 |
| search total | 37 | 251 | 417 | 521 |
| load details | 37 | 2 | 3 | 3 |
| generate | 37 | 2,353 | 4,623 | 5,428 |
| first words | 37 | 1,175 | 3,116 | 3,477 |
| total | 37 | 2,600 | 4,848 | 5,645 |

`embed (wait)` is how long retrieval waited for the query vector: near 0 when the prefetch finished while query understanding ran. `first words`: from the start of the turn to the first streamed text (turns answered by the model only; fixed replies have none).

**Slowest turns:** `compare-chuck-70-all-star` 5,645 ms (compare); `rec-newborn-gift` 4,848 ms (recommend); `rec-saint-laurent` 4,191 ms (recommend); `compare-nuptse-himalayan` 4,138 ms (no_match); `rec-kith-asics` 4,057 ms (recommend)
