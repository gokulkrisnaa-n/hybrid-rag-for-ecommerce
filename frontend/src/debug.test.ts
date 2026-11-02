import { describe, expect, it } from "vitest";

import type { ChatAnswer } from "./api/types";
import { hasDebug, querySummary, timeline, tokens, traceUrlFor } from "./debug";

const answer = (over: Partial<ChatAnswer> = {}): ChatAnswer => ({
  request_id: "abc", kind: "recommend", message: "m", products: [], comparison: null, follow_up: null,
  source: "llm", warnings: [],
  timings_ms: { search: 1800, context: 2, generate: 2300, first_token: 2800, total: 4100 },
  debug: {
    search_timings: { understand: 1600, embed_query: 0, bm25_search: 46, dense_search: 13, total: 1800 },
    qu: {
      used: true, prompt: "qu-v6", cached: false, search_text: "warm jacket",
      plan: { intent: "recommend" }, inferred_filters: { max_price: 300 }, dropped: [],
      tokens: { input: 1700, cached: 1500, output: 90 },
    },
    tokens: { input: 4000, cached: 2800, output: 210 },
  },
  ...over,
});

describe("timeline", () => {
  it("lays the turn out in pipeline order", () => {
    expect(timeline(answer())).toEqual([
      { label: "Understand", start: 0, duration: 1600, kind: "llm" },
      { label: "BM25", start: 1600, duration: 46, kind: "retrieval" },
      { label: "Vector", start: 1600, duration: 13, kind: "retrieval" },
      { label: "Load details", start: 1800, duration: 2, kind: "retrieval" },
      { label: "Generate", start: 1802, duration: 2300, kind: "llm" },
      { label: "First words", start: 2800, duration: 0, kind: "marker" },
      { label: "Total", start: 0, duration: 4100, kind: "total" },
    ]);
  });

  it("shows the embedding wait when retrieval had to wait for it", () => {
    const a = answer();
    (a.debug!.search_timings as Record<string, number>).embed_query = 220;
    const rows = timeline(a);
    expect(rows[1]).toEqual({ label: "Embed (wait)", start: 1600, duration: 220, kind: "wait" });
    expect(rows[2].start).toBe(1820);
  });

  it("copes with missing data", () => {
    expect(timeline(answer({ timings_ms: {}, debug: {} }))).toEqual([]);
  });
});

it("collects token counts per model call", () => {
  expect(tokens(answer())).toEqual([
    { step: "Understand", input: 1700, cached: 1500, output: 90 },
    { step: "Generate", input: 4000, cached: 2800, output: 210 },
  ]);
  expect(tokens(answer({ debug: {} }))).toEqual([]);
});

describe("querySummary", () => {
  it("reads the plan and inferred filters", () => {
    expect(querySummary(answer())).toMatchObject({
      used: true, searchText: "warm jacket", inferred: { max_price: 300 }, prompt: "qu-v6", relaxed: null,
    });
  });

  it("explains why understanding was skipped or failed", () => {
    expect(querySummary(answer({ debug: { qu: { used: false, reason: "style code" } } }))).toMatchObject({
      used: false, reason: "style code",
    });
    expect(querySummary(answer({ debug: { qu: { used: false, reason: "error", error: "Timeout" } } })).reason)
      .toBe("error: Timeout");
  });
});

it("knows when a turn has debug data", () => {
  expect(hasDebug(answer())).toBe(true);
  expect(hasDebug(answer({ debug: null }))).toBe(false);
  expect(hasDebug(null)).toBe(false);
});

it("builds a Langfuse link only when a project URL is configured", () => {
  // Pure on purpose: the test must not depend on the VITE_LANGFUSE_PROJECT_URL in your .env.local.
  expect(traceUrlFor("abc", "https://cloud.langfuse.com/project/p1/")).toBe("https://cloud.langfuse.com/project/p1/traces/abc");
  expect(traceUrlFor("abc", undefined)).toBeNull();
  expect(traceUrlFor("abc", "")).toBeNull();
});