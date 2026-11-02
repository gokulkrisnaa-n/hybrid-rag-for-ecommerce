import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, parseRetryAfter, streamChat } from "./chat";

/** A fetch Response whose body arrives in the given chunks. */
function streamResponse(chunks: string[], init: ResponseInit = {}): Response {
  const enc = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(enc.encode(c));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" }, ...init });
}

const ev = (event: string, data: unknown) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

const ANSWER = {
  request_id: "req1", kind: "recommend", message: "Final.", products: [], comparison: null,
  follow_up: null, source: "llm", warnings: [], timings_ms: {}, debug: null,
};

afterEach(() => vi.unstubAllGlobals());

describe("streamChat", () => {
  it("reports meta, status and deltas, then resolves with the answer", async () => {
    const body = ev("meta", { request_id: "req1" }) + ev("status", { stage: "searching" })
      + ev("status", { stage: "writing" }) + ev("delta", { text: "Fin" }) + ev("delta", { text: "al" })
      + ev("answer", ANSWER);
    // split mid-event to make sure parsing across chunks works end to end
    vi.stubGlobal("fetch", vi.fn(async () => streamResponse([body.slice(0, 50), body.slice(50)])));

    const seen: string[] = [];
    const result = await streamChat({ message: "q", history: [] }, {
      onMeta: (id) => seen.push(`meta:${id}`),
      onStatus: (s) => seen.push(`status:${s}`),
      onDelta: (t) => seen.push(`delta:${t}`),
    });
    expect(seen).toEqual(["meta:req1", "status:searching", "status:writing", "delta:Fin", "delta:al"]);
    expect(result.message).toBe("Final.");
  });

  it("throws the API error for a JSON error response, with Retry-After", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { code: "rate_limited", message: "Too many requests", request_id: "r2" } }),
      { status: 429, headers: { "Content-Type": "application/json", "Retry-After": "7" } },
    )));
    const err = await streamChat({ message: "q", history: [] }).catch((e) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 429, code: "rate_limited", requestId: "r2", retryAfterSeconds: 7 });
  });

  it("throws an error event sent after the stream started", async () => {
    const body = ev("meta", { request_id: "req3" })
      + ev("error", { error: { code: "database_unavailable", message: "DB down", request_id: "req3" } });
    vi.stubGlobal("fetch", vi.fn(async () => streamResponse([body])));
    const err = await streamChat({ message: "q", history: [] }).catch((e) => e);
    expect(err).toMatchObject({ code: "database_unavailable", requestId: "req3" });
  });

  it("treats a stream that ends without an answer as interrupted", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => streamResponse([ev("meta", { request_id: "r4" }), ev("delta", { text: "Hal" })])));
    const err = await streamChat({ message: "q", history: [] }).catch((e) => e);
    expect(err).toMatchObject({ code: "stream_interrupted", requestId: "r4" });
  });

  it("reports a network failure as network_error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => { throw new TypeError("Failed to fetch"); }));
    const err = await streamChat({ message: "q", history: [] }).catch((e) => e);
    expect(err).toMatchObject({ status: 0, code: "network_error" });
  });

  it("handles a non-JSON error page", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>Bad gateway</html>", { status: 502 })));
    const err = await streamChat({ message: "q", history: [] }).catch((e) => e);
    expect(err).toMatchObject({ status: 502, code: "http_error" });
  });
});

describe("parseRetryAfter", () => {
  it("reads seconds and HTTP dates", () => {
    expect(parseRetryAfter("3")).toBe(3);
    expect(parseRetryAfter(null)).toBeNull();
    expect(parseRetryAfter("soon")).toBeNull();
    const now = Date.parse("2026-11-02T12:00:00Z");
    expect(parseRetryAfter("Mon, 02 Nov 2026 12:00:05 GMT", now)).toBe(5);
  });
});