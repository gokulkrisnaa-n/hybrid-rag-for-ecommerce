import { afterEach, describe, expect, it, vi } from "vitest";

import { sendFeedback } from "./feedback";

afterEach(() => vi.unstubAllGlobals());

describe("sendFeedback", () => {
  it("posts the rating for a request id", async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ request_id: "r1", status: "recorded" }), { status: 202 }));
    vi.stubGlobal("fetch", fetch);
    await sendFeedback("r1", -1);
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/v1/feedback");
    expect(JSON.parse(init.body as string)).toEqual({ request_id: "r1", rating: -1 });
  });

  it("reports that feedback is off when tracing is not configured", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(
      JSON.stringify({ error: { code: "tracing_disabled", message: "off", request_id: "x" } }), { status: 503 },
    )));
    await expect(sendFeedback("r1", 1)).rejects.toMatchObject({ status: 503, code: "tracing_disabled" });
  });
});