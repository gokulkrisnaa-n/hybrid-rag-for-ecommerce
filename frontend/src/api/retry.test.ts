import { describe, expect, it } from "vitest";

import { MAX_RETRIES, backoffSeconds, retryDelaySeconds, sleep } from "./retry";

describe("backoffSeconds (full jitter)", () => {
  it("is a random share of an exponential window, capped", () => {
    expect(backoffSeconds(0, 1, 8, () => 1)).toBe(1);
    expect(backoffSeconds(2, 1, 8, () => 1)).toBe(4);
    expect(backoffSeconds(5, 1, 8, () => 1)).toBe(8);           // capped
    expect(backoffSeconds(2, 1, 8, () => 0.25)).toBe(1);
    expect(backoffSeconds(3, 1, 8, () => 0)).toBe(0);
  });
});

describe("retryDelaySeconds", () => {
  it("uses Retry-After when the server sends it", () => {
    expect(retryDelaySeconds(7, 0, () => 0.5)).toBe(7);
  });

  it("gives up when the server asks for more than a minute", () => {
    expect(retryDelaySeconds(120, 0)).toBeNull();
  });

  it("backs off with jitter without Retry-After", () => {
    expect(retryDelaySeconds(null, 1, () => 0.5)).toBe(1);
  });

  it("stops after the last retry", () => {
    expect(retryDelaySeconds(1, MAX_RETRIES)).toBeNull();
    expect(retryDelaySeconds(null, MAX_RETRIES)).toBeNull();
  });
});

describe("sleep", () => {
  it("resolves after the wait", async () => {
    await expect(sleep(1)).resolves.toBeUndefined();
  });

  it("rejects when aborted, before or during the wait", async () => {
    const before = new AbortController();
    before.abort();
    await expect(sleep(1000, before.signal)).rejects.toMatchObject({ name: "AbortError" });

    const during = new AbortController();
    const p = sleep(10_000, during.signal);
    during.abort();
    await expect(p).rejects.toMatchObject({ name: "AbortError" });
  });
});