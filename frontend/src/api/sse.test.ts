import { describe, expect, it } from "vitest";

import { SSEParser } from "./sse";

const all = (chunks: string[]) => {
  const p = new SSEParser();
  return chunks.flatMap((c) => p.feed(c));
};

describe("SSEParser", () => {
  it("parses a complete event", () => {
    expect(all(['event: meta\ndata: {"request_id":"abc"}\n\n'])).toEqual([
      { event: "meta", data: '{"request_id":"abc"}' },
    ]);
  });

  it("parses several events in one chunk", () => {
    const out = all(['event: status\ndata: {"stage":"searching"}\n\nevent: delta\ndata: {"text":"Hi"}\n\n']);
    expect(out.map((e) => e.event)).toEqual(["status", "delta"]);
  });

  it("joins an event split across chunks at every position", () => {
    const text = 'event: delta\ndata: {"text":"Hello, world"}\n\n';
    for (let i = 1; i < text.length; i++) {
      expect(all([text.slice(0, i), text.slice(i)])).toEqual([{ event: "delta", data: '{"text":"Hello, world"}' }]);
    }
  });

  it("handles CRLF line endings, even split between \\r and \\n", () => {
    expect(all(["event: a\r", "\ndata: 1\r\n\r", "\n"])).toEqual([{ event: "a", data: "1" }]);
  });

  it("handles bare CR line endings", () => {
    // A trailing \r could be half of \r\n, so it is settled by the next chunk.
    const p = new SSEParser();
    expect(p.feed("event: a\rdata: 1\r\r")).toEqual([]);
    expect(p.feed("event: b")).toEqual([{ event: "a", data: "1" }]);
  });

  it("joins multi-line data with newlines", () => {
    expect(all(["data: line one\ndata: line two\n\n"])).toEqual([{ event: "message", data: "line one\nline two" }]);
  });

  it("ignores comments and events without data", () => {
    expect(all([": keep-alive\n\nevent: empty\n\nevent: x\ndata: y\n\n"])).toEqual([{ event: "x", data: "y" }]);
  });

  it("keeps a value's spaces after the first one", () => {
    expect(all(["data:  two spaces\n\n"])).toEqual([{ event: "message", data: " two spaces" }]);
  });

  it("does not dispatch an event until its blank line arrives", () => {
    const p = new SSEParser();
    expect(p.feed("event: answer\ndata: {}\n")).toEqual([]);
    expect(p.feed("\n")).toEqual([{ event: "answer", data: "{}" }]);
  });

  it("keeps unicode intact", () => {
    expect(all(['data: {"text":"café — ✓"}\n\n'])[0].data).toBe('{"text":"café — ✓"}');
  });
});