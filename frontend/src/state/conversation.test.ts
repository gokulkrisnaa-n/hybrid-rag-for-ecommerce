import { describe, expect, it } from "vitest";

import type { ChatAnswer } from "../api/types";
import { type AssistantTurn, type Conversation, buildHistory, newConversation, openTurn, reducer } from "./conversation";

const answer = (message: string, request_id = "req1"): ChatAnswer => ({
  request_id, kind: "recommend", message, products: [], comparison: null, follow_up: null,
  source: "llm", warnings: [], timings_ms: {}, debug: null,
});

const send = (s: Conversation, n: number, text: string) =>
  reducer(s, { type: "send", userId: `u${n}`, assistantId: `a${n}`, text });

const assistant = (s: Conversation, id: string) => s.turns.find((t) => t.id === id) as AssistantTurn;

describe("conversation reducer", () => {
  it("adds the question and an open answer", () => {
    const s = send(newConversation("s"), 1, "warm jacket");
    expect(s.turns.map((t) => t.role)).toEqual(["user", "assistant"]);
    expect(assistant(s, "a1").status).toBe("pending");
    expect(openTurn(s)?.id).toBe("a1");
  });

  it("streams deltas, then replaces them with the validated message", () => {
    let s = send(newConversation("s"), 1, "q");
    s = reducer(s, { type: "meta", id: "a1", requestId: "req1" });
    s = reducer(s, { type: "status", id: "a1", stage: "searching" });
    s = reducer(s, { type: "delta", id: "a1", text: "Here " });
    s = reducer(s, { type: "delta", id: "a1", text: "are" });
    expect(assistant(s, "a1")).toMatchObject({ status: "writing", text: "Here are", requestId: "req1" });
    s = reducer(s, { type: "answer", id: "a1", answer: answer("Here are three picks.") });
    expect(assistant(s, "a1")).toMatchObject({ status: "done", text: "Here are three picks." });
    expect(openTurn(s)).toBeNull();
  });

  it("keeps partial text when stopped and ignores anything that arrives later", () => {
    let s = send(newConversation("s"), 1, "q");
    s = reducer(s, { type: "delta", id: "a1", text: "Partial" });
    s = reducer(s, { type: "stop", id: "a1" });
    s = reducer(s, { type: "delta", id: "a1", text: " more" });
    s = reducer(s, { type: "answer", id: "a1", answer: answer("Full") });
    expect(assistant(s, "a1")).toMatchObject({ status: "stopped", text: "Partial", answer: null });
  });

  it("records an error with its code", () => {
    let s = send(newConversation("s"), 1, "q");
    s = reducer(s, { type: "error", id: "a1", code: "rate_limited", message: "Slow down", requestId: "r9" });
    expect(assistant(s, "a1")).toMatchObject({
      status: "error", error: { code: "rate_limited", message: "Slow down" }, requestId: "r9",
    });
  });

  it("resets to an empty conversation with a new session id", () => {
    const s = reducer(send(newConversation("s1"), 1, "q"), { type: "reset", sessionId: "s2" });
    expect(s).toEqual({ sessionId: "s2", turns: [] });
  });
});

describe("buildHistory", () => {
  it("includes finished exchanges with the final message, oldest first", () => {
    let s = send(newConversation("s"), 1, "warm jacket");
    s = reducer(s, { type: "delta", id: "a1", text: "streamed draft" });
    s = reducer(s, { type: "answer", id: "a1", answer: answer("Final message") });
    expect(buildHistory(s)).toEqual([
      { role: "user", content: "warm jacket" },
      { role: "assistant", content: "Final message" },
    ]);
  });

  it("leaves out stopped, failed and open exchanges together with their questions", () => {
    let s = send(newConversation("s"), 1, "first");
    s = reducer(s, { type: "answer", id: "a1", answer: answer("A1") });
    s = send(s, 2, "stopped one");
    s = reducer(s, { type: "stop", id: "a2" });
    s = send(s, 3, "failed one");
    s = reducer(s, { type: "error", id: "a3", code: "internal_error", message: "x" });
    s = send(s, 4, "still open");
    expect(buildHistory(s).map((t) => t.content)).toEqual(["first", "A1"]);
  });

  it("keeps only the most recent turns", () => {
    let s = newConversation("s");
    for (let n = 1; n <= 8; n++) {
      s = send(s, n, `q${n}`);
      s = reducer(s, { type: "answer", id: `a${n}`, answer: answer(`a${n}`) });
    }
    const h = buildHistory(s, 4);
    expect(h.map((t) => t.content)).toEqual(["q7", "a7", "q8", "a8"]);
  });
});