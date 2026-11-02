import { describe, expect, it } from "vitest";

import type { ChatAnswer } from "../api/types";
import { type Conversation, newConversation, reducer } from "./conversation";
import { MAX_SESSIONS, STORAGE_KEY, createSessionStore } from "./storage";

class MemoryStorage {
  data = new Map<string, string>();
  limit = Infinity;
  getItem(k: string) { return this.data.get(k) ?? null; }
  setItem(k: string, v: string) {
    if (v.length > this.limit) throw new DOMException("full", "QuotaExceededError");
    this.data.set(k, v);
  }
}

const answer = (message: string): ChatAnswer => ({
  request_id: "r", kind: "recommend", message, products: [], comparison: null, follow_up: null,
  source: "llm", warnings: [], timings_ms: {}, debug: { qu: "big" },
});

function chat(id: string, ...questions: string[]): Conversation {
  let s = newConversation(id);
  questions.forEach((q, i) => {
    s = reducer(s, { type: "send", userId: `u${i}`, assistantId: `a${i}`, text: q });
    s = reducer(s, { type: "answer", id: `a${i}`, answer: answer(`answer to ${q}`) });
  });
  return s;
}

describe("session store", () => {
  it("saves, lists newest first and loads conversations", () => {
    let t = 0;
    const store = createSessionStore(new MemoryStorage(), () => ++t);
    store.save(chat("s1", "warm jacket"));
    store.save(chat("s2", "black sneakers"));
    expect(store.list().map((s) => [s.id, s.title])).toEqual([["s2", "black sneakers"], ["s1", "warm jacket"]]);
    expect(store.load("s1")?.turns).toHaveLength(2);
    expect(store.load("nope")).toBeNull();
  });

  it("keeps finished turns only and drops debug data", () => {
    const store = createSessionStore(new MemoryStorage());
    let s = chat("s1", "first");
    s = reducer(s, { type: "send", userId: "u9", assistantId: "a9", text: "still answering" });
    store.save(s);
    const turns = store.load("s1")!.turns;
    expect(turns).toHaveLength(2);
    const a = turns[1];
    expect(a.role === "assistant" && a.answer?.debug).toBeNull();
  });

  it("does not bump a conversation that hasn't changed", () => {
    let t = 0;
    const store = createSessionStore(new MemoryStorage(), () => ++t);
    store.save(chat("s1", "a"));
    store.save(chat("s2", "b"));
    store.save(chat("s1", "a"));                      // same content: stays second
    expect(store.list()[0].id).toBe("s2");
  });

  it("caps the number of conversations and removes them on request", () => {
    const store = createSessionStore(new MemoryStorage());
    for (let i = 0; i < MAX_SESSIONS + 5; i++) store.save(chat(`s${i}`, `q${i}`));
    expect(store.list()).toHaveLength(MAX_SESSIONS);
    store.remove(`s${MAX_SESSIONS + 4}`);
    expect(store.list().some((s) => s.id === `s${MAX_SESSIONS + 4}`)).toBe(false);
  });

  it("drops the oldest conversations when storage is full", () => {
    const mem = new MemoryStorage();
    const store = createSessionStore(mem);
    store.save(chat("s1", "first"));
    store.save(chat("s2", "second"));
    mem.limit = (mem.getItem(STORAGE_KEY) ?? "").length;   // room for about two
    store.save(chat("s3", "third"));
    const ids = store.list().map((s) => s.id);
    expect(ids[0]).toBe("s3");
    expect(ids).not.toContain("s1");
  });

  it("works, empty, without storage or with corrupt data", () => {
    const none = createSessionStore(null);
    none.save(chat("s1", "x"));
    expect(none.list()).toEqual([]);
    expect(none.available).toBe(false);

    const mem = new MemoryStorage();
    mem.setItem(STORAGE_KEY, "{not json");
    expect(createSessionStore(mem).list()).toEqual([]);
  });

  it("shortens long titles", () => {
    const store = createSessionStore(new MemoryStorage());
    store.save(chat("s1", "x".repeat(100)));
    expect(store.list()[0].title).toHaveLength(60);
  });
});