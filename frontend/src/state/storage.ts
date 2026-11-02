/**
 * Recent conversations, kept in this browser only (localStorage).
 *
 * The API keeps no conversation state, so this is the only place a conversation
 * lives after the page closes. It's a convenience, not a record: storage can be
 * blocked (private windows, strict settings) or full, and every call here fails
 * quietly, leaving the app working without history.
 */

import type { Conversation, Turn } from "./conversation";
import { isOpen } from "./conversation";

export interface SessionSummary {
  id: string;
  title: string;
  updatedAt: number;
}

interface StoredSession extends SessionSummary {
  turns: Turn[];
}

type KeyValue = Pick<Storage, "getItem" | "setItem">;

export const STORAGE_KEY = "shopping-assistant.sessions.v1";
export const MAX_SESSIONS = 20;
const TITLE_CHARS = 60;

function browserStorage(): KeyValue | null {
  try {
    const s = window.localStorage;
    const probe = `${STORAGE_KEY}.probe`;
    s.setItem(probe, "1");
    s.removeItem(probe);
    return s;
  } catch {
    return null;
  }
}

/** Finished turns only, without debug data: open answers can't be resumed after a reload. */
function storableTurns(turns: Turn[]): Turn[] {
  const out: Turn[] = [];
  for (const t of turns) {
    if (t.role === "user") out.push(t);
    else if (!isOpen(t)) out.push(t.answer ? { ...t, answer: { ...t.answer, debug: null } } : t);
  }
  // A question whose answer is still open is left out with it.
  while (out.length && out[out.length - 1].role === "user") out.pop();
  return out;
}

function titleOf(turns: Turn[]): string {
  const first = turns.find((t) => t.role === "user");
  const text = first && first.role === "user" ? first.text : "Conversation";
  return text.length > TITLE_CHARS ? `${text.slice(0, TITLE_CHARS - 1)}…` : text;
}

export function createSessionStore(storage: KeyValue | null = browserStorage(), now: () => number = Date.now) {
  const read = (): StoredSession[] => {
    if (!storage) return [];
    try {
      const data = JSON.parse(storage.getItem(STORAGE_KEY) ?? "[]");
      return Array.isArray(data)
        ? data.filter((s) => s && typeof s.id === "string" && Array.isArray(s.turns))
        : [];
    } catch {
      return [];
    }
  };

  const write = (sessions: StoredSession[]): void => {
    if (!storage) return;
    let list = sessions;
    // Storage full: drop the oldest conversations until it fits.
    while (list.length) {
      try {
        storage.setItem(STORAGE_KEY, JSON.stringify(list));
        return;
      } catch {
        list = list.slice(0, -1);
      }
    }
  };

  return {
    available: storage !== null,

    /** Newest first, without the turns. */
    list(): SessionSummary[] {
      return read().map(({ id, title, updatedAt }) => ({ id, title, updatedAt }));
    },

    load(id: string): Conversation | null {
      const s = read().find((x) => x.id === id);
      return s ? { sessionId: s.id, turns: s.turns } : null;
    },

    /** Saves the finished part of a conversation; a no-op when nothing changed. */
    save(conversation: Conversation): void {
      const turns = storableTurns(conversation.turns);
      if (!turns.length) return;
      const sessions = read();
      const existing = sessions.find((s) => s.id === conversation.sessionId);
      if (existing && JSON.stringify(existing.turns) === JSON.stringify(turns)) return;
      const updated: StoredSession = { id: conversation.sessionId, title: titleOf(turns), updatedAt: now(), turns };
      write([updated, ...sessions.filter((s) => s.id !== conversation.sessionId)].slice(0, MAX_SESSIONS));
    },

    remove(id: string): void {
      write(read().filter((s) => s.id !== id));
    },
  };
}

export type SessionStore = ReturnType<typeof createSessionStore>;