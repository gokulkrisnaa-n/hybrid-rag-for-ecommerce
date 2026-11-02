/**
 * One conversation as a list of turns, and the actions that change it.
 *
 * A pure reducer (no fetch, no timers), so every transition is unit-tested.
 * The API is stateless: buildHistory() turns the finished turns back into the
 * {role, content} list each request carries.
 */

import type { ChatAnswer, HistoryTurn, Stage } from "../api/types";
import { MAX_HISTORY_TURNS } from "../api/types";

export interface UserTurn {
  id: string;
  role: "user";
  text: string;
}

export type AssistantStatus = "pending" | Stage | "done" | "error" | "stopped";

export interface AssistantTurn {
  id: string;
  role: "assistant";
  status: AssistantStatus;
  text: string;                      // streamed text; replaced by answer.message when done
  answer: ChatAnswer | null;
  requestId: string | null;
  error: { code: string; message: string } | null;
}

export type Turn = UserTurn | AssistantTurn;

export interface Conversation {
  sessionId: string;
  turns: Turn[];
}

export type Action =
  | { type: "send"; userId: string; assistantId: string; text: string }
  | { type: "meta"; id: string; requestId: string }
  | { type: "status"; id: string; stage: Stage }
  | { type: "delta"; id: string; text: string }
  | { type: "answer"; id: string; answer: ChatAnswer }
  | { type: "error"; id: string; code: string; message: string; requestId?: string | null }
  | { type: "stop"; id: string }
  | { type: "reset"; sessionId: string };

export const newConversation = (sessionId: string): Conversation => ({ sessionId, turns: [] });

function updateAssistant(state: Conversation, id: string, change: (t: AssistantTurn) => AssistantTurn): Conversation {
  return {
    ...state,
    turns: state.turns.map((t) => (t.id === id && t.role === "assistant" ? change(t) : t)),
  };
}

const isOpen = (t: AssistantTurn) => t.status === "pending" || t.status === "searching" || t.status === "writing";

export function reducer(state: Conversation, action: Action): Conversation {
  switch (action.type) {
    case "send":
      return {
        ...state,
        turns: [
          ...state.turns,
          { id: action.userId, role: "user", text: action.text },
          {
            id: action.assistantId, role: "assistant", status: "pending",
            text: "", answer: null, requestId: null, error: null,
          },
        ],
      };
    case "meta":
      return updateAssistant(state, action.id, (t) => ({ ...t, requestId: action.requestId }));
    case "status":
      return updateAssistant(state, action.id, (t) => (isOpen(t) ? { ...t, status: action.stage } : t));
    case "delta":
      // A late delta after Stop or an error is ignored.
      return updateAssistant(state, action.id, (t) =>
        isOpen(t) ? { ...t, status: "writing", text: t.text + action.text } : t);
    case "answer":
      return updateAssistant(state, action.id, (t) =>
        isOpen(t)
          ? { ...t, status: "done", text: action.answer.message, answer: action.answer,
              requestId: action.answer.request_id ?? t.requestId }
          : t);
    case "error":
      return updateAssistant(state, action.id, (t) =>
        isOpen(t)
          ? { ...t, status: "error", error: { code: action.code, message: action.message },
              requestId: action.requestId ?? t.requestId }
          : t);
    case "stop":
      return updateAssistant(state, action.id, (t) => (isOpen(t) ? { ...t, status: "stopped" } : t));
    case "reset":
      return newConversation(action.sessionId);
  }
}

/** The assistant turn still being answered, if any. */
export function openTurn(state: Conversation): AssistantTurn | null {
  const last = state.turns[state.turns.length - 1];
  return last && last.role === "assistant" && isOpen(last) ? last : null;
}

/**
 * Earlier turns for the next request, oldest first. Only finished exchanges count:
 * a question whose answer was stopped or failed is left out with it, so the model
 * never sees half an answer. Capped to the last MAX_HISTORY_TURNS turns.
 */
export function buildHistory(state: Conversation, maxTurns: number = MAX_HISTORY_TURNS): HistoryTurn[] {
  const out: HistoryTurn[] = [];
  const turns = state.turns;
  for (let i = 0; i + 1 < turns.length; i++) {
    const q = turns[i];
    const a = turns[i + 1];
    if (q.role === "user" && a.role === "assistant" && a.status === "done" && a.answer) {
      out.push({ role: "user", content: q.text }, { role: "assistant", content: a.answer.message });
      i++;
    }
  }
  return maxTurns > 0 ? out.slice(-maxTurns) : [];
}