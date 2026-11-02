/**
 * The conversation state plus the streaming call: send(), stop(), newChat().
 * Components only read `conversation` and call these three.
 */

import { useCallback, useReducer, useRef } from "react";

import { ApiError, streamChat } from "../api/chat";
import { buildHistory, newConversation, openTurn, reducer } from "./conversation";

export const newId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

export function useChat() {
  const [conversation, dispatch] = useReducer(reducer, undefined, () => newConversation(newId()));
  const controller = useRef<AbortController | null>(null);
  // The latest state, for callbacks that outlive a render.
  const latest = useRef(conversation);
  latest.current = conversation;

  const busy = openTurn(conversation) !== null;

  const send = useCallback(async (text: string) => {
    const message = text.trim();
    if (!message || openTurn(latest.current)) return;

    const history = buildHistory(latest.current);
    const id = newId();
    dispatch({ type: "send", userId: newId(), assistantId: id, text: message });

    const ctrl = new AbortController();
    controller.current = ctrl;
    try {
      const answer = await streamChat(
        { message, history, session_id: latest.current.sessionId },
        {
          onMeta: (requestId) => dispatch({ type: "meta", id, requestId }),
          onStatus: (stage) => dispatch({ type: "status", id, stage }),
          onDelta: (t) => dispatch({ type: "delta", id, text: t }),
        },
        ctrl.signal,
      );
      dispatch({ type: "answer", id, answer });
    } catch (e) {
      if (ctrl.signal.aborted) return;            // stop() already marked the turn
      const err = e instanceof ApiError
        ? e
        : new ApiError(0, "client_error", "Something went wrong in the page. Please try again.");
      dispatch({ type: "error", id, code: err.code, message: err.message, requestId: err.requestId });
    } finally {
      if (controller.current === ctrl) controller.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    const open = openTurn(latest.current);
    if (!open) return;
    dispatch({ type: "stop", id: open.id });
    controller.current?.abort();       // the server sees the disconnect and stops the model
  }, []);

  const newChat = useCallback(() => {
    controller.current?.abort();
    dispatch({ type: "reset", sessionId: newId() });
  }, []);

  return { conversation, busy, send, stop, newChat };
}