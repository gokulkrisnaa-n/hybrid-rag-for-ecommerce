/**
 * The conversation, the streaming call and the saved sessions.
 * Components read state from here and call its actions; nothing else talks to the API.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";

import { ApiError, streamChat } from "../api/chat";
import { MAX_RETRIES, retryDelaySeconds, sleep } from "../api/retry";
import type { ChatAnswer } from "../api/types";
import { buildHistory, newConversation, openTurn, reducer, retryableTurn } from "./conversation";
import { type SessionSummary, createSessionStore } from "./storage";

export const newId = (): string =>
  typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

/** Shown while a rate-limited request waits to retry. */
export interface RetryNotice {
  until: number;          // epoch ms
  attempt: number;        // 1-based retry number
  max: number;
}

const DEBUG_KEY = "shopping-assistant.debug";

function readDebugPref(): boolean {
  try {
    return window.localStorage.getItem(DEBUG_KEY) === "1";
  } catch {
    return false;
  }
}

export function useChat() {
  const store = useMemo(() => createSessionStore(), []);
  // Debug: ask the API for the query plan, timings and tokens with each answer.
  // `debugAllowed` turns false for good once the server answers 403 debug_disabled.
  const [debugOn, setDebugOnState] = useState(readDebugPref);
  const [debugAllowed, setDebugAllowed] = useState(true);
  const debugRef = useRef(debugOn && debugAllowed);
  debugRef.current = debugOn && debugAllowed;
  const [conversation, dispatch] = useReducer(reducer, undefined, () => newConversation(newId()));
  const [sessions, setSessions] = useState<SessionSummary[]>(() => store.list());
  const [notice, setNotice] = useState<RetryNotice | null>(null);
  const controller = useRef<AbortController | null>(null);
  // The latest state, for callbacks that outlive a render.
  const latest = useRef(conversation);
  latest.current = conversation;

  const busy = openTurn(conversation) !== null;

  // Save each conversation once its answer is finished (the store skips unchanged ones).
  useEffect(() => {
    if (busy || conversation.turns.length === 0) return;
    store.save(conversation);
    setSessions(store.list());
  }, [conversation, busy, store]);

  const send = useCallback(async (text: string) => {
    const message = text.trim();
    if (!message || openTurn(latest.current)) return;

    const history = buildHistory(latest.current);
    const sessionId = latest.current.sessionId;
    const id = newId();
    dispatch({ type: "send", userId: newId(), assistantId: id, text: message });

    const ctrl = new AbortController();
    controller.current = ctrl;
    let debug = debugRef.current;
    try {
      for (let attempt = 0; ; attempt++) {
        try {
          const answer: ChatAnswer = await streamChat(
            { message, history, session_id: sessionId, ...(debug ? { debug: true } : {}) },
            {
              onMeta: (requestId) => dispatch({ type: "meta", id, requestId }),
              onStatus: (stage) => dispatch({ type: "status", id, stage }),
              onDelta: (t) => dispatch({ type: "delta", id, text: t }),
            },
            ctrl.signal,
          );
          dispatch({ type: "answer", id, answer });
          return;
        } catch (e) {
          if (ctrl.signal.aborted) return;            // stop() already marked the turn
          const err = e instanceof ApiError
            ? e
            : new ApiError(0, "client_error", "Something went wrong in the page. Please try again.");
          // The server doesn't allow debug output: stop asking, and send this one again without it.
          if (err.code === "debug_disabled" && debug) {
            debug = false;
            setDebugAllowed(false);
            attempt--;
            continue;
          }
          // A 429 arrives before streaming starts, so nothing was shown yet: wait and retry.
          const wait = err.status === 429 ? retryDelaySeconds(err.retryAfterSeconds, attempt) : null;
          if (wait === null) {
            dispatch({ type: "error", id, code: err.code, message: err.message, requestId: err.requestId });
            return;
          }
          setNotice({ until: Date.now() + wait * 1000, attempt: attempt + 1, max: MAX_RETRIES });
          try {
            await sleep(wait * 1000, ctrl.signal);
          } catch {
            return;                                    // stopped while waiting
          } finally {
            setNotice(null);
          }
        }
      }
    } finally {
      if (controller.current === ctrl) controller.current = null;
    }
  }, []);

  const stop = useCallback(() => {
    const open = openTurn(latest.current);
    if (!open) return;
    dispatch({ type: "stop", id: open.id });
    controller.current?.abort();       // the server sees the disconnect and stops the model
    setNotice(null);
  }, []);

  /** Ask the last question again after it failed or was stopped. */
  const retry = useCallback(() => {
    const last = retryableTurn(latest.current);
    if (!last) return;
    dispatch({ type: "remove", ids: [last.userId, last.assistantId] });
    // The removal is applied before send's own dispatch, and the failed pair was
    // never part of the history, so send() sees the right state.
    void send(last.text);
  }, [send]);

  const newChat = useCallback(() => {
    controller.current?.abort();
    setNotice(null);
    dispatch({ type: "reset", sessionId: newId() });
  }, []);

  const openSession = useCallback((sessionId: string) => {
    if (sessionId === latest.current.sessionId) return;
    const saved = store.load(sessionId);
    if (!saved) return;
    controller.current?.abort();
    setNotice(null);
    dispatch({ type: "load", conversation: saved });
  }, [store]);

  const deleteSession = useCallback((sessionId: string) => {
    store.remove(sessionId);
    setSessions(store.list());
    if (sessionId === latest.current.sessionId) {
      controller.current?.abort();
      dispatch({ type: "reset", sessionId: newId() });
    }
  }, [store]);

  const setDebugOn = useCallback((on: boolean) => {
    setDebugOnState(on);
    try {
      window.localStorage.setItem(DEBUG_KEY, on ? "1" : "0");
    } catch {
      // storage blocked: the setting lasts until the page closes
    }
  }, []);

  return {
    conversation, busy, notice, sessions, historyAvailable: store.available,
    debugOn: debugOn && debugAllowed, debugAllowed, setDebugOn,
    canRetry: retryableTurn(conversation) !== null,
    send, stop, retry, newChat, openSession, deleteSession,
  };
}