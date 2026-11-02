/**
 * Client for POST /v1/chat/stream.
 *
 * Errors found before the stream starts (invalid body, rate limit, database down)
 * come back as normal JSON errors with a status code and are thrown as ApiError.
 * Once the stream has started the status is 200, so a failure arrives as an
 * "error" event and is thrown the same way. Retrying is the caller's decision.
 */

import { SSEParser } from "./sse";
import type { ChatAnswer, ChatRequest, ErrorBody, Stage } from "./types";

export class ApiError extends Error {
  constructor(
    readonly status: number,          // HTTP status; 0 when the connection failed
    readonly code: string,            // the API's error code, e.g. "rate_limited"
    message: string,
    readonly requestId: string | null = null,
    readonly retryAfterSeconds: number | null = null,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export interface StreamHandlers {
  onMeta?: (requestId: string) => void;
  onStatus?: (stage: Stage) => void;
  onDelta?: (text: string) => void;
}

/** Retry-After as seconds: either a number of seconds or an HTTP date. */
export function parseRetryAfter(value: string | null, now: number = Date.now()): number | null {
  if (!value) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, seconds);
  const date = Date.parse(value);
  return Number.isNaN(date) ? null : Math.max(0, (date - now) / 1000);
}

async function errorFromResponse(res: Response): Promise<ApiError> {
  const retryAfter = parseRetryAfter(res.headers.get("Retry-After"));
  try {
    const body = (await res.json()) as { error?: ErrorBody };
    if (body.error) {
      return new ApiError(res.status, body.error.code, body.error.message, body.error.request_id, retryAfter);
    }
  } catch {
    // not our JSON (a proxy's HTML page, say): fall through
  }
  return new ApiError(res.status, "http_error", `The server answered ${res.status}.`, null, retryAfter);
}

/** Streams one chat turn. Resolves with the final answer; rejects with ApiError or an AbortError. */
export async function streamChat(
  request: ChatRequest,
  handlers: StreamHandlers = {},
  signal?: AbortSignal,
): Promise<ChatAnswer> {
  let res: Response;
  try {
    res = await fetch("/v1/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
      body: JSON.stringify(request),
      signal,
    });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new ApiError(0, "network_error", "Could not reach the server. Check your connection and try again.");
  }
  if (!res.ok) throw await errorFromResponse(res);
  if (!res.body) throw new ApiError(res.status, "empty_response", "The server sent an empty response.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parser = new SSEParser();
  let requestId: string | null = res.headers.get("X-Request-ID");

  try {
    for (;;) {
      let chunk: ReadableStreamReadResult<Uint8Array>;
      try {
        chunk = await reader.read();
      } catch (e) {
        if (signal?.aborted) throw e;
        throw new ApiError(0, "stream_interrupted", "The connection dropped before the answer finished.", requestId);
      }
      if (chunk.done) break;
      for (const { event, data } of parser.feed(decoder.decode(chunk.value, { stream: true }))) {
        const payload = JSON.parse(data);
        switch (event) {
          case "meta":
            requestId = payload.request_id;
            handlers.onMeta?.(payload.request_id);
            break;
          case "status":
            handlers.onStatus?.(payload.stage);
            break;
          case "delta":
            handlers.onDelta?.(payload.text);
            break;
          case "answer":
            return payload as ChatAnswer;
          case "error": {
            const err = payload.error as ErrorBody;
            throw new ApiError(200, err.code, err.message, err.request_id ?? requestId);
          }
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
  throw new ApiError(200, "stream_interrupted", "The connection dropped before the answer finished.", requestId);
}