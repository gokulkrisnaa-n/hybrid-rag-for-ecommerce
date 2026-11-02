/** POST /v1/feedback: a thumbs up (1) or down (-1) on one answer, by its request_id. */

import { ApiError, errorFromResponse } from "./chat";

export type Rating = 1 | -1;

export async function sendFeedback(requestId: string, rating: Rating, signal?: AbortSignal): Promise<void> {
  let res: Response;
  try {
    res = await fetch("/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ request_id: requestId, rating }),
      signal,
    });
  } catch (e) {
    if (signal?.aborted) throw e;
    throw new ApiError(0, "network_error", "Could not reach the server.");
  }
  if (!res.ok) throw await errorFromResponse(res);   // 503 tracing_disabled: feedback is off on this server
}