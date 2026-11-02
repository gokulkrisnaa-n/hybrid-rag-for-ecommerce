/**
 * Waiting before retrying a rate-limited (429) request.
 *
 * Same policy as the server's retries (app/core/retry.py): honour Retry-After when
 * the server sends it, give up if it asks for more than a minute, otherwise back
 * off with full jitter: a random wait between 0 and min(cap, base * 2^attempt).
 * Full jitter spreads clients out so they don't all come back at the same moment.
 */

export const MAX_RETRIES = 3;
export const MAX_RETRY_AFTER_SECONDS = 60;

export function backoffSeconds(attempt: number, base = 1, cap = 8, rand: () => number = Math.random): number {
  return rand() * Math.min(cap, base * 2 ** attempt);
}

/** Seconds to wait before retry number `attempt` (0-based), or null to give up. */
export function retryDelaySeconds(
  retryAfter: number | null,
  attempt: number,
  rand: () => number = Math.random,
): number | null {
  if (attempt >= MAX_RETRIES) return null;
  if (retryAfter !== null) return retryAfter > MAX_RETRY_AFTER_SECONDS ? null : retryAfter;
  return backoffSeconds(attempt, 1, 8, rand);
}

/** setTimeout as a promise that rejects with an AbortError when the signal fires. */
export function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException("Aborted", "AbortError"));
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, ms);
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}