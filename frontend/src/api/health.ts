/**
 * GET /health/ready: is the API up, and can it reach the database?
 *   ok        200, everything ready
 *   degraded  503 from the API itself: it runs, but the catalog database is down
 *   offline   no answer, or a proxy's error page: the API isn't reachable
 */

export type Health = "checking" | "ok" | "degraded" | "offline";

export async function checkHealth(signal?: AbortSignal): Promise<Exclude<Health, "checking">> {
  try {
    const res = await fetch("/health/ready", { signal, cache: "no-store" });
    if (res.ok) return "ok";
    try {
      const body = (await res.json()) as { status?: string };
      if (body.status === "unavailable") return "degraded";
    } catch {
      // not the API's JSON: a proxy answered
    }
    return "offline";
  } catch (e) {
    if (signal?.aborted) throw e;
    return "offline";
  }
}