/** Polls GET /health/ready: on load, every 30 s while the tab is visible, and when it becomes visible again. */

import { useEffect, useState } from "react";

import { type Health, checkHealth } from "../api/health";

const INTERVAL_MS = 30_000;

export function useHealth(): Health {
  const [health, setHealth] = useState<Health>("checking");

  useEffect(() => {
    let ctrl: AbortController | null = null;
    const run = async () => {
      if (document.visibilityState === "hidden") return;
      ctrl?.abort();
      ctrl = new AbortController();
      try {
        setHealth(await checkHealth(ctrl.signal));
      } catch {
        // aborted: a newer check is running
      }
    };
    void run();
    const timer = setInterval(run, INTERVAL_MS);
    document.addEventListener("visibilitychange", run);
    return () => {
      clearInterval(timer);
      document.removeEventListener("visibilitychange", run);
      ctrl?.abort();
    };
  }, []);

  return health;
}