import { useEffect, useState } from "react";

import type { RetryNotice } from "../state/useChat";
import "./RetryToast.css";

/** "Busy right now. Retrying in 3 s (attempt 1 of 3)", counting down while a 429 waits. */
export function RetryToast({ notice }: { notice: RetryNotice | null }) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    if (!notice) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(timer);
  }, [notice]);

  if (!notice) return null;
  const seconds = Math.max(0, Math.ceil((notice.until - now) / 1000));
  return (
    <div className="toast" role="status">
      Busy right now. Retrying in {seconds} s (attempt {notice.attempt} of {notice.max})
    </div>
  );
}