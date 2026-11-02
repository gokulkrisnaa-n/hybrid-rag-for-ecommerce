import { useState } from "react";

import { ApiError } from "../api/chat";
import { type Rating, sendFeedback } from "../api/feedback";
import { shortId } from "../format";
import "./Feedback.css";

type State = "idle" | "sending" | "sent" | "failed" | "unavailable";

interface Props {
  requestId: string;
  /** Set once the server says feedback is off, so the buttons hide for the session. */
  disabled: boolean;
  onUnavailable: () => void;
}

function Thumb({ down }: { down?: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
         aria-hidden="true" style={down ? { transform: "rotate(180deg)" } : undefined}>
      <path d="M7 10v11H3V10h4zm0 0l4-8a3 3 0 013 3v4h5.5a2 2 0 012 2.3l-1.4 8A2 2 0 0118.1 21H7" />
    </svg>
  );
}

export function Feedback({ requestId, disabled, onUnavailable }: Props) {
  const [rating, setRating] = useState<Rating | null>(null);
  const [state, setState] = useState<State>("idle");
  const [copied, setCopied] = useState(false);

  const rate = async (r: Rating) => {
    if (state === "sending" || r === rating) return;
    const previous = rating;
    setRating(r);                 // optimistic; voting again replaces the score on the server
    setState("sending");
    try {
      await sendFeedback(requestId, r);
      setState("sent");
    } catch (e) {
      setRating(previous);
      if (e instanceof ApiError && e.code === "tracing_disabled") {
        setState("unavailable");
        onUnavailable();
      } else {
        setState("failed");
      }
    }
  };

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(requestId);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // clipboard blocked: the id is still visible to select by hand
    }
  };

  return (
    <div className="feedback">
      {!disabled && (
        <div className="feedback__thumbs" role="group" aria-label="Rate this answer">
          <button type="button" className="feedback__btn" aria-pressed={rating === 1}
                  aria-label="Helpful" onClick={() => rate(1)}>
            <Thumb />
          </button>
          <button type="button" className="feedback__btn" aria-pressed={rating === -1}
                  aria-label="Not helpful" onClick={() => rate(-1)}>
            <Thumb down />
          </button>
        </div>
      )}
      <button type="button" className="feedback__id" onClick={copy} title="Copy the request id">
        {copied ? "Copied" : `Ref ${shortId(requestId)}`}
      </button>
      <span className="feedback__note" aria-live="polite">
        {state === "sent" && "Thanks for the feedback"}
        {state === "failed" && "Couldn't send feedback. Try again."}
        {state === "unavailable" && "Feedback is off on this server"}
      </span>
    </div>
  );
}