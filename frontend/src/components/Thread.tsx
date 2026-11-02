import { useEffect } from "react";

import { hasDebug } from "../debug";
import type { AssistantTurn, Turn } from "../state/conversation";
import { ComparisonTable } from "./ComparisonTable";
import { Feedback } from "./Feedback";
import { FilterChips } from "./FilterChips";
import { ProductCards } from "./ProductCards";
import "./Thread.css";

const STAGE_TEXT: Record<string, string> = {
  pending: "Reading your question",
  searching: "Searching the catalog",
  writing: "Writing",
};

interface AssistantProps {
  turn: AssistantTurn;
  busy: boolean;
  /** Set on the last turn when it failed or was stopped. */
  onRetry?: () => void;
  feedbackOff: boolean;
  onFollowUp: (text: string) => void;
  onFeedbackOff: () => void;
  /** Debug mode: show this turn's details in the panel. */
  inspected?: boolean;
  onInspect?: () => void;
}

function AssistantMessage({ turn, busy, onRetry, feedbackOff, onFollowUp, onFeedbackOff, inspected,
                            onInspect }: AssistantProps) {
  const open = turn.status === "pending" || turn.status === "searching" || turn.status === "writing";
  const a = turn.answer;
  return (
    <article className={`msg msg--assistant${inspected ? " is-inspected" : ""}`} aria-busy={open}>
      <div className="msg__head">
        <p className="label">Assistant</p>
        {onInspect && hasDebug(a) && (
          <button type="button" className="msg__inspect" onClick={onInspect} aria-pressed={inspected}>
            {inspected ? "Showing details" : "Details"}
          </button>
        )}
      </div>

      {a && <FilterChips applied={a.applied_filters} />}

      {open && !turn.text && (
        <p className="msg__status" role="status">
          {STAGE_TEXT[turn.status]}
          <span className="msg__dots" aria-hidden="true" />
        </p>
      )}

      {turn.text && (
        <div className="msg__text">
          {turn.text}
          {open && <span className="msg__caret" aria-hidden="true" />}
        </div>
      )}

      {turn.status === "stopped" && <p className="msg__note">Stopped.</p>}
      {turn.status === "error" && turn.error && (
        <p className="msg__error" role="alert">
          {turn.error.message}
        </p>
      )}

      {onRetry && (
        <button type="button" className="btn btn--outline msg__retry" onClick={onRetry}>
          Retry
        </button>
      )}

      {a && a.products.length > 0 && <ProductCards products={a.products} />}
      {a?.comparison && <ComparisonTable table={a.comparison} products={a.products} />}

      {a?.follow_up && (
        <button type="button" className="msg__followup" disabled={busy} onClick={() => onFollowUp(a.follow_up!)}>
          {a.follow_up}
        </button>
      )}

      {turn.status === "done" && turn.requestId && (
        <Feedback requestId={turn.requestId} disabled={feedbackOff} onUnavailable={onFeedbackOff} />
      )}
    </article>
  );
}

interface Props {
  turns: Turn[];
  busy: boolean;
  canRetry: boolean;
  onRetry: () => void;
  feedbackOff: boolean;
  onFollowUp: (text: string) => void;
  onFeedbackOff: () => void;
  inspectedId?: string | null;
  onInspect?: (id: string) => void;
}

export function Thread({ turns, busy, canRetry, onRetry, feedbackOff, onFollowUp, onFeedbackOff, inspectedId,
                         onInspect }: Props) {
  const last = turns[turns.length - 1];
  const lastText = last && last.role === "assistant" ? last.text.length : 0;
  const lastStatus = last && last.role === "assistant" ? last.status : "";

  // Follow the answer as it streams in, and once more when the cards arrive. Scrolling the
  // page to its very end keeps the last message above the sticky composer, not behind it.
  useEffect(() => {
    window.scrollTo({ top: document.documentElement.scrollHeight, behavior: "smooth" });
  }, [turns.length, lastText, lastStatus]);

  return (
    <div className="thread">
      {turns.map((t, i) =>
        t.role === "user" ? (
          <article key={t.id} className="msg msg--user">
            <p className="label">You</p>
            <p className="msg__question">{t.text}</p>
          </article>
        ) : (
          <AssistantMessage key={t.id} turn={t} busy={busy} feedbackOff={feedbackOff}
                            onRetry={canRetry && i === turns.length - 1 ? onRetry : undefined}
                            onFollowUp={onFollowUp} onFeedbackOff={onFeedbackOff}
                            inspected={inspectedId === t.id}
                            onInspect={onInspect ? () => onInspect(t.id) : undefined} />
        ),
      )}
    </div>
  );
}