import { useEffect, useRef } from "react";

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
  feedbackOff: boolean;
  onFollowUp: (text: string) => void;
  onFeedbackOff: () => void;
}

function AssistantMessage({ turn, busy, feedbackOff, onFollowUp, onFeedbackOff }: AssistantProps) {
  const open = turn.status === "pending" || turn.status === "searching" || turn.status === "writing";
  const a = turn.answer;
  return (
    <article className="msg msg--assistant" aria-busy={open}>
      <p className="label">Assistant</p>

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
  feedbackOff: boolean;
  onFollowUp: (text: string) => void;
  onFeedbackOff: () => void;
}

export function Thread({ turns, busy, feedbackOff, onFollowUp, onFeedbackOff }: Props) {
  const end = useRef<HTMLDivElement>(null);
  const last = turns[turns.length - 1];
  const lastText = last && last.role === "assistant" ? last.text.length : 0;
  const lastStatus = last && last.role === "assistant" ? last.status : "";

  // Follow the answer as it streams in, and once more when the cards arrive.
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length, lastText, lastStatus]);

  return (
    <div className="thread">
      {turns.map((t) =>
        t.role === "user" ? (
          <article key={t.id} className="msg msg--user">
            <p className="label">You</p>
            <p className="msg__question">{t.text}</p>
          </article>
        ) : (
          <AssistantMessage key={t.id} turn={t} busy={busy} feedbackOff={feedbackOff}
                            onFollowUp={onFollowUp} onFeedbackOff={onFeedbackOff} />
        ),
      )}
      <div ref={end} />
    </div>
  );
}