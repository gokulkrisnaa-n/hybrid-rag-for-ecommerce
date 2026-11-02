import { useEffect, useRef } from "react";

import type { AssistantTurn, Turn } from "../state/conversation";
import "./Thread.css";

const STAGE_TEXT: Record<string, string> = {
  pending: "Reading your question",
  searching: "Searching the catalog",
  writing: "Writing",
};

function formatPrice(value: string): string {
  const n = Number(value);
  return Number.isFinite(n) ? `$${n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : value;
}

function AssistantMessage({ turn }: { turn: AssistantTurn }) {
  const open = turn.status === "pending" || turn.status === "searching" || turn.status === "writing";
  return (
    <article className="msg msg--assistant" aria-busy={open}>
      <p className="label">Assistant</p>

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

      {/* Plain list for now; product cards come next. */}
      {turn.answer && turn.answer.products.length > 0 && (
        <ul className="msg__products">
          {turn.answer.products.map(({ product, reason }) => (
            <li key={product.product_id}>
              <a href={product.url} target="_blank" rel="noreferrer">
                {product.title}
              </a>
              <span className="msg__price">{formatPrice(product.price_min)}</span>
              <span className="msg__reason">{reason}</span>
            </li>
          ))}
        </ul>
      )}
    </article>
  );
}

interface Props {
  turns: Turn[];
}

export function Thread({ turns }: Props) {
  const end = useRef<HTMLDivElement>(null);
  const last = turns[turns.length - 1];
  const lastText = last && last.role === "assistant" ? last.text.length : 0;

  // Follow the answer as it streams in.
  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [turns.length, lastText]);

  return (
    <div className="thread">
      {turns.map((t) =>
        t.role === "user" ? (
          <article key={t.id} className="msg msg--user">
            <p className="label">You</p>
            <p className="msg__question">{t.text}</p>
          </article>
        ) : (
          <AssistantMessage key={t.id} turn={t} />
        ),
      )}
      <div ref={end} />
    </div>
  );
}