import { type FormEvent, type KeyboardEvent, useEffect, useRef, useState } from "react";

import { MAX_MESSAGE_CHARS } from "../api/types";
import "./Composer.css";

interface Props {
  busy: boolean;
  onSend: (text: string) => void;
  onStop: () => void;
}

export function Composer({ busy, onSend, onStop }: Props) {
  const [text, setText] = useState("");
  const box = useRef<HTMLTextAreaElement>(null);
  const tooLong = text.length > MAX_MESSAGE_CHARS;
  const canSend = !busy && text.trim().length > 0 && !tooLong;

  // Grow with the text, up to a limit.
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text]);

  // Esc stops a streaming answer from anywhere on the page.
  useEffect(() => {
    if (!busy) return;
    const onKey = (e: globalThis.KeyboardEvent) => e.key === "Escape" && onStop();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onStop]);

  const submit = (e?: FormEvent) => {
    e?.preventDefault();
    if (!canSend) return;
    onSend(text);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="composer-input" className="visually-hidden">
        Your message
      </label>
      <textarea
        id="composer-input"
        ref={box}
        className="composer__input"
        rows={1}
        value={text}
        placeholder="Ask about a product, a style or a budget"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        aria-invalid={tooLong}
      />
      <div className="composer__row">
        <span className={`composer__hint${tooLong ? " composer__hint--over" : ""}`}>
          {tooLong
            ? `${text.length} / ${MAX_MESSAGE_CHARS} characters`
            : busy
              ? "Esc to stop"
              : "Enter to send · Shift + Enter for a new line"}
        </span>
        {busy ? (
          <button type="button" className="btn btn--outline" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button type="submit" className="btn" disabled={!canSend}>
            Send
          </button>
        )}
      </div>
    </form>
  );
}