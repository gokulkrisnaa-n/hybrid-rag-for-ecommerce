import { useState } from "react";

import { type TimelineRow, hasDebug, querySummary, timeline, tokens, traceUrl } from "../debug";
import { filterChips } from "../format";
import type { AssistantTurn } from "../state/conversation";
import "./DebugPanel.css";

const ms = (v: number) => `${Math.round(v).toLocaleString("en-US")} ms`;

function Timeline({ rows }: { rows: TimelineRow[] }) {
  const end = Math.max(1, ...rows.map((r) => r.start + r.duration));
  return (
    <div className="dbg__timeline">
      {rows.map((r) => (
        <div key={r.label} className="dbg__row">
          <span className="dbg__rowlabel">{r.label}</span>
          <span className="dbg__track">
            {r.kind === "marker" ? (
              <i className="dbg__marker" style={{ left: `${(r.start / end) * 100}%` }} />
            ) : (
              <i className={`dbg__bar dbg__bar--${r.kind}`}
                 style={{ left: `${(r.start / end) * 100}%`, width: `max(2px, ${(r.duration / end) * 100}%)` }} />
            )}
          </span>
          <span className="dbg__num">{ms(r.kind === "marker" ? r.start : r.duration)}</span>
        </div>
      ))}
      <p className="dbg__hint">Durations are measured; positions follow the pipeline's order.</p>
    </div>
  );
}

interface Props {
  turn: AssistantTurn | null;
  onClose: () => void;
}

export function DebugPanel({ turn, onClose }: Props) {
  const [showPlan, setShowPlan] = useState(false);
  const answer = turn?.answer ?? null;

  let body;
  if (!hasDebug(answer)) {
    body = (
      <p className="dbg__empty">
        {turn && turn.status !== "done"
          ? "Details appear when the answer is finished."
          : "Send a message with debug on to see how it was answered: the query plan, retrieval, timings and tokens."}
      </p>
    );
  } else {
    const q = querySummary(answer);
    const tok = tokens(answer);
    const link = traceUrl(answer.request_id);
    const d = answer.debug as Record<string, unknown>;
    const retrieved = Array.isArray(d.retrieved) ? d.retrieved.length : null;
    const inferred = filterChips(q.inferred);

    body = (
      <>
        <section className="dbg__section">
          <h3 className="dbg__h">Timeline</h3>
          <Timeline rows={timeline(answer)} />
        </section>

        <section className="dbg__section">
          <h3 className="dbg__h">Query understanding {q.prompt && <span className="dbg__tag">{q.prompt}</span>}</h3>
          {q.used ? (
            <dl className="dbg__kv">
              <dt>Search text</dt><dd>{q.searchText ?? "—"}</dd>
              <dt>Inferred filters</dt><dd>{inferred.length ? inferred.map((c) => c.text).join(", ") : "None"}</dd>
              {q.relaxed && (<><dt>Relaxed</dt><dd>{q.relaxed}</dd></>)}
              {q.dropped.length > 0 && (<><dt>Dropped</dt><dd>{q.dropped.map(String).join(", ")}</dd></>)}
              {q.cached && (<><dt>Plan</dt><dd>From cache</dd></>)}
            </dl>
          ) : (
            <p className="dbg__text">Not used: {q.reason ?? "unknown"}. The raw message was searched.</p>
          )}
          {q.plan && (
            <>
              <button type="button" className="dbg__toggle" onClick={() => setShowPlan((v) => !v)}
                      aria-expanded={showPlan}>
                {showPlan ? "Hide" : "Show"} the full plan
              </button>
              {showPlan && <pre className="dbg__pre">{JSON.stringify(q.plan, null, 2)}</pre>}
            </>
          )}
        </section>

        <section className="dbg__section">
          <h3 className="dbg__h">Answer</h3>
          <dl className="dbg__kv">
            <dt>Kind</dt><dd>{answer.kind}{answer.source === "template" ? " (fixed reply)" : ""}</dd>
            <dt>Picks</dt><dd>{answer.products.length}{retrieved !== null ? ` of ${retrieved} retrieved` : ""}</dd>
            {typeof d.model === "string" && (<><dt>Model</dt><dd>{d.model}{typeof d.prompt === "string" ? ` · ${d.prompt}` : ""}</dd></>)}
            <dt>Checks</dt>
            <dd>
              {answer.warnings.length === 0 ? "Passed" : (
                <ul className="dbg__warnings">{answer.warnings.map((w) => <li key={w}>{w}</li>)}</ul>
              )}
            </dd>
          </dl>
        </section>

        {tok.length > 0 && (
          <section className="dbg__section">
            <h3 className="dbg__h">Tokens</h3>
            <table className="dbg__table">
              <thead><tr><th /><th>In</th><th>Cached</th><th>Out</th></tr></thead>
              <tbody>
                {tok.map((t) => (
                  <tr key={t.step}>
                    <th scope="row">{t.step}</th>
                    <td>{t.input.toLocaleString("en-US")}</td>
                    <td>{t.cached.toLocaleString("en-US")}</td>
                    <td>{t.output.toLocaleString("en-US")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>
        )}

        <section className="dbg__section">
          <h3 className="dbg__h">Trace</h3>
          <p className="dbg__text dbg__mono">{answer.request_id}</p>
          {link && <a className="dbg__link" href={link} target="_blank" rel="noreferrer">Open in Langfuse ↗</a>}
        </section>
      </>
    );
  }

  return (
    <aside className="dbg" aria-label="Debug details">
      <div className="dbg__head">
        <p className="label">Debug</p>
        <button type="button" className="dbg__close" onClick={onClose} aria-label="Close debug panel">×</button>
      </div>
      {body}
      <p className="dbg__foot">For developers. Shoppers never see this panel.</p>
    </aside>
  );
}