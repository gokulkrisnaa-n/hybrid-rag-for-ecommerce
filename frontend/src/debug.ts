/**
 * Reading a turn's debug data (sent when the request asks for "debug": true) into
 * what the debug panel shows. The shape comes from app/chat/service.py and
 * app/query/search.py; every field is optional here, so an older or newer server
 * never breaks the panel.
 */

import type { ChatAnswer } from "./api/types";

type Dict = Record<string, unknown>;
const obj = (v: unknown): Dict => (v && typeof v === "object" && !Array.isArray(v) ? (v as Dict) : {});
const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);

export interface TimelineRow {
  label: string;
  start: number;          // ms from the start of the turn
  duration: number;       // ms; 0 for a point in time (first words)
  kind: "llm" | "retrieval" | "wait" | "total" | "marker";
}

/**
 * The turn as a waterfall. Positions follow the pipeline's order (understand, then
 * the retrievers, then loading details, then generation); the server reports
 * durations, not start times, so positions are approximate.
 */
export function timeline(answer: ChatAnswer): TimelineRow[] {
  const t = answer.timings_ms ?? {};
  const st = obj(obj(answer.debug).search_timings);
  const understand = num(st.understand) ?? 0;
  const embedWait = num(st.embed_query) ?? 0;
  const bm25 = num(st.bm25_search);
  const dense = num(st.dense_search);
  const search = num(t.search) ?? 0;
  const context = num(t.context) ?? 0;
  const generate = num(t.generate);
  const first = num(t.first_token);
  const total = num(t.total);

  const rows: TimelineRow[] = [];
  if (understand) rows.push({ label: "Understand", start: 0, duration: understand, kind: "llm" });
  if (embedWait) rows.push({ label: "Embed (wait)", start: understand, duration: embedWait, kind: "wait" });
  const retrievalStart = understand + embedWait;
  if (bm25 !== null) rows.push({ label: "BM25", start: retrievalStart, duration: bm25, kind: "retrieval" });
  if (dense !== null) rows.push({ label: "Vector", start: retrievalStart, duration: dense, kind: "retrieval" });
  if (context) rows.push({ label: "Load details", start: search, duration: context, kind: "retrieval" });
  if (generate !== null) rows.push({ label: "Generate", start: search + context, duration: generate, kind: "llm" });
  if (first !== null) rows.push({ label: "First words", start: first, duration: 0, kind: "marker" });
  if (total !== null) rows.push({ label: "Total", start: 0, duration: total, kind: "total" });
  return rows;
}

export interface TokenRow {
  step: string;
  input: number;
  cached: number;
  output: number;
}

export function tokens(answer: ChatAnswer): TokenRow[] {
  const d = obj(answer.debug);
  const rows: TokenRow[] = [];
  for (const [step, raw] of [["Understand", obj(obj(d.qu).tokens)], ["Generate", obj(d.tokens)]] as const) {
    const input = num(raw.input);
    if (input === null) continue;
    rows.push({ step, input, cached: num(raw.cached) ?? 0, output: num(raw.output) ?? 0 });
  }
  return rows;
}

export interface QuerySummary {
  used: boolean;
  reason: string | null;        // why it was skipped or failed
  searchText: string | null;
  plan: Dict | null;
  inferred: Dict;
  dropped: unknown[];
  relaxed: string | null;
  prompt: string | null;
  cached: boolean;
}

export function querySummary(answer: ChatAnswer): QuerySummary {
  const qu = obj(obj(answer.debug).qu);
  const reason = typeof qu.reason === "string" ? qu.reason : null;
  return {
    used: qu.used === true,
    reason: qu.error ? `${reason ?? "error"}: ${String(qu.error)}` : reason,
    searchText: typeof qu.search_text === "string" ? qu.search_text : null,
    plan: qu.plan ? obj(qu.plan) : null,
    inferred: obj(qu.inferred_filters),
    dropped: Array.isArray(qu.dropped) ? qu.dropped : [],
    relaxed: qu.relaxed ? String(obj(qu.relaxed).reason ?? "yes") : null,
    prompt: typeof qu.prompt === "string" ? qu.prompt : null,
    cached: qu.cached === true,
  };
}

export function hasDebug(answer: ChatAnswer | null): answer is ChatAnswer {
  return !!answer && !!answer.debug && Object.keys(answer.debug).length > 0;
}

/** Langfuse trace link (the trace id is the request id), or null when no project URL is set. */
export function traceUrlFor(requestId: string, projectUrl: string | null | undefined): string | null {
  if (!projectUrl) return null;
  return `${projectUrl.replace(/\/+$/, "")}/traces/${encodeURIComponent(requestId)}`;
}

/** The link for this build: VITE_LANGFUSE_PROJECT_URL is read at build time. */
export const traceUrl = (requestId: string): string | null =>
  traceUrlFor(requestId, import.meta.env.VITE_LANGFUSE_PROJECT_URL);