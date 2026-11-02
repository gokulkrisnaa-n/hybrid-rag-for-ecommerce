/** Display formatting: prices, sizes and the applied-filter chips. Pure, so it's unit-tested. */

import type { ProductCard } from "./api/types";

const toNumber = (v: string | number | null | undefined): number | null => {
  if (v === null || v === undefined || v === "") return null;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? n : null;
};

/** "$1,495.00"; whole amounts in chips drop the cents ("$300"). */
export function formatPrice(v: string | number, { cents = true } = {}): string {
  const n = toNumber(v);
  if (n === null) return String(v);
  const whole = Number.isInteger(n) && !cents;
  return `$${n.toLocaleString("en-US", { minimumFractionDigits: whole ? 0 : 2, maximumFractionDigits: whole ? 0 : 2 })}`;
}

export interface PriceView {
  current: string;          // "$97.00" or "$90.00 – $110.00" when variants differ
  was: string | null;       // the compare-at price when on sale
  discountPercent: number | null;
}

export function priceView(p: ProductCard): PriceView {
  const lo = toNumber(p.price_min);
  const hi = toNumber(p.price_max);
  const current = lo !== null && hi !== null && hi > lo
    ? `${formatPrice(lo)} – ${formatPrice(hi)}`
    : formatPrice(p.price_min);
  const compare = toNumber(p.compare_at_price);
  if (!p.on_sale || compare === null || lo === null || compare <= lo) {
    return { current, was: null, discountPercent: null };
  }
  return { current, was: formatPrice(compare), discountPercent: Math.round((1 - lo / compare) * 100) };
}

/** "S · M · L"; long size runs are cut: "7 · 7.5 · 8 · 8.5 · 9 · 9.5 +6". */
export function sizesText(sizes: string[], max = 6): string {
  if (sizes.length <= max) return sizes.join(" · ");
  return `${sizes.slice(0, max).join(" · ")} +${sizes.length - max}`;
}

const titleCase = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());
const list = (v: unknown): string[] => (Array.isArray(v) ? v.map(String) : []);

const GENDER: Record<string, string> = { mens: "Men", womens: "Women", kids: "Kids", unisex: "Unisex" };

/**
 * The search's applied filters as short chips, in a stable order. Unknown keys are
 * ignored, so a new server-side filter never shows up as raw JSON.
 */
export function filterChips(applied: Record<string, unknown> | null | undefined): { text: string; negative?: boolean }[] {
  if (!applied) return [];
  const chips: { text: string; negative?: boolean }[] = [];
  for (const c of list(applied.categories)) chips.push({ text: titleCase(c) });
  for (const c of list(applied.subcategories)) chips.push({ text: titleCase(c) });
  for (const g of list(applied.genders)) chips.push({ text: GENDER[g] ?? titleCase(g) });
  for (const b of list(applied.brands)) chips.push({ text: b });

  const lo = toNumber(applied.min_price as string | number | null);
  const hi = toNumber(applied.max_price as string | number | null);
  if (lo !== null && hi !== null) chips.push({ text: `${formatPrice(lo, { cents: false })} – ${formatPrice(hi, { cents: false })}` });
  else if (hi !== null) chips.push({ text: `Under ${formatPrice(hi, { cents: false })}` });
  else if (lo !== null) chips.push({ text: `From ${formatPrice(lo, { cents: false })}` });

  const sizes = list(applied.sizes);
  if (sizes.length) chips.push({ text: `Size ${sizes.join(", ")}` });
  if (applied.on_sale_only === true) chips.push({ text: "On sale" });
  for (const b of list(applied.exclude_brands)) chips.push({ text: `Not ${b}`, negative: true });
  for (const g of list(applied.exclude_genders)) chips.push({ text: `No ${(GENDER[g] ?? g).toLowerCase()}`, negative: true });
  return chips;
}

/** "Just now", "5 min ago", "3 h ago", "Yesterday", "Oct 30". */
export function timeAgo(ts: number, now: number = Date.now()): string {
  const minutes = Math.floor((now - ts) / 60_000);
  if (minutes < 1) return "Just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  if (hours < 48) return "Yesterday";
  return new Date(ts).toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

/** The first 8 characters, enough to find a trace: "7f3a9c1e…". */
export const shortId = (id: string): string => (id.length > 10 ? `${id.slice(0, 8)}…` : id);