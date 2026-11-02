import { filterChips } from "../format";
import "./FilterChips.css";

/** What the search filtered on, so the shopper can see why the results look this way. */
export function FilterChips({ applied }: { applied?: Record<string, unknown> }) {
  const chips = filterChips(applied);
  if (chips.length === 0) return null;
  return (
    <ul className="chips" aria-label="Filters applied">
      {chips.map((c) => (
        <li key={c.text} className={c.negative ? "chips__chip chips__chip--negative" : "chips__chip"}>
          {c.text}
        </li>
      ))}
    </ul>
  );
}