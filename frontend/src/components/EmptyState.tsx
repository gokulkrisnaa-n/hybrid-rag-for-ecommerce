import "./EmptyState.css";

const SUGGESTIONS = [
  "A warm winter jacket under $300",
  "Black sneakers for everyday wear",
  "Compare the Samba and the Gazelle",
  "A hoodie for a kid, as a gift",
];

interface Props {
  onPick: (text: string) => void;
}

export function EmptyState({ onPick }: Props) {
  return (
    <section className="empty">
      <p className="label">Shopping Assistant</p>
      <h1 className="empty__title">What are you looking for today?</h1>
      <p className="empty__lede">
        Describe what you need in your own words: a style, a budget, a brand to avoid. Answers use only products
        in the catalog, with live prices and sizes.
      </p>
      <div className="empty__grid">
        {SUGGESTIONS.map((s) => (
          <button key={s} type="button" className="empty__chip" onClick={() => onPick(s)}>
            {s}
          </button>
        ))}
      </div>
    </section>
  );
}