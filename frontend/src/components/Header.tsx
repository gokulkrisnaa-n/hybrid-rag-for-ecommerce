import type { Health } from "../api/health";
import "./Header.css";

const HEALTH_TEXT: Record<Health, string> = {
  checking: "Connecting",
  ok: "Online",
  degraded: "Catalog offline",
  offline: "API offline",
};

interface Props {
  health: Health;
  onToggleHistory: () => void;
}

export function Header({ health, onToggleHistory }: Props) {
  return (
    <header className="header">
      <nav className="header__side">
        <button type="button" className="link-btn header__history" onClick={onToggleHistory}>
          History
        </button>
      </nav>
      <div className="header__title">Shopping Assistant</div>
      <div className="header__side header__side--right">
        <span className={`health health--${health}`} title={`Server status: ${HEALTH_TEXT[health]}`}>
          <span className="health__dot" aria-hidden="true" />
          <span className="health__text">{HEALTH_TEXT[health]}</span>
        </span>
      </div>
    </header>
  );
}