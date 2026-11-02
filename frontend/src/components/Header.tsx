import "./Header.css";

interface Props {
  onNewChat: () => void;
}

export function Header({ onNewChat }: Props) {
  return (
    <header className="header">
      <nav className="header__side">
        <button type="button" className="link-btn" onClick={onNewChat}>
          New chat
        </button>
      </nav>
      <div className="header__title">Shopping Assistant</div>
      <div className="header__side header__side--right">
        <span className="header__tag">Demo</span>
      </div>
    </header>
  );
}