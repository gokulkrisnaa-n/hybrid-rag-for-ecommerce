import { timeAgo } from "../format";
import type { SessionSummary } from "../state/storage";
import "./Sidebar.css";

interface Props {
  open: boolean;                        // only matters on narrow screens, where it's a drawer
  sessions: SessionSummary[];
  currentId: string;
  available: boolean;                   // browser storage usable
  onNewChat: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
  onClose: () => void;
}

export function Sidebar({ open, sessions, currentId, available, onNewChat, onOpen, onDelete, onClose }: Props) {
  return (
    <>
      <div className={`sidebar__scrim${open ? " is-open" : ""}`} onClick={onClose} aria-hidden="true" />
      <aside className={`sidebar${open ? " is-open" : ""}`} aria-label="Conversations">
        <button type="button" className="btn btn--outline sidebar__new" onClick={onNewChat}>
          New chat
        </button>

        <p className="label sidebar__heading">Recent</p>
        {sessions.length === 0 ? (
          <p className="sidebar__empty">
            {available ? "Your conversations will appear here." : "History is off: this browser blocks storage."}
          </p>
        ) : (
          <ul className="sidebar__list">
            {sessions.map((s) => (
              <li key={s.id} className={s.id === currentId ? "sidebar__item is-current" : "sidebar__item"}>
                <button type="button" className="sidebar__open" onClick={() => onOpen(s.id)}
                        aria-current={s.id === currentId ? "page" : undefined}>
                  <span className="sidebar__title">{s.title}</span>
                  <span className="sidebar__time">{timeAgo(s.updatedAt)}</span>
                </button>
                <button type="button" className="sidebar__delete" onClick={() => onDelete(s.id)}
                        aria-label={`Delete "${s.title}"`}>
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        <p className="sidebar__note">Saved in this browser only.</p>
      </aside>
    </>
  );
}