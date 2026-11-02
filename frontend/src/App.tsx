import { useState } from "react";

import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { Header } from "./components/Header";
import { RetryToast } from "./components/RetryToast";
import { Sidebar } from "./components/Sidebar";
import { Thread } from "./components/Thread";
import { useChat } from "./state/useChat";
import { useHealth } from "./state/useHealth";
import "./App.css";

export default function App() {
  const chat = useChat();
  const health = useHealth();
  // Once the server says feedback is off (no tracing configured), hide the thumbs everywhere.
  const [feedbackOff, setFeedbackOff] = useState(false);
  // The sidebar is always shown on wide screens; on narrow ones it's a drawer.
  const [historyOpen, setHistoryOpen] = useState(false);
  const empty = chat.conversation.turns.length === 0;

  return (
    <div className="app">
      <Header health={health} onToggleHistory={() => setHistoryOpen((o) => !o)} />
      <div className="app__body">
        <Sidebar
          open={historyOpen}
          sessions={chat.sessions}
          currentId={chat.conversation.sessionId}
          available={chat.historyAvailable}
          onNewChat={() => { chat.newChat(); setHistoryOpen(false); }}
          onOpen={(id) => { chat.openSession(id); setHistoryOpen(false); }}
          onDelete={chat.deleteSession}
          onClose={() => setHistoryOpen(false)}
        />
        <div className="app__chat">
          <main className="app__main">
            <div className="app__column">
              {empty ? (
                <EmptyState onPick={chat.send} />
              ) : (
                <Thread
                  turns={chat.conversation.turns}
                  busy={chat.busy}
                  canRetry={chat.canRetry}
                  onRetry={chat.retry}
                  feedbackOff={feedbackOff}
                  onFollowUp={chat.send}
                  onFeedbackOff={() => setFeedbackOff(true)}
                />
              )}
            </div>
          </main>
          <footer className="app__footer">
            <div className="app__column">
              <RetryToast notice={chat.notice} />
              <Composer busy={chat.busy} onSend={chat.send} onStop={chat.stop} />
              <p className="app__disclaimer">
                Unofficial portfolio demo built on public catalog data. Not affiliated with or endorsed by the retailer.
              </p>
            </div>
          </footer>
        </div>
      </div>
    </div>
  );
}