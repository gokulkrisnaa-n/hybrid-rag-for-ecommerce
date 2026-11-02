import { useState } from "react";

import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { Header } from "./components/Header";
import { Thread } from "./components/Thread";
import { useChat } from "./state/useChat";
import "./App.css";

export default function App() {
  const { conversation, busy, send, stop, newChat } = useChat();
  // Once the server says feedback is off (no tracing configured), hide the thumbs everywhere.
  const [feedbackOff, setFeedbackOff] = useState(false);
  const empty = conversation.turns.length === 0;

  return (
    <div className="app">
      <Header onNewChat={newChat} />
      <main className="app__main">
        <div className="app__column">
          {empty ? (
            <EmptyState onPick={send} />
          ) : (
            <Thread turns={conversation.turns} busy={busy} feedbackOff={feedbackOff}
                    onFollowUp={send} onFeedbackOff={() => setFeedbackOff(true)} />
          )}
        </div>
      </main>
      <footer className="app__footer">
        <div className="app__column">
          <Composer busy={busy} onSend={send} onStop={stop} />
          <p className="app__disclaimer">
            Unofficial portfolio demo built on public catalog data. Not affiliated with or endorsed by the retailer.
          </p>
        </div>
      </footer>
    </div>
  );
}