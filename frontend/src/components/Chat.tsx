import { FormEvent, useEffect, useRef, useState } from "react";
import { AskResponse, ask } from "../api";
import SourcesList from "./SourcesList";

interface Turn { question: string; response?: AskResponse; error?: string }

export default function Chat({ hasDocs }: { hasDocs: boolean }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth" }); }, [turns]);

  async function submit(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setQuestion("");
    setBusy(true);
    setTurns((t) => [...t, { question: q }]);
    try {
      const response = await ask(q);
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, response } : x)));
    } catch (err) {
      setTurns((t) => t.map((x, i) => (i === t.length - 1 ? { ...x, error: (err as Error).message } : x)));
    }
    setBusy(false);
  }

  return (
    <div className="chat">
      <div className="messages">
        {!turns.length && <p className="muted center">{hasDocs ? "Ask a question about your documents." : "Upload a PDF to get started."}</p>}
        {turns.map((t, i) => (
          <div key={i}>
            <div className="bubble user">{t.question}</div>
            <div className={`bubble bot ${t.response && !t.response.found ? "notfound" : ""}`}>
              {t.error ? <span className="error">{t.error}</span> : t.response ? (
                <>
                  <p>{t.response.answer}</p>
                  <SourcesList sources={t.response.sources} />
                </>
              ) : <span className="muted">Thinking…</span>}
            </div>
          </div>
        ))}
        <div ref={end} />
      </div>
      <form onSubmit={submit}>
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder={hasDocs ? "e.g. How many attention heads are used?" : "Upload a document first"}
          disabled={!hasDocs || busy}
          aria-label="Question"
        />
        <button type="submit" disabled={!hasDocs || busy || !question.trim()}>Ask</button>
      </form>
    </div>
  );
}
