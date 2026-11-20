import { useState } from "react";
import { Source } from "../api";

export default function SourcesList({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!sources.length) return null;
  return (
    <div className="sources">
      <span className="muted">Sources:</span>
      {sources.map((s, i) => (
        <div key={i} className="source">
          <button className="chip" onClick={() => setOpen(open === i ? null : i)} aria-expanded={open === i}>
            {s.file} · p.{s.page}
          </button>
          {open === i && <blockquote>{s.snippet}</blockquote>}
        </div>
      ))}
    </div>
  );
}
