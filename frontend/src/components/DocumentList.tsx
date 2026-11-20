import { DocumentInfo, deleteDocument } from "../api";

export default function DocumentList({ docs, onChanged }: { docs: DocumentInfo[]; onChanged: () => void }) {
  if (!docs.length) return <p className="muted">No documents yet.</p>;
  return (
    <ul className="docs">
      {docs.map((d) => (
        <li key={d.id}>
          <div>
            <strong>{d.filename}</strong>
            <div className="muted">{d.pages} pages · {d.chunks} chunks</div>
          </div>
          <button
            className="link"
            aria-label={`Delete ${d.filename}`}
            onClick={async () => { await deleteDocument(d.id); onChanged(); }}
          >
            Delete
          </button>
        </li>
      ))}
    </ul>
  );
}
