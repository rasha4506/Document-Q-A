import { useCallback, useEffect, useState } from "react";
import { DocumentInfo, listDocuments } from "./api";
import Chat from "./components/Chat";
import DocumentList from "./components/DocumentList";
import UploadBox from "./components/UploadBox";

export default function App() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError(null);
    } catch {
      setError("Cannot reach the API. Is the backend running on port 8000?");
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  return (
    <div className="layout">
      <aside>
        <h1>Document Q&amp;A</h1>
        <p className="muted">Upload PDFs, then ask questions. Answers come only from your documents.</p>
        <UploadBox onUploaded={refresh} />
        <DocumentList docs={docs} onChanged={refresh} />
        {error && <p className="error">{error}</p>}
      </aside>
      <main>
        <Chat hasDocs={docs.length > 0} />
      </main>
    </div>
  );
}
