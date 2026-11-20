import { useRef, useState } from "react";
import { uploadDocument } from "../api";

export default function UploadBox({ onUploaded }: { onUploaded: () => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [drag, setDrag] = useState(false);
  const [msg, setMsg] = useState<{ text: string; kind: "ok" | "error" } | null>(null);

  async function handle(files: FileList | null) {
    if (!files?.length) return;
    setBusy(true);
    setMsg(null);
    for (const file of Array.from(files)) {
      try {
        const r = await uploadDocument(file);
        setMsg(
          r.duplicate
            ? { text: `${file.name} was already uploaded`, kind: "ok" }
            : { text: `Indexed ${file.name}: ${r.document.pages} pages, ${r.document.chunks} chunks`, kind: "ok" }
        );
      } catch (e) {
        setMsg({ text: `${file.name}: ${(e as Error).message}`, kind: "error" });
      }
    }
    setBusy(false);
    if (input.current) input.current.value = "";
    onUploaded();
  }

  return (
    <div
      className={`upload ${drag ? "drag" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); handle(e.dataTransfer.files); }}
    >
      <input ref={input} type="file" accept="application/pdf" multiple hidden onChange={(e) => handle(e.target.files)} />
      <button disabled={busy} onClick={() => input.current?.click()}>
        {busy ? "Indexing…" : "Choose PDF"}
      </button>
      <span className="muted">or drop files here</span>
      {msg && <p className={msg.kind === "error" ? "error" : "ok"}>{msg.text}</p>}
    </div>
  );
}
