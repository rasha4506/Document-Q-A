const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export interface DocumentInfo { id: string; filename: string; pages: number; chunks: number; uploaded_at: string }
export interface Source { file: string; page: number; snippet: string; score: number }
export interface AskResponse { answer: string; found: boolean; sources: Source[] }

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON error */ }
    throw new Error(detail);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

export const listDocuments = () => request<DocumentInfo[]>("/documents");

export function uploadDocument(file: File) {
  const body = new FormData();
  body.append("file", file);
  return request<{ document: DocumentInfo; duplicate: boolean }>("/upload", { method: "POST", body });
}

export const deleteDocument = (id: string) => request<void>(`/documents/${id}`, { method: "DELETE" });

export const ask = (question: string) =>
  request<AskResponse>("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
