# Document Q&A with RAG

Upload PDFs, ask questions in plain language, get an answer **using only your documents** plus the file and page it came from. Built with plain Python and direct API calls (no LangChain / LlamaIndex) so every step is visible.

```
Ingestion:  PDF → parse (PyMuPDF) → clean → chunk (token window, page-bounded) → embed (bge-small, local) → Chroma
Query:      question → embed → top-k chunks → similarity floor → LLM (JSON output) → answer + sources
```

**Stack:** Python 3.12 · FastAPI · PyMuPDF · fastembed (`BAAI/bge-small-en-v1.5`, runs locally, free) · Chroma · any OpenAI-compatible chat API (Groq free tier by default) · React + Vite + TypeScript · pytest

## Evaluation: which chunk size retrieves best?

The part that matters. [`eval/questions.json`](eval/questions.json) has 26 questions over two public papers (*Attention Is All You Need*, the original *RAG* paper; 34 pages): 23 with a known answer and page, and 3 unanswerable. `eval/run_eval.py` re-indexes the documents for each chunking config and checks whether the correct **file + page** is among the top-k retrieved chunks.

| chunk size (tokens) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |
|---|---|---|---|---|---|---|
| **100** | **20** | 353 | **87%** | **100%** | **100%** | **0.93** |
| 100 | 0 | 291 | 78% | 100% | 100% | 0.88 |
| 200 | 40 | 178 | 87% | 91% | 96% | 0.90 |
| 200 | 0 | 152 | 70% | 87% | 96% | 0.80 |
| 300 | 60 | 123 | 65% | 83% | 87% | 0.75 |
| 300 | 0 | 109 | 57% | 91% | 96% | 0.73 |
| 500 | 100 | 78 | 61% | 83% | 96% | 0.73 |
| 500 | 0 | 70 | 52% | 83% | 87% | 0.67 |

(Reproduce with `python eval/run_eval.py`; add `--misses` to see which questions failed.)

**What this shows**
- **Smaller chunks retrieve better here.** 100-token chunks reach 100% hit@3; 500-token chunks 83%. These are factual lookups ("what is d_ff?", "what beam size?"), and a small chunk keeps the answer sentence from being diluted by unrelated text in the same embedding.
- **20% overlap helps at every size on hit@1/MRR** (e.g. 100 tokens: hit@1 78% → 87%; 200 tokens: 70% → 87%), because answers that straddle a chunk boundary stay intact. Hit@3/hit@5 are less consistent (at 300 tokens, overlap *lowered* hit@3 from 91% to 83%), so I'd read overlap as a ranking improvement, not a guarantee.
- The remaining misses at larger sizes are mostly "explanatory" questions where several pages discuss the same concept (e.g. which pages describe RAG-Sequence vs RAG-Token).

**Caveats (be honest about these)**
- 23 answerable questions means one question is ~4 points; differences of a few points are noise. The 100-vs-500 gap is large enough to trust, the 200-vs-100 gap less so.
- Hit is page-level, not chunk-level, so it measures "found the right page", not "the chunk contains the answer text".
- Chunk sizes stop at 500 because bge-small truncates input at 512 tokens; 800-token chunks would be silently cut off when embedded.
- Smaller chunks give the LLM less surrounding context per passage, so end-to-end answer quality may favour a larger chunk or higher top-k. The default is top-5 of 100-token chunks. `python eval/run_eval.py --answers` scores the full LLM path (answer keywords, correct page cited, refusal on unanswerable questions) but **needs an `LLM_API_KEY` and has not been run yet**.

## Design decisions

- **Chunks never cross page boundaries**, so every chunk has exactly one page to cite.
- **Header/footer cleaning** drops lines (digits normalised) that repeat on >50% of pages, plus bare page numbers, and re-joins hyphenated line breaks. Documents under 3 pages are left alone since repeats can't be told from content.
- **Structured output:** the LLM must return `{answer, found, cited_sources}`. Cited passage numbers are mapped back to chunk metadata, so shown sources are the ones the model used.
- **Hallucination guards:** (1) prompt says answer only from context, else `found=false`; (2) a cosine floor (`MIN_SCORE=0.65`) skips the LLM entirely for off-topic questions. I calibrated it from data: answerable questions had a top-1 score ≥ 0.74, off-topic ones ~0.48–0.55. A near-domain unanswerable question (BLEU on English-to-Spanish, which the paper doesn't report) scored 0.80, so a score floor can't catch those and the prompt must.
- **Duplicate uploads:** the SHA-256 of the file is the document id; re-uploading returns the existing record without re-embedding.
- **Cost:** embeddings are local and persisted in Chroma, so each document is embedded once.

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # add a free key from https://console.groq.com/keys as LLM_API_KEY
.venv/bin/uvicorn app.main:app --reload --port 8000
```
```bash
cd frontend && npm install && npm run dev      # http://localhost:5173
```

Any OpenAI-compatible provider works by changing `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env`. The first upload downloads the embedding model (~70 MB) to `data/models`.

Tests: `.venv/bin/python -m pytest` (parsing, chunking, retrieval on a generated PDF, full API flow with a fake LLM).

## API

| | |
|---|---|
| `POST /upload` | multipart PDF → `{document, duplicate}` |
| `POST /ask` | `{question, top_k?, doc_ids?}` → `{answer, found, sources: [{file, page, snippet, score}]}` |
| `GET /documents` | list uploaded files |
| `DELETE /documents/{id}` | remove a file and its chunks |

## Limitations and next steps

- Tables and multi-column layouts can extract jumbled; scanned PDFs have no text (rejected with a clear error). Swapping in a layout-aware parser or OCR for those pages is the next step.
- No re-ranker or hybrid (BM25 + dense) retrieval yet. The RAG paper's own ablation found BM25 won on entity-heavy FEVER, so hybrid is the first experiment I'd add to the eval.
- Single-user, no auth; uploads and the index live in `./data`.
- Rebuild one stage with LlamaIndex and compare against this eval.
