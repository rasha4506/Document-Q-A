"""Retrieval evaluation: for each (chunk_size, overlap) config, how often does the correct page
appear in the top-k retrieved chunks?

    python eval/run_eval.py                  # retrieval grid -> eval/results.md + results.json
    python eval/run_eval.py --answers        # also run the LLM end-to-end on the best config (needs LLM_API_KEY)
    python eval/run_eval.py --misses         # print the questions each config missed
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_pages  # noqa: E402
from app.embeddings import get_embedder  # noqa: E402
from app.parsing import parse_pdf  # noqa: E402
from app.store import VectorStore  # noqa: E402

DOCS = ROOT / "eval" / "docs"
CACHE = ROOT / "eval" / ".cache"
KS = (1, 3, 5)
# bge-small truncates input at 512 tokens, so 500 is the largest size that is embedded in full.
CHUNK_SIZES = (100, 200, 300, 500)
OVERLAP_FRACS = (0.0, 0.2)


def load_questions() -> list[dict]:
    return json.loads((ROOT / "eval" / "questions.json").read_text())


def build_store(pages_by_file: dict, size: int, overlap: int, embedder) -> tuple[VectorStore, int]:
    CACHE.mkdir(exist_ok=True)
    store = VectorStore(str(CACHE / f"chroma_{size}_{overlap}"), f"eval_cs{size}_ov{overlap}")
    chunks = []
    for fname, pages in pages_by_file.items():
        chunks += chunk_pages(pages, fname, fname, size, overlap)
    store.add(chunks, embedder.embed_documents([c.text for c in chunks]))
    return store, len(chunks)


def evaluate(store: VectorStore, questions: list[dict], embedder, max_k: int) -> dict:
    answerable = [q for q in questions if q["page"] is not None]
    hits_at = {k: 0 for k in KS}
    rr_sum, misses = 0.0, []
    for q in answerable:
        hits = store.query(embedder.embed_query(q["question"]), max_k)
        rank = next((i for i, h in enumerate(hits, 1) if h.file == q["file"] and h.page == q["page"]), None)
        for k in KS:
            hits_at[k] += rank is not None and rank <= k
        rr_sum += 1 / rank if rank else 0.0
        if rank is None or rank > 3:
            misses.append({"id": q["id"], "question": q["question"], "expected": f'{q["file"]} p.{q["page"]}',
                           "got": [f"{h.file} p.{h.page}" for h in hits[:3]]})
    n = len(answerable)
    return {**{f"hit@{k}": hits_at[k] / n for k in KS}, "mrr": rr_sum / n, "n": n, "misses": misses}


def run_answers(size: int, overlap: int, questions: list[dict], pages_by_file: dict, embedder) -> dict:
    from app.config import get_settings
    from app.llm import OpenAICompatClient
    from app.qa import answer_question

    s = get_settings()
    store, _ = build_store(pages_by_file, size, overlap, embedder)
    llm = OpenAICompatClient()
    correct = page_ok = refused_ok = n_ans = n_unans = 0
    for q in questions:
        time.sleep(2)  # stay inside free-tier rate limits
        r = answer_question(q["question"], store, embedder, llm, s.top_k, s.min_score)
        if q["page"] is None:
            n_unans += 1
            refused_ok += not r.found
        else:
            n_ans += 1
            text = r.answer.lower()
            correct += r.found and any(k in text for k in q["keywords"])
            page_ok += any(src.file == q["file"] and src.page == q["page"] for src in r.sources)
    return {"answer_keyword_match": correct / n_ans, "cited_correct_page": page_ok / n_ans,
            "refused_unanswerable": refused_ok / n_unans, "n_answerable": n_ans, "n_unanswerable": n_unans}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="store_true", help="also evaluate end-to-end answers with the LLM")
    ap.add_argument("--misses", action="store_true", help="print questions missed in the top 3 for each config")
    args = ap.parse_args()

    if CACHE.exists():
        shutil.rmtree(CACHE)
    questions = load_questions()
    embedder = get_embedder()
    pages_by_file = {p.name: parse_pdf(p) for p in sorted(DOCS.glob("*.pdf"))}
    print(f"{len(questions)} questions ({sum(q['page'] is not None for q in questions)} answerable) "
          f"over {len(pages_by_file)} docs, {sum(len(v) for v in pages_by_file.values())} pages\n")

    rows = []
    for size in CHUNK_SIZES:
        for frac in OVERLAP_FRACS:
            overlap = int(size * frac)
            store, n_chunks = build_store(pages_by_file, size, overlap, embedder)
            res = evaluate(store, questions, embedder, max(KS))
            rows.append({"chunk_size": size, "overlap": overlap, "chunks": n_chunks, **res})
            print(f"size={size:<4} overlap={overlap:<3} chunks={n_chunks:<4} "
                  + " ".join(f"hit@{k}={res[f'hit@{k}']:.0%}" for k in KS) + f" mrr={res['mrr']:.2f}")
            if args.misses:
                for m in res["misses"]:
                    print(f"    miss #{m['id']}: {m['question']}  expected {m['expected']}, got {m['got']}")

    lines = ["| chunk size (tokens) | overlap | chunks | hit@1 | hit@3 | hit@5 | MRR |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['chunk_size']} | {r['overlap']} | {r['chunks']} | " +
                     " | ".join(f"{r[f'hit@{k}']:.0%}" for k in KS) + f" | {r['mrr']:.2f} |")
    table = "\n".join(lines)

    best = max(rows, key=lambda r: (r["hit@3"], r["mrr"]))
    out = {"retrieval": rows, "best": {"chunk_size": best["chunk_size"], "overlap": best["overlap"]}}
    md = f"# Retrieval evaluation\n\nn = {rows[0]['n']} answerable questions. Hit = correct file+page in top-k chunks.\n\n{table}\n"
    if args.answers:
        ans = run_answers(best["chunk_size"], best["overlap"], questions, pages_by_file, embedder)
        out["answers"] = ans
        md += (f"\n## End-to-end answers (chunk size {best['chunk_size']}, overlap {best['overlap']})\n\n"
               f"- keyword match on answerable: {ans['answer_keyword_match']:.0%}\n"
               f"- cited the correct page: {ans['cited_correct_page']:.0%}\n"
               f"- correctly refused unanswerable: {ans['refused_unanswerable']:.0%} ({ans['n_unanswerable']} questions)\n")
        print("\n", ans)

    (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=2))
    (ROOT / "eval" / "results.md").write_text(md)
    print("\n" + table + f"\n\nBest by hit@3 then MRR: size={best['chunk_size']} overlap={best['overlap']}")


if __name__ == "__main__":
    main()
