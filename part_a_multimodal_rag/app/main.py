"""FastAPI — Multimodal RAG over Laporan Bank Mandiri 2025.

Endpoint (sesuai technical test):
  POST /api/ingest — upload & proses PDF → parsing multimodal (teks/tabel)
                     + caption visual (VLM) → chunking layout-aware
                     → embedding → ChromaDB
  POST /api/query  — jawab pertanyaan via retrieval + LLM synthesis
                     + metadata halaman sumber      (rilis v2)

Jalankan:
  PYTHONPATH=. uvicorn app.main:app --reload --port 8000
  → Swagger UI: http://localhost:8000/docs
"""
from __future__ import annotations

import shutil
from pathlib import Path
import tempfile
import time

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.chunk import build_chunks, chunk_summary
from pipeline.graph import run_query
from pipeline.parse import parse_pdf, parse_summary
from pipeline.vectorstore import VectorStore
from pipeline.vision import caption_page

app = FastAPI(
    title="Multimodal RAG — Bank Mandiri Report QnA",
    description="Technical Test AI Engineer — Bagian A. "
                "Ingestion multimodal (teks/tabel/visual) + Query dengan sitasi halaman.",
    version="0.2.0",
)

_STATE: dict = {"ingested": False, "filename": None, "stats": {}}
_store = VectorStore()

_BASE = Path(__file__).resolve().parents[1]          # part_a_multimodal_rag/
_PART_B = _BASE.parent / "part_b_layout_extraction" / "outputs"

@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Antarmuka web demo sederhana."""
    return FileResponse(_BASE / "app" / "static" / "index.html")

app.mount("/static", StaticFiles(directory=_BASE / "app" / "static"), name="static")
if _PART_B.exists():
    app.mount("/part-b", StaticFiles(directory=_PART_B), name="partb")


@app.get("/api/health")
def health() -> dict:
    """Status API + statistik knowledge base."""
    return {
        "status": "ok",
        "ingested": _STATE["ingested"],
        "indexed_chunks": _store.count(),
        "stats": _STATE["stats"],
    }


@app.post("/api/ingest")
def ingest(file: UploadFile = File(...)) -> dict:
    """Upload PDF → pipeline ingestion lengkap.

    Tahap: parsing teks+tabel (PyMuPDF) → caption visual per halaman
    (Gemini VLM, di-cache) → chunking layout-aware → embedding
    (multilingual-e5-small) → ChromaDB.
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "File harus PDF")
    t0 = time.perf_counter()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    timing: dict[str, float] = {}
    try:
        # 1) parsing teks & tabel (struktur dipertahankan)
        t = time.perf_counter()
        sections = parse_pdf(tmp_path)
        n_pages = len({s.page for s in sections})
        timing["parse_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 2) caption visual per halaman (cache → hemat kuota; skip jika
        #    halaman tanpa elemen visual)
        captions: dict[int, str] = {}
        caption_errors: list[str] = []
        t = time.perf_counter()
        for page in range(1, n_pages + 1):
            try:
                captions[page] = caption_page(tmp_path, page)
            except Exception as exc:  # VLM gagal → lanjut tanpa caption
                caption_errors.append(f"hal {page}: {exc}")
        timing["caption_ms"] = round((time.perf_counter() - t) * 1000, 1)

        # 3) chunking layout-aware
        t = time.perf_counter()
        chunks = build_chunks(sections, captions)
        timing["chunk_ms"] = round((time.perf_counter() - t) * 1000, 1)
        # 4+5) embedding + simpan ke vector DB
        t = time.perf_counter()
        indexed = _store.build([c.to_dict() for c in chunks])
        timing["embed_store_ms"] = round((time.perf_counter() - t) * 1000, 1)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Gagal memproses PDF: {exc}") from exc
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)

    elapsed = round((time.perf_counter() - t0) * 1000, 1)
    _STATE.update({
        "ingested": True,
        "filename": file.filename,
        "stats": {
            "pages": n_pages,
            "sections": parse_summary(sections),
            "captions": sum(
                1 for c in captions.values()
                if c and "TIDAK ADA ELEMEN VISUAL" not in c.upper()
            ),
            **chunk_summary(chunks),
            "indexed": indexed["indexed"],
        },
    })
    return {
        "status": "ok",
        "filename": file.filename,
        "elapsed_ms": elapsed,
        "timing": timing,
        **_STATE["stats"],
        **({"caption_errors": caption_errors} if caption_errors else {}),
    }


class QueryIn(BaseModel):
    question: str
    top_k: int = 6


@app.get("/api/kb")
def kb() -> dict:
    """Knowledge Base Explorer — seluruh chunk hasil parsing multimodal
    (teks, tabel terstruktur, caption visual) beserta metadata halamannya."""
    chunks = _store.all_chunks()
    pages = sorted({c["page"] for c in chunks})
    by_page = {pg: [c for c in chunks if c["page"] == pg] for pg in pages}
    return {"count": len(chunks), "pages": pages, "by_page": by_page}


@app.post("/api/query")
def query(body: QueryIn) -> dict:
    """Jawab pertanyaan: retrieval → grade (LangGraph) → synthesis LLM
    + sitasi halaman. Fallback ke retrieval murni bila LLM tidak tersedia."""
    if _store.count() == 0:
        raise HTTPException(400, "Knowledge base kosong — jalankan /api/ingest dulu")
    t0 = time.perf_counter()
    try:
        result = run_query(body.question, _store)
        ctx = (result.get("relevant_hits") or result.get("hits") or [])[:8]
        return {
            "question": body.question,
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "contexts": [{"page": h["page"], "type": h["type"],
                          "content": h["content"]} for h in ctx],
            "engine": "langgraph+gemini",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
    except Exception as exc:
        hits = _store.search(body.question, k=body.top_k)
        seen: set = set()
        sources = []
        for h in hits:
            if h["page"] not in seen:
                seen.add(h["page"])
                sources.append({"page": h["page"], "type": h["type"],
                                "preview": h["content"][:160]})
        return {
            "question": body.question,
            "answer": None,
            "sources": sources,
            "hits": hits,
            "engine": "retrieval-fallback",
            "error": str(exc),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
