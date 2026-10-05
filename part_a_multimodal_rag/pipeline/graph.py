"""LangGraph orchestration — Bagian A tahap 5 (query v2).

Graph sederhana namun terstruktur (syarat tech stack: LangChain/LangGraph):

    retrieve → grade → (retrieved lemah? → rewrite query → retrieve ulang)
            → synthesize → jawaban + sitasi halaman

Node:
  retrieve   : pencarian vector store (top-k, metadata halaman)
  grade      : LLM menilai relevansi konteks terhadap pertanyaan (binary)
  refine     : (opsional) perbaiki query bila konteks tak relevan, lalu retrieve ulang
  synthesize : LLM menyusun jawaban AKHIR — hanya dari konteks, dengan halaman sumber
"""
from __future__ import annotations

import json
import os
from typing import TypedDict

# ---- LLM via REST (Gemini) — dipakai langsung agar dependensi minim; ----
# ---- LangGraph menyusun ORKESTRASINYA.                              ----
import httpx

from pipeline.vectorstore import VectorStore
from pipeline.vision import get_api_key

_GRADE_PROMPT = """Anda penilai relevansi dokumen. Pertanyaan pengguna dan beberapa
potongan dokumen diberikan. Pilih indeks potongan yang RELEVAN atau BERPOTENSI
memuat jawaban (bila ragu, TETAP sertakan — jangan buang potongan yang mungkin
memuat jawabannya).

Jawab HANYA dengan JSON: {{"relevant": [indeks, ...]}}

Pertanyaan: {question}

Potongan dokumen:
{snippets}"""

_ANSWER_PROMPT = """Anda asisten analisis laporan Bank Mandiri. Jawab pertanyaan pengguna
SECARA LANGSUNG, HANYA berdasarkan potongan dokumen yang diberikan.

Aturan:
1. Sebutkan angka/fakta persis seperti pada dokumen (jangan mengarang).
2. Jika informasi tersebar di beberapa potongan, gabungkan.
3. Jika dokumen tidak memuat jawabannya, katakan apa yang paling relevan dan
   nyatakan keterbatasannya — JANGAN mengarang.
4. Tidak perlu menyebut "potongan dokumen" di jawaban; jawab natural.

Pertanyaan: {question}

Potongan dokumen:
{context}"""


class GraphState(TypedDict, total=False):
    question: str
    hits: list[dict]
    relevant_hits: list[dict]
    needs_retry: bool
    answer: str
    sources: list[dict]


def _gemini_call(prompt: str) -> str:
    key = get_api_key()
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    models = [model] + [m for m in os.environ.get(
        "GEMINI_FALLBACKS", "gemini-flash-latest,gemini-3.8-flash").split(",") if m]
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    import time as _time
    last: Exception | None = None
    for attempt, mdl in enumerate(dict.fromkeys(m.strip() for m in models)):
        try:
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{mdl}:generateContent",
                json=payload, headers={"x-goog-api-key": key}, timeout=60.0,
            )
            resp.raise_for_status()
            out = "".join(p["text"] for p in
                          resp.json()["candidates"][0]["content"]["parts"]
                          if "text" in p).strip()
            _time.sleep(1.2)  # pacing — jaga rate limit free tier
            return out
        except httpx.HTTPStatusError as exc:
            last = exc
            if exc.response.status_code == 429 and attempt == 0:
                _time.sleep(25)  # rate limit: tunggu sebelum fallback model
            elif exc.response.status_code not in (404, 429, 500, 503):
                raise
        except Exception as exc:  # noqa: BLE001 — coba model berikutnya
            last = exc
    raise RuntimeError(f"Gemini gagal semua model: {last}")


# ---------------- nodes ----------------

def retrieve_node(state: GraphState, store: VectorStore) -> GraphState:
    hits = store.search(state["question"], k=8)
    return {**state, "hits": hits}


def grade_node(state: GraphState) -> GraphState:
    """Nilai relevansi SEMUA hit dalam SATU panggilan LLM (hemat kuota).
    Potongan diberikan utuh (tanpa dipotong) agar informasi di ujung chunk
    tidak terlewat. Gagal menilai → semua dianggap relevan (recall utama)."""
    q = state["question"]
    hits = state["hits"]
    snippets = "\n".join(
        f"[{i}] (hal {h['page']}) {h['content']}"
        for i, h in enumerate(hits))
    try:
        raw = _gemini_call(_GRADE_PROMPT.format(
            question=q, snippets=snippets))
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        idx = [int(i) for i in data.get("relevant", []) if 0 <= int(i) < len(hits)]
        relevant = [hits[i] for i in idx] or hits  # jangan kosongkan konteks
    except Exception:  # gagal menilai → pakai semua (jangan buang konten)
        relevant = hits
    return {**state, "relevant_hits": relevant,
            "needs_retry": not relevant}


def refine_node(state: GraphState, store: VectorStore) -> GraphState:
    """Konteks tak relevan → perbaiki query (LLM) lalu retrieve ulang sekali."""
    rewrite = _gemini_call(
        "Tulis ulang pertanyaan berikut menjadi kueri pencarian dokumen laporan "
        "bank yang efektif dalam bahasa Indonesia, tanpa penjelasan tambahan.\n"
        f"Pertanyaan: {state['question']}")
    hits = store.search(rewrite or state["question"], k=8)
    return {**state, "hits": hits, "needs_retry": False,
            "refined_query": rewrite}


def synthesize_node(state: GraphState) -> GraphState:
    hits = state.get("relevant_hits") or state["hits"]
    # konten terstruktur (chart/infografis/tabel) didahulukan agar
    # asosiasi label↔nilai tidak terbuang saat konteks padat
    hits = sorted(hits, key=lambda h: 0 if h["type"] in ("figure", "table") else 1)
    ctx = "\n\n".join(
        f"[Halaman {h['page']} · {h['type']}]\n{h['content']}" for h in hits[:8])
    answer = _gemini_call(_ANSWER_PROMPT.format(
        question=state["question"], context=ctx))
    seen: set[int] = set()
    sources = []
    for h in hits:
        if h["page"] not in seen:
            seen.add(h["page"])
            sources.append({
                "page": h["page"], "type": h["type"],
                "preview": h["content"][:160].replace("\n", " ") + "…",
            })
    return {**state, "answer": answer, "sources": sources}


# ---------------- graph ----------------

def build_rag_graph(store: VectorStore):
    """Bangun graph dengan langgraph (StateGraph)."""
    from langgraph.graph import StateGraph, END

    g = StateGraph(GraphState)
    g.add_node("retrieve", lambda s: retrieve_node(s, store))
    g.add_node("grade", grade_node)
    g.add_node("refine", lambda s: refine_node(s, store))
    g.add_node("synthesize", synthesize_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        # refine hanya SEKALI (anti loop tak berujung)
        lambda s: "refine" if s["needs_retry"] and not s.get("refined")
                  else "synthesize",
        {"refine": "refine", "synthesize": "synthesize"},
    )
    g.add_edge("refine", "grade")   # nilai ulang hasil retrieve baru
    g.add_edge("synthesize", END)
    return g.compile()


def run_query(question: str, store: VectorStore) -> dict:
    graph = build_rag_graph(store)
    return graph.invoke({"question": question})
