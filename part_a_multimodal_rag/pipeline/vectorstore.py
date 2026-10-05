"""Vector store — Bagian A tahap 4.

Embedding : fastembed `paraphrase-multilingual-MiniLM-L12-v2` (lokal,
            multilingual — kuat utk bahasa Indonesia, 0,22 GB, tanpa biaya API).
Storage   : ChromaDB (persistent ./chroma_db) dengan metadata {page, type}
            → mendukung penyaringan & sitasi halaman.
"""
from __future__ import annotations

from pathlib import Path

from chromadb import PersistentClient
from fastembed import TextEmbedding

_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_USE_PREFIX = "e5" in _MODEL  # model keluarga E5 wajib prefix query:/passage:
_COLLECTION = "mandiri_report"


class VectorStore:
    def __init__(self, persist_dir: str | Path = "chroma_db") -> None:
        self.persist_dir = str(persist_dir)
        self.client = PersistentClient(path=self.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = None  # lazy-load (unduh model saat pertama kali)

    # ---------- embedding ----------
    @property
    def embedder(self) -> TextEmbedding:
        if self._embedder is None:
            self._embedder = TextEmbedding(model_name=_MODEL)
        return self._embedder

    def _embed_passages(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"passage: {t}" for t in texts] if _USE_PREFIX else texts
        return [list(v) for v in self.embedder.embed(prefixed)]

    def _embed_query(self, text: str) -> list[float]:
        q = f"query: {text}" if _USE_PREFIX else text
        return list(next(iter(self.embedder.embed([q]))))

    # ---------- operasi ----------
    def count(self) -> int:
        return self.collection.count()

    def build(self, chunks: list[dict]) -> dict:
        """(Re)bangun index dari daftar chunk {id, page, type, content}."""
        self.client.delete_collection(_COLLECTION)
        self.collection = self.client.get_or_create_collection(
            name=_COLLECTION, metadata={"hnsw:space": "cosine"},
        )
        if chunks:
            self.collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["content"] for c in chunks],
                metadatas=[{"page": c["page"], "type": c["type"]} for c in chunks],
                embeddings=self._embed_passages([c["content"] for c in chunks]),
            )
        return {"indexed": self.count()}

    def all_chunks(self) -> list[dict]:
        """Seluruh chunk beserta metadata — untuk Knowledge Base Explorer."""
        if self.count() == 0:
            return []
        res = self.collection.get(include=["documents", "metadatas"])
        items = []
        for id_, doc, meta in zip(res["ids"], res["documents"], res["metadatas"]):
            items.append({"id": id_, "page": meta.get("page"),
                          "type": meta.get("type"), "content": doc})
        items.sort(key=lambda x: (x["page"], x["id"]))
        return items

    def search(
        self,
        question: str,
        k: int = 6,
        *,
        where: dict | None = None,
    ) -> list[dict]:
        """Retrieval top-k → [{content, page, type, distance}] (urut skor)."""
        if self.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[self._embed_query(question)],
            n_results=min(k, self.count()),
            where=where,
        )
        out = []
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]
        for doc, meta, dist in zip(docs, metas, dists):
            out.append({
                "content": doc,
                "page": meta.get("page"),
                "type": meta.get("type"),
                "distance": round(dist, 4),
            })
        return out
