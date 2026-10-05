"""Chunking layout-aware — Bagian A tahap 3.

Prinsip (syarat soal: "pemotongan cerdas agar konteks tidak terputus"):
  - Tabel       → SATU chunk utuh (tidak boleh terpotong baris).
  - Caption VLM → SATU chunk utuh (asosiasi chart sudah terjaga di caption).
  - Teks        → heading MENEMPEL pada paragraf sesudahnya; blok berurutan
                  digabung; jika terlalu panjang dipotong di batas kalimat
                  dengan overlap.
Setiap chunk membawa metadata {page, type} — dipakai untuk sitasi sumber.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from pipeline.parse import Section

MAX_CHARS = 1200          # batas lembut per chunk teks
OVERLAP_CHARS = 180       # overlap antar potongan kalimat
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    id: str
    page: int
    type: str            # text | table | figure
    content: str
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Potong teks panjang di batas kalimat dengan overlap."""
    if len(text) <= max_chars:
        return [text]
    sents = _SENT_SPLIT.split(text)
    parts: list[str] = []
    buf = ""
    for s in sents:
        if buf and len(buf) + len(s) + 1 > max_chars:
            parts.append(buf.strip())
            # mulai potongan baru dengan kalimat terakhir sebagai overlap
            tail = buf[-OVERLAP_CHARS:].strip()
            buf = (tail + " " + s) if tail else s
        else:
            buf = (buf + " " + s).strip() if buf else s
    if buf.strip():
        parts.append(buf.strip())
    return parts


def build_chunks(
    sections: list[Section],
    captions: dict[int, str] | None = None,   # {page: caption}
) -> list[Chunk]:
    """Gabungkan section hasil parsing + caption visual menjadi chunk final."""
    captions = captions or {}
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}

    _code = {"table": "tbl", "text": "txt", "figure": "fig"}

    def _add(page: int, ctype: str, content: str, **meta) -> None:
        counters[ctype] = counters.get(ctype, 0) + 1
        chunks.append(Chunk(
            id=f"p{page:02d}-{_code.get(ctype, ctype[:3])}{counters[ctype]:03d}",
            page=page, type=ctype, content=content, meta=meta,
        ))

    # ---- 1) Tabel & caption: chunk utuh ----
    for s in sections:
        if s.type == "table":
            _add(s.page, "table", s.content, rows=s.meta.get("rows"))
    for page in sorted(captions):
        cap = captions[page].strip()
        if cap and "TIDAK ADA ELEMEN VISUAL" not in cap.upper():
            _add(page, "figure", cap, source="vlm-caption")

    # ---- 2) Teks: heading menempel + merge blok berurutan per halaman ----
    by_page: dict[int, list[Section]] = {}
    for s in sections:
        if s.type == "text":
            by_page.setdefault(s.page, []).append(s)

    for page in sorted(by_page):
        blocks = by_page[page]
        merged: list[str] = []
        pending_heading = ""
        for s in blocks:
            if s.meta.get("heading"):
                # heading menempel ke konten sesudahnya (konteks tidak terputus)
                if pending_heading and merged:
                    merged[-1] = pending_heading + " — " + merged[-1]
                pending_heading = s.content.removeprefix("## ").strip()
                continue
            text = s.content.strip()
            if not text:
                continue
            if pending_heading:
                text = f"[{pending_heading}] {text}"
                pending_heading = ""
            if merged and len(merged[-1]) + len(text) < MAX_CHARS:
                merged[-1] = merged[-1] + " " + text
            else:
                merged.append(text)
        if pending_heading:  # heading tanpa isi setelahnya
            merged.append(pending_heading)
        for m in merged:
            for part in _split_long(m):
                _add(page, "text", part)

    return chunks


def chunk_summary(chunks: list[Chunk]) -> dict:
    by_type: dict[str, int] = {}
    for c in chunks:
        by_type[c.type] = by_type.get(c.type, 0) + 1
    return {"chunks": len(chunks), "by_type": by_type}
