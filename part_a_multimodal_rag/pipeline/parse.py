"""
Multimodal PDF Parsing — Bagian A (Multimodal RAG)

Mengubah PDF mentah menjadi section terstruktur per halaman:
  - type "text"  : blok teks naratif (layout-aware, bbox dipertahankan)
  - type "table" : tabel dengan struktur baris×kolom dipertahankan (Markdown)
  - type "figure": region visual/chart (diproses VLM di pipeline/vision.py)

Setiap Section WAJIB membawa metadata halaman — dipakai untuk sitasi sumber
pada endpoint query (syarat technical test).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pymupdf  # PyMuPDF

# Ambang ukuran font untuk mendeteksi judul section (berdasarkan recon dokumen)
HEADING_FONT_SIZE = 13.0
_WORD = re.compile(r"[\w(),/%.&-]")


@dataclass
class Section:
    """Satu unit konten hasil parsing, siap di-chunk & di-embed."""
    page: int                    # 1-based (nomor halaman fisik PDF)
    type: str                    # "text" | "table" | "figure"
    content: str                 # isi bersih (Markdown utk tabel)
    bbox: tuple | None = None    # (x0, y0, x1, y1) — untuk debugging layout
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _bbox_inside(inner: tuple, outer: tuple, tol: float = 6.0) -> bool:
    """Apakah bbox inner berada (sebagian besar) di dalam bbox outer."""
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = max(ix0, ox0), max(iy0, oy0), min(ix1, ox1), min(iy1, oy1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
    return inter / area > 0.55


def _clean_cell(cell: str | None) -> str:
    if cell is None:
        return ""
    return re.sub(r"\s+", " ", cell.replace("\n", " ")).strip()


def _split_merged_number(cell: str) -> str:
    """Pisahkan sel gabungan hasil merge, mis. '18.787.107 9,47' → '18.787.107 | 9,47'.
    Terjadi pada kolom Pertumbuhan (Nominal + % menyatu di satu sel)."""
    m = re.match(r"^(\(?[\d.]+\)?)\s+(-?[\d.,]+%)$", cell)
    if m:
        return f"{m.group(1)} | {m.group(2)}"
    return cell


def rows_to_markdown(rows: list[list[str | None]]) -> str:
    """Render baris tabel menjadi Markdown, dengan pembersihan sel merge."""
    cleaned = [
        [_split_merged_number(_clean_cell(c)) for c in row]
        for row in rows
    ]
    if not cleaned:
        return ""
    ncol = max(len(r) for r in cleaned)
    for r in cleaned:
        r.extend([""] * (ncol - len(r)))
    # baris kedua = header separator (baris pertama = header)
    md = ["| " + " | ".join(cleaned[0]) + " |",
          "| " + " | ".join(["---"] * ncol) + " |"]
    md += ["| " + " | ".join(r) + " |" for r in cleaned[1:]]
    return "\n".join(md)


def parse_pdf(path: str | Path) -> list[Section]:
    """Parsing utama: teks + tabel per halaman, dengan metadata halaman.

    Strategi (hasil recon):
      - Tabel   : PyMuPDF `find_tables` (terbukti paling akurat utk dokumen ini;
                  pdfplumber kehilangan kolom nama sektor).
      - Teks    : blok `get_text("dict")` DI LUAR bbox tabel (hindari duplikasi
                  konten tabel sebagai teks).
      - Judul   : deteksi via ukuran font maksimum blok.
    """
    doc = pymupdf.open(str(path))
    sections: list[Section] = []

    for pno, page in enumerate(doc, start=1):
        # ---- 1) TABEL (struktur dipertahankan) ----
        table_bboxes: list[tuple] = []
        try:
            for t in page.find_tables().tables:
                md = rows_to_markdown(t.extract())
                if md and t.row_count >= 2:
                    sections.append(Section(
                        page=pno, type="table", content=md, bbox=tuple(t.bbox),
                        meta={"rows": t.row_count, "cols": t.col_count},
                    ))
                    table_bboxes.append(tuple(t.bbox))
        except Exception:  # pragma: no cover - parser defensif
            pass

        # ---- 2) TEKS (layout-aware, di luar tabel) ----
        try:
            data = page.get_text("dict")
        except Exception:  # pragma: no cover
            continue
        for block in data.get("blocks", []):
            if block.get("type") != 0:
                continue
            bbox = tuple(block["bbox"])
            if any(_bbox_inside(bbox, tb) for tb in table_bboxes):
                continue  # konten tabel sudah ditangani sebagai Section table
            spans = [s for line in block.get("lines", []) for s in line.get("spans", [])]
            text = " ".join(s["text"] for s in spans).strip()
            if not text or not _WORD.search(text):
                continue
            max_size = max((s["size"] for s in spans), default=0)
            is_heading = max_size >= HEADING_FONT_SIZE and len(text) < 120
            if is_heading:
                sections.append(Section(
                    page=pno, type="text", content=f"## {text}", bbox=bbox,
                    meta={"heading": True, "font_size": round(max_size, 1)},
                ))
            else:
                sections.append(Section(
                    page=pno, type="text", content=text, bbox=bbox,
                    meta={"heading": False},
                ))

    doc.close()
    return sections


def parse_summary(sections: list[Section]) -> dict:
    """Ringkasan statistik hasil parsing (dipakai endpoint ingest & eval)."""
    pages = {s.page for s in sections}
    by_type: dict[str, int] = {}
    for s in sections:
        by_type[s.type] = by_type.get(s.type, 0) + 1
    return {
        "pages": len(pages),
        "sections": len(sections),
        "by_type": by_type,
    }


if __name__ == "__main__":  # pragma: no cover - utilitas CLI cepat
    import sys
    import json
    secs = parse_pdf(sys.argv[1])
    print(json.dumps(parse_summary(secs), indent=2))
    for s in secs:
        head = s.content.replace("\n", " ")[:88]
        print(f"p{s.page:02d} [{s.type:5s}] {head}")
