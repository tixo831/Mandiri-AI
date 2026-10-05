"""Evaluasi otomatis Bagian A — 6 pertanyaan resmi technical test.

Jalankan SETELAH /api/ingest (atau skrip ini melakukan ingest sendiri):
    PYTHONPATH=. python -m eval.run_eval <path_pdf>

Output:
  - cetak jawaban + halaman sumber per soal (dengan ekspektasi halaman)
  - simpan eval/results.md (tabel untuk README) & eval/results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

QUESTIONS = [
    ("Apa saja peran Unit Pelindungan Nasabah menurut peraturan POJK No. 22 Tahun 2023?", 7),
    ("Jika Bank Mandiri menggunakan jasa Perusahaan Jasa Penagihan untuk menagih debitur, apakah penagihan boleh dilakukan pada jam 21.00?", 7),
    ("Berapa nominal dan persentase pertumbuhan kredit yang diberikan di sektor tambang dan konstruksi?", 4),
    ("Sebutkan presentase komposisi dana pihak ketiga (DPK) di Bank Mandiri pada tahun 2024 dan 2025?", 6),
    ("Bagaimana alur penganganan nasabah di bank mandiri jika terdapat laporan pengaduan?", 8),
    ("Apa saja saluran pengaduan yang disediakan oleh Bank Mandiri?", 9),
]

KEY_FACTS = {  # string wajib muncul di jawaban (pengecekan substansi)
    1: ["Pelindungan Nasabah"],
    2: ["08.00", "20.00"],
    3: ["157.186.029", "108.148.636", "7,98", "8,27"],
    4: ["36,66", "34,23", "29,11"],
    5: ["investigasi"],
    6: ["WhatsApp", "14000"],
}


def main(pdf_path: str) -> int:
    client = TestClient(app)

    indexed = client.get("/api/health").json().get("indexed_chunks", 0)
    if indexed == 0:
        print("Knowledge base kosong → menjalankan ingestion…")
        with open(pdf_path, "rb") as f:
            r = client.post("/api/ingest",
                            files={"file": (Path(pdf_path).name, f, "application/pdf")})
            r.raise_for_status()
            print("ingest:", r.json()["chunks"], "chunk")

    rows = []
    n_page_ok = n_fact_ok = 0
    for i, (q, expected_page) in enumerate(QUESTIONS, 1):
        # hingga 3 percobaan: LLM transient error (429 rate limit) → tunggu & ulangi
        for attempt in range(3):
            r = client.post("/api/query", json={"question": q})
            d = r.json()
            if d.get("answer") or attempt == 2:
                break
            print(f"  (soal {i}: belum terjawab — tunggu 30 dtk, coba lagi [{attempt+2}/3])")
            time.sleep(30)
        answer = d.get("answer") or ""
        pages = ([s["page"] for s in d.get("sources", [])]
                 or [h["page"] for h in d.get("hits", [])])
        page_ok = expected_page in pages
        facts = KEY_FACTS.get(i, [])
        missing = [f for f in facts if f not in answer]
        fact_ok = not missing
        n_page_ok += page_ok
        n_fact_ok += fact_ok
        print(f"\n===== SOAL {i} ({d.get('engine')}) =====")
        print(answer[:800])
        print(f"halaman sumber: {pages} | ekspektasi: {expected_page} "
              f"{'✓' if page_ok else '✗'} | fakta kunci: "
              f"{'✓ lengkap' if fact_ok else f'kurang {missing}'}")
        rows.append({
            "no": i, "question": q, "expected_page": expected_page,
            "pages": pages, "page_ok": page_ok, "fact_ok": fact_ok,
            "answer": answer, "engine": d.get("engine"),
            "error": d.get("error"),
            "elapsed_ms": d.get("elapsed_ms"),
        })
        time.sleep(8)  # pacing rate limit (free tier)

    md = ["# Hasil Evaluasi — 6 Pertanyaan Technical Test", "",
          "| # | Pertanyaan (ringkas) | Halaman sumber | Ekspektasi | Halaman ✓ | Substansi ✓ |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        short = r["question"][:55] + "…"
        md.append(f"| {r['no']} | {short} | {r['pages']} | {r['expected_page']} "
                  f"| {'✅' if r['page_ok'] else '❌'} | {'✅' if r['fact_ok'] else '❌'} |")
    md += ["", f"**Ringkasan:** halaman sumber benar {n_page_ok}/6 · "
               f"substansi jawaban lengkap {n_fact_ok}/6"]
    Path("eval").mkdir(exist_ok=True)
    Path("eval/results.md").write_text("\n".join(md), encoding="utf-8")
    Path("eval/results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n=== RINGKASAN: halaman {n_page_ok}/6 · substansi {n_fact_ok}/6 ===")
    print("tersimpan: eval/results.md, eval/results.json")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("pakai: PYTHONPATH=. python -m eval.run_eval <path_pdf>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
