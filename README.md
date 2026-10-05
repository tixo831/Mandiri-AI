# 🤖 AI Engineer Intern — Technical Test

> 🌐 **Demo interaktif 24/7:** https://tiko76-mandiri-ai.static.hf.space/index.html — knowledge base + retrieval + jawaban (statis, gratis)
> 🌐 **Halaman showcase:** https://tixo831.github.io/Mandiri-AI/ — hasil evaluasi, arsitektur & demo interaktif Bagian B

Solusi dua bagian:

| Bagian | Deliverable | Status |
|---|---|---|
| **A. Multimodal RAG** | REST API (FastAPI + LangGraph + ChromaDB) atas *Laporan Keuangan Bank Mandiri 2025* — endpoint **Ingestion** (parsing teks/tabel + caption visual VLM → chunking → embedding → vector DB) & **Query** (retrieval → grade → synthesis LLM + **sitasi halaman**) | ✅ **selesai — evaluasi resmi 6/6: halaman sumber & substansi jawaban lengkap** |
| **B. Layout-Aware Text Extraction** | Pipeline Python: slide → OCR (bbox) + style (warna, ukuran, bold) → **inpainting** anti-duplikasi → **HTML overlay** + Mode Edit | ✅ 5/5 slide dikonversi — QA: deviasi posisi rata2 **0,125%**, mode edit aktif |

## Bagian A — Cara Menjalankan

```bash
cd part_a_multimodal_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env          # isi GEMINI_API_KEY ( gratis: aistudio.google.com/apikey )
uvicorn app.main:app --reload --port 8000
# Swagger UI → http://localhost:8000/docs
```

```bash
# Ingestion
curl -X POST http://localhost:8000/api/ingest \
  -F "file=@Laporan Keuangan Bank Mandiri 2025.pdf"

# Query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Apa saja peran Unit Pelindungan Nasabah menurut POJK No. 22 Tahun 2023?"}'
# → { "answer": "...", "sources": [{"page": 7, "type": "text", "preview": "..."}], ... }
```

## Arsitektur Bagian A

```
INGESTION  :  PDF ─▶ parse teks (PyMuPDF, layout-aware)
                    ├─▶ tabel  (PyMuPDF find_tables → Markdown, struktur utuh)
                    ├─▶ visual (render halaman → VLM caption)          [v1]
                    ─▶ chunking layout-aware + metadata halaman        [v1]
                    ─▶ embedding (multilingual-e5-small) → ChromaDB   [v1]
QUERY      :  pertanyaan ─▶ embedding ─▶ retrieval ─▶ LangGraph
                    (grade → synthesize) ─▶ jawaban + sources[{page}]  [v1]
```

### Validasi parsing (Hari 1) — 6 soal evaluasi

Skrip `scripts/inspect_parse.py` memverifikasi konten sumber keenam soal uji
sudah tertangkap parser **dengan metadata halaman yang benar**:

```
[OK] hal 7 — soal #1 peran Unit Pelindungan Nasabah (POJK 22/2023)
[OK] hal 7 — soal #2 jam penagihan (08.00–20.00 → 21.00 tidak boleh)
[OK] hal 4 — soal #3 kredit sektor tambang   (157.186.029 · +7,98%)
[OK] hal 4 — soal #3 kredit sektor konstruksi (108.148.636 · +8,27%)
[OK] hal 6 — soal #4 komposisi DPK 2025 (40,12 / 39,31 / 20,57)
[OK] hal 8 — soal #5 alur penanganan pengaduan
[OK] hal 9 — soal #6 saluran pengaduan
```

> 168 section (5 tabel + 163 blok teks) · ingestion ±1 detik.

## Struktur

```
part_a_multimodal_rag/
├─ app/main.py            # FastAPI: /api/health · /api/ingest · /api/query (v1)
├─ pipeline/parse.py      # parsing multimodal (teks layout-aware + tabel Markdown)
├─ scripts/inspect_parse.py
└─ eval/                  # evaluasi otomatis 6 soal (v1)
```



## Hasil Evaluasi Bagian A — 6 Pertanyaan Resmi

# Hasil Evaluasi — 6 Pertanyaan Technical Test

| # | Pertanyaan (ringkas) | Halaman sumber | Ekspektasi | Halaman ✓ | Substansi ✓ |
|---|---|---|---|---|---|
| 1 | Apa saja peran Unit Pelindungan Nasabah menurut peratur… | [7] | 7 | ✅ | ✅ |
| 2 | Jika Bank Mandiri menggunakan jasa Perusahaan Jasa Pena… | [7] | 7 | ✅ | ✅ |
| 3 | Berapa nominal dan persentase pertumbuhan kredit yang d… | [4] | 4 | ✅ | ✅ |
| 4 | Sebutkan presentase komposisi dana pihak ketiga (DPK) d… | [6, 7, 5, 4, 8] | 6 | ✅ | ✅ |
| 5 | Bagaimana alur penganganan nasabah di bank mandiri jika… | [9, 8] | 8 | ✅ | ✅ |
| 6 | Apa saja saluran pengaduan yang disediakan oleh Bank Ma… | [9, 8] | 9 | ✅ | ✅ |

**Ringkasan:** halaman sumber benar 6/6 · substansi jawaban lengkap 6/6

## Bagian B — Layout-Aware Text Extraction

```bash
cd part_b_layout_extraction
pip install rapidocr-onnxruntime opencv-python-headless numpy

# konversi satu slide
python -m pipeline.extract "slide.jpg" -o out.html

# tanpa inpainting (background asli dipakai apa adanya)
python -m pipeline.extract "slide.jpg" --no-inpaint
```

### Pipeline

```
slide 5734px
 ├─ 1. DOWNSCALE 2000px          (stabil untuk OCR; posisi tetap akurat —
 │                                 koordinat HTML memakai persen & cqw)
 ├─ 2. OCR RapidOCR              → teks + bounding box + confidence
 ├─ 3. STYLE DETECTION           → warna (median piksel "tinta inti"),
 │                                 font-size (0,78×tinggi box), bold (densitas)
 ├─ 4. INPAINTING (anti-duplikasi): mask semua box → cv2.inpaint TELEA
 │                                 → background bersih dari teks asli
 └─ 5. HTML GENERATION           → background base64 + <span> absolut
                                   (left/top/width %, font-size cqw) + toolbar
```

### Hasil & QA (headless Chrome)

- `outputs/slide-01.html` … `slide-05.html` — hasil konversi 5 slide (interaktif: **✏️ Mode Edit** → teks bisa diklik & diedit, ↺ Reset)
- `outputs/compare.html` — galeri perbandingan **asli vs hasil**
- QA otomatis (`outputs/qa/`): deviasi posisi style vs render **rata2 0,125% · maks 0,957%**; inpaint terverifikasi mengubah 7,3% area (region teks) sehingga tidak ada efek teks ganda
- Skema `cqw` + persen → hasil tetap proporsional di layar/zoom berapa pun


## Aset

Dokumen & slide resmi diunduh dari Google Drive sesuai soal (lihat `assets/README.md`).
