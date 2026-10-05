# 📋 Rencana Teknis — Technical Test AI Engineer Intern

> Durasi resmi: **1 minggu** · Pengumpulan: **Video demo + GitHub (source & dokumentasi)**
> Target kita: **selesai H+5**, H+6 rekaman video, H+7 buffer & submit.

---

## 0. Ringkasan Tugas

| | Bagian A | Bagian B |
|---|---|---|
| **Tugas** | Pipeline **Multimodal RAG** end-to-end atas PDF Laporan Bank Mandiri 2025 | **Layout-aware text extraction**: gambar slide → HTML (teks terpisah dari background) |
| **Output** | REST API: endpoint **Ingestion** + **Query** (wajib metadata halaman) | Pipeline Python/Notebook + **file HTML** (background layer + text layer overlay) |
| **Stack wajib** | FastAPI, LangChain/LangGraph, Vector DB bebas, LLM/Embedding bebas | Python, OCR/Vision bebas |
| **Evaluasi** | 6 pertanyaan contoh (teks, tabel, chart, infografis) | Akurasi teks, posisi, style + bonus anti-duplikasi & edit teks |

---

## 1. Hasil Recon Aset (sudah diunduh & dibedah)

**PDF `Laporan Keuangan Bank Mandiri 2025.pdf` — 9 halaman, TEXT-BASED (bukan scan):**

| Hal | Konten kunci | Temuan teknis |
|---|---|---|
| 1 | Laporan posisi keuangan | tabel terdeteksi ✓ |
| 4 | Tabel "Kredit … Sektor Ekonomi" (soal #3: Tambang & Konstruksi) | baris `Tambang`, `Konstruksi` + nominal ada ✓ |
| 6 | Chart vektor "KOMPOSISI DPK" (soal #4) | **angka % ADA di text layer (20,57/39,31/40,12 & 29,11/36,66/34,23) tapi TERPISAH dari label komponennya** → butuh parsing layout-aware / VLM |
| 7 | POJK 22/2023, Unit Pelindungan Nasabah, jam penagihan (soal #1, #2) | `pukul 08.00 s.d. 20.00` ada ✓ → jawaban #2: **tidak boleh** 21.00 |
| 8 | Infografis alur penanganan pengaduan (soal #5) | teks alur ada, 86 objek vektor |
| 9 | Daftar saluran pengaduan (soal #6) | WhatsApp, Email, Kantor, dll ✓ |

**Slide (Bagian B):** 5 file JPG `Profile Image Studio…` resolusi besar **5734×3200** — bagus untuk OCR.

> 💡 Artinya: text-extraction murni sudah bisa menjawab banyak soal, tapi **asosiasi chart (label↔nilai) & struktur infografis** hanya benar lewat parsing posisi (layout-aware) + interpretasi visual (VLM). Di situlah nilai "multimodal" kita dibuktikan.

---

## 2. Bagian A — Arsitektur Multimodal RAG

### Alur End-to-End

```
┌───────────────────── INGESTION (POST /api/ingest) ─────────────────────┐
│ PDF mentah                                                            │
│  ├─ 1. PARSE TEKS      : PyMuPDF per halaman (blok + posisi)          │
│  ├─ 2. PARSE TABEL     : pdfplumber → struktur Markdown (baris utuh)  │
│  ├─ 3. PARSE VISUAL    : render halaman → PNG → VLM caption           │
│  │   (Gemini Flash) — chart/infografis/gambar dibaca & dinarasikan    │
│  ├─ 4. GABUNG          : per halaman → section {teks, tabel, caption} │
│  ├─ 5. CHUNKING        : layout-aware — tabel & chart TIDAK dipotong, │
│  │   chunk per section + overlap; setiap chunk bawa metadata halaman  │
│  ├─ 6. EMBEDDING       : multilingual-e5-small (lokal, via fastembed) │
│  └─ 7. STORE           : ChromaDB (persist) + metadata {page, type,   │
│                          section, content_preview}                    │
└───────────────────────────────────────────────────────────────────────┘

┌───────────────────── QUERY (POST /api/query) ─────────────────────────┐
│ {question}                                                            │
│  ├─ 1. Embed pertanyaan                                               │
│  ├─ 2. Retrieval top-k (ChromaDB, filter metadata bila perlu)         │
│  ├─ 3. LangGraph: grade relevansi → (retrieve ulang bila lemah)       │
│  ├─ 4. Synthesis LLM (Gemini Flash) — jawab HANYA dari konteks        │
│  └─ 5. Response: {answer, sources:[{page, type, preview}], confidence}│
└───────────────────────────────────────────────────────────────────────┘
```

### Pilihan Stack + Alasan

| Komponen | Pilihan | Alasan |
|---|---|---|
| API | **FastAPI** + Uvicorn | Wajib soal; auto Swagger `/docs` (nilai tambah demo) |
| Orchestration | **LangChain + LangGraph** | Wajib soal; graph `ingest` & `query` terlihat rapi di kode |
| Parsing | **PyMuPDF + pdfplumber** | Teks berposisi + tabel terstruktur (hal 1/4/6) |
| Visual/chart | **Gemini Flash (VLM)** — render halaman → caption | Wajib "interpretasi gambar/grafik" + memecah masalah asosiasi chart DPK |
| Embedding | **fastembed `multilingual-e5-small`** (lokal) | Bahasa Indonesia kuat, tanpa kuota API, deterministik |
| Vector DB | **ChromaDB** (persist `./chroma_db`) | Simpel, metadata filtering, mudah di-review juri |
| LLM synthesis | **Gemini Flash** (fallback: Groq Llama-3.3-70B / OpenAI) | Gratis & cepat; jawaban + sitasi halaman |

### Kontrak Endpoint (desain)

```
POST /api/ingest   — multipart file PDF
  → {status, pages, chunks, tables, images_captions, elapsed_ms}

POST /api/query    — {question, top_k?}
  → {answer, sources: [{page, type, preview}], model, elapsed_ms}

GET  /api/health   — status + statistik knowledge base
GET  /docs         — Swagger UI (bonus demo)
```

### Strategi Kunci (biar 6 soal lolos semua)

1. **Chunk per-section, bukan per-N-karakter**: teks naratif, tabel (utuh), caption visual per halaman → chunk terpisah dengan `type` berbeda.
2. **Caption VLM ditulis deskriptif**: "Chart komposisi DPK 2025: Tabungan & Tabungan Wadiah 39,31%, Giro & Giro Wadiah 40,12%, Deposito Berjangka 20,57%…" → soal #4 terjawab presisi.
3. **Metadata halaman menempel di SETIAP chunk** → sitasi otomatis (syarat wajib).
4. **Script evaluasi otomatis** `eval/eval_questions.py`: 6 soal resmi → cetak jawaban + halaman sumber → tabel hasil masuk README (bukti untuk juri).

---

## 3. Bagian B — Layout-Aware Text Extraction

### Alur Pipeline

```
Slide JPG (5734×3200)
  1. DETEKSI TEKS     : OCR dengan bounding box (PaddleOCR/EasyOCR —
                        dipilih hasil uji cepat; fallback Tesseract)
  2. EKSTRAKSI STYLE  :
     • warna teks  = piksel dominan di dalam box (exclude warna background
                     lokal) → RGB → hex
     • font-size   = tinggi bounding box (px) → pt/px
     • bold/ketebalan = rasio stroke teks terhadap tinggi box
  3. ANTI-DUPLIKASI (bonus) : mask = semua box teks (dilasi) →
                        OpenCV INPAINT → background BERSIH tanpa teks asli
  4. GENERATE HTML    :
     <div class="bg-layer">  → gambar hasil inpaint (base64)
     <span class="txt" contenteditable>  → absolut posisi (top,left,w,h),
                        warna & font-size terdeteksi
  5. BONUS EDIT       : contenteditable + toggle "Mode Edit" (toolbar kecil)
```

### Deliverable Bagian B
- `notebooks/part_b_pipeline.ipynb` — end-to-end + visualisasi tiap tahap (untuk video demo)
- `pipeline/extract.py` — CLI: `python extract.py slide.jpg -o out.html`
- `outputs/slide-XXXX.html` — 5 hasil konversi (semua slide)
- Metrik visual side-by-side (original vs hasil) di README

---

## 4. Struktur Repo yang Diusulkan

```
ai-engineer-technical-test/
├─ README.md              # arsitektur, cara run, hasil evaluasi, screenshot
├─ part_a_multimodal_rag/
│  ├─ app/main.py         # FastAPI (ingest & query)
│  ├─ pipeline/parse.py   # teks+tabel+layout (PyMuPDF, pdfplumber)
│  ├─ pipeline/vision.py  # render halaman + VLM caption
│  ├─ pipeline/chunk.py   # chunking layout-aware
│  ├─ pipeline/graph.py   # LangChain/LangGraph orchestration
│  ├─ vectorstore.py      # ChromaDB + fastembed
│  ├─ requirements.txt · Dockerfile · .env.example
│  └─ eval/eval_questions.py   # 6 soal otomatis + laporan
├─ part_b_layout_extraction/
│  ├─ notebooks/part_b_pipeline.ipynb
│  ├─ pipeline/extract.py
│  └─ outputs/*.html
├─ assets/                # (gitignore file besar; dokumentasi cara unduh)
└─ docs/architecture.png  # diagram alur
```

---

## 5. Milestone (7 hari)

| Hari | Target | Bukti selesai |
|---|---|---|
| **1** | Setup repo + Part A skeleton: FastAPI jalan, parsing teks & tabel + metadata halaman | `GET /health` ok; dump parse per halaman |
| **2** | Vision caption (VLM) + chunking + embedding + Chroma + `POST /ingest` penuh | ingestion 1 klik via Swagger |
| **3** | `POST /query` + LangGraph + sitasi; **eval 6 soal → semua terjawab benar** | tabel hasil eval |
| **4** | Part B: OCR + style + HTML overlay + inpaint (5 slide) | 5 file HTML + side-by-side |
| **5** | Polish: Dockerfile, README lengkap + diagram, edge case, bonus edit-teks | repo siap |
| **6** | **Video demo** (skrip & storyboard dari saya, kamu rekam ±5–7 mnt) | file video |
| **7** | Buffer + final check + submit | ✅ |

---

## 6. Yang Saya Butuhkan dari Kamu

1. **API key LLM** — paling ideal: **Google Gemini API key (GRATIS)** di aistudio.google.com → untuk VLM caption + synthesis. (Alternatif: Groq/OpenAI.) Tanpa key, saya tetap bisa bikin 90% (parsing, embedding lokal, Chroma) dan synthesis ditunda.
2. **Repo GitHub** — buat repo baru (nama usulan: `ai-engineer-technical-test`) — bisa saya yang buat via akun tixo831 seperti kemarin.
3. **Rekaman video** — kamu yang merekam (wajah/tidak bebas); saya siapkan skrip + storyboard kata per kata.

## 7. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Chart DPK salah asosiasi label↔nilai | layout-aware clustering posisi + verifikasi manual di eval |
| Kuota VLM habis saat demo | cache caption hasil ingestion (cukup 1× jalan); fallback tanpa VLM tetap jalan |
| OCR kurang akurat di slide bergaya | uji 3 engine, pilih terbaik; preprocessing (kontras, upscale) |
| Waktu develop mepet | target H+5 + semua bisa dikerjakan paralel oleh saya di sini |
