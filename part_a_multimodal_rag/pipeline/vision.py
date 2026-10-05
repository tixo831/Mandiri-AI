"""Captioning visual (VLM) — Bagian A tahap 2.

Render halaman PDF (yang memuat chart/infografis/gambar) menjadi PNG lalu
dianalisis Gemini agar informasi visual — termasuk ASOSIASI label↔nilai pada
chart — menjadi teks yang bisa diretrieval.

Kenapa perlu: pada dokumen uji, angka chart "Komposisi DPK" (hal 6) berada
di text layer tetapi TERPISAH dari labelnya; caption VLM menyatukan kembali
asosiasi tersebut sehingga soal #4 dapat dijawab presisi.

Hasil caption di-cache ke disk (JSON) agar pemanggilan ulang ingestion
tidak memboroskan kuota API.
"""
from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path

import httpx
import pymupdf

_API_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
            "{model}:generateContent")

CAPTION_PROMPT = """Anda asisten ekstraksi data untuk laporan perbankan.
Analisis GAMBAR halaman laporan berikut. Fokus pada elemen VISUAL: chart, grafik,
diagram, infografis, dan gambar (abaikan teks paragraf biasa).

Tuliskan deskripsi PADAT DATA dalam bahasa Indonesia, dengan aturan:
1. Sebutkan judul chart/infografis persis seperti tertulis (jika ada).
2. Untuk chart: sebutkan SETIAP label beserta nilainya dan satuannya, termasuk
   persentase. Tulis asosiasi label→nilai secara eksplisit
   (contoh: "Giro dan Giro Wadiah: 40,12%").
3. Untuk infografis/alur: uraikan URUTAN langkah/alur secara berurutan
   (Langkah 1 → Langkah 2 → ...), sebutkan angka hari/nomor jika ada.
4. Untuk daftar/gambar berisi teks: salin seluruh item beserta detail kontak/
   nomor/alamat jika terlihat.
5. JANGAN mengarang data yang tidak terlihat. Jika elemen visual tidak ada,
   jawab hanya: "TIDAK ADA ELEMEN VISUAL".

Keluaran: teks naratif terstruktur, maksimal 250 kata."""


def _load_env() -> None:
    """Baca .env sederhana (tanpa dependensi tambahan)."""
    for cand in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
        if cand.exists():
            for line in cand.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    os.environ.setdefault(k.strip(), v.strip())
            break


def get_api_key() -> str:
    _load_env()
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY belum di-set (isi file .env)")
    return key


def render_page(pdf_path: str | Path, page: int, dpi: int = 150) -> bytes:
    """Render halaman (1-based) menjadi PNG bytes."""
    doc = pymupdf.open(str(pdf_path))
    try:
        if not 1 <= page <= len(doc):
            raise ValueError(f"halaman {page} di luar 1..{len(doc)}")
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    finally:
        doc.close()


def caption_image(
    png: bytes,
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout: float = 60.0,
) -> str:
    """Kirim PNG ke Gemini → teks caption padat data."""
    key = api_key or get_api_key()
    model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    payload = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.b64encode(png).decode()}},
                {"text": CAPTION_PROMPT},
            ],
        }],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 4096},
    }
    # fallback antar model: 404 (pensiun) / 429 & 503 (overload) → coba berikutnya
    models = [model] + [m for m in os.environ.get(
        "GEMINI_FALLBACKS", "gemini-flash-latest,gemini-3.8-flash").split(",") if m]
    last_err: Exception | None = None
    data = None
    attempt = 0
    max_attempts = 4
    backoffs = [2, 6, 14]  # detik — antisipasi 429/503 transien
    tried: list[tuple[str, int]] = []
    while data is None and attempt < max_attempts:
        for mdl in dict.fromkeys(m.strip() for m in models):  # unik, jaga urutan
            try:
                resp = httpx.post(
                    _API_URL.format(model=mdl),
                    json=payload,
                    headers={"x-goog-api-key": key},
                    timeout=timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except httpx.HTTPStatusError as exc:
                last_err = exc
                tried.append((mdl, exc.response.status_code))
                if exc.response.status_code not in (404, 429, 500, 503):
                    raise
            except httpx.HTTPError as exc:
                last_err = exc
                tried.append((mdl, 0))
        if data is None and attempt < max_attempts - 1:
            time.sleep(backoffs[min(attempt, len(backoffs) - 1)])
        attempt += 1
    if data is None:
        raise RuntimeError(f"Semua model gagal setelah {attempt} percobaan: {last_err}")
    try:
        return "".join(
            part["text"]
            for part in data["candidates"][0]["content"]["parts"]
            if "text" in part
        ).strip()
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Respons Gemini tidak terduga: {str(data)[:300]}") from exc


def _pdf_fingerprint(pdf_path: str | Path) -> str:
    """Hash isi file — kunci cache stabil walau nama file berubah
    (upload API memakai temp file)."""
    import hashlib
    h = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()[:16]


def caption_page(
    pdf_path: str | Path,
    page: int,
    *,
    cache_dir: str | Path = ".caption_cache",
    dpi: int = 150,
) -> str:
    """Render + caption sebuah halaman, dengan cache berdasar hash konten PDF."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"{_pdf_fingerprint(pdf_path)}_p{page}_{dpi}dpi"
    cache_file = cache / f"{tag}.json"
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text())["caption"]
        except Exception:
            pass  # cache rusak → proses ulang
    png = render_page(pdf_path, page, dpi=dpi)
    caption = caption_image(png)
    cache_file.write_text(json.dumps({"page": page, "caption": caption},
                                     ensure_ascii=False, indent=1))
    return caption


if __name__ == "__main__":  # pragma: no cover
    import sys
    pdf_arg = sys.argv[1]
    pages = [int(p) for p in sys.argv[2:]] or [6, 8, 9]
    for p in pages:
        cap = caption_page(pdf_arg, p)
        print(f"===== HAL {p} =====\n{cap}\n")
