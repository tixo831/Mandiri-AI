"""Layout-Aware Text Extraction — Bagian B.

Pipeline: gambar slide → HTML dengan text layer terpisah dari background.

Tahap:
  1. DOWNSCALE   : 5734px → 2000px (OCR stabil, HTML tetap tajam & ringan;
                   posisi memakai persen → resolusi-independen)
  2. OCR         : RapidOCR → teks + bounding box + confidence
  3. STYLE       : per teks — warna (median piksel tinta di dalam box),
                   font-size (tinggi box → em), bold (densitas tinta)
  4. ANTI-DUPLIKASI (bonus): mask semua box → OpenCV inpaint TELEA
                   → background bersih tanpa teks asli
  5. HTML        : background layer (inpaint, base64) + text layer
                   (<span> absolut persen, warna & font cqw) + toolbar
                   "Mode Edit" (contenteditable — bonus)

CLI:
    python -m pipeline.extract slide.jpg -o out.html
    python -m pipeline.extract slide.jpg --no-inpaint   # CSS-cover mode saja
"""
from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

WORK_WIDTH = 2000          # lebar kerja (px) — seimbang akurasi/memori
INK_DIST = 58              # jarak warna minimal agar piksel dianggap "tinta"
BOLD_DENSITY = 0.30        # rasio tinta utk menganggap teks tebal
MIN_CONF = 0.50            # buang hasil OCR sangat tidak yakin


@dataclass
class TextItem:
    text: str
    x: int; y: int; w: int; h: int          # px pada skala kerja
    color: str = "#000000"                   # hasil deteksi
    font_px: float = 16.0                    # em size (px) pada skala kerja
    bold: bool = False
    conf: float = 0.0


# ---------------------------------------------------------------- tahap 1-2
def load_and_ocr(path: str | Path, work_width: int = WORK_WIDTH):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"gambar tidak terbaca: {path}")
    scale = work_width / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale != 1 else img
    ocr = RapidOCR()
    res, _ = ocr(small)
    items: list[TextItem] = []
    for box, txt, conf in (res or []):
        conf = float(conf)
        if conf < MIN_CONF or not str(txt).strip():
            continue
        xs = [p[0] for p in box]; ys = [p[1] for p in box]
        x, y = int(min(xs)), int(min(ys))
        w, h = int(max(xs) - x), int(max(ys) - y)
        if w < 4 or h < 6:
            continue
        items.append(TextItem(text=str(txt).strip(), x=x, y=y, w=w, h=h,
                              conf=round(conf, 3)))
    return small, items, scale


# ---------------------------------------------------------------- tahap 3
def _box_bg(img: np.ndarray, it: TextItem) -> tuple:
    """Warna latar lokal box (modus piksel di tepi box)."""
    H, W = img.shape[:2]
    x0, y0 = max(it.x - 1, 0), max(it.y - 1, 0)
    x1, y1 = min(it.x + it.w + 1, W), min(it.y + it.h + 1, H)
    roi = img[y0:y1, x0:x1].reshape(-1, 3)
    colors, counts = np.unique(roi, axis=0, return_counts=True)
    return tuple(int(v) for v in colors[counts.argmax()])


def extract_styles(img: np.ndarray, items: list[TextItem]) -> list[TextItem]:
    """Deteksi warna teks, ukuran font, dan ketebalan untuk tiap item."""
    for it in items:
        H, W = img.shape[:2]
        x0, y0 = it.x, it.y
        x1, y1 = min(it.x + it.w, W), min(it.y + it.h, H)
        if x1 <= x0 or y1 <= y0:
            continue
        roi = img[y0:y1, x0:x1].astype(np.int32)
        bg = np.array(_box_bg(img, it), dtype=np.int32)
        dist = np.sqrt(((roi - bg) ** 2).sum(axis=2))
        ink = dist > INK_DIST
        if ink.sum() < 8:            # tak ada tinta jelas → biarkan default
            continue
        # warna tinta INTI = median dari 50% piksel terjauh dari background
        # (menghindari bias piksel anti-aliasing di tepi huruf)
        d = dist[ink]
        if ink.sum() >= 16:
            thr = np.quantile(d, 0.5)
            core = ink & (dist >= thr)
        else:
            core = ink
        b, g, r = [int(np.median(roi[:, :, c][core])) for c in range(3)]
        it.color = f"#{r:02x}{g:02x}{b:02x}"
        it.bold = ink.mean() > BOLD_DENSITY
        # em size ≈ 0.78 × tinggi box (tinggi box ≈ ascender→descender)
        it.font_px = round(it.h * 0.78, 1)
    return items


# ---------------------------------------------------------------- tahap 4
def inpaint_text(img: np.ndarray, items: list[TextItem],
                 dilate: int = 3) -> np.ndarray:
    """Hapus teks asli dari gambar (mask box → inpaint) → background bersih."""
    mask = np.zeros(img.shape[:2], np.uint8)
    for it in items:
        x0 = max(it.x - 1, 0); y0 = max(it.y - 1, 0)
        x1 = min(it.x + it.w + 1, img.shape[1])
        y1 = min(it.y + it.h + 1, img.shape[0])
        mask[y0:y1, x0:x1] = 255
    mask = cv2.dilate(mask, np.ones((dilate, dilate), np.uint8))
    return cv2.inpaint(img, mask, inpaintRadius=4, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------- tahap 5
_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hasil Ekstraksi — {title}</title>
<style>
  :root {{ --bg: #1a1a2e; }}
  * {{ box-sizing: border-box; margin: 0; }}
  body {{ background: var(--bg); font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
         display: grid; place-items: center; min-height: 100vh; padding: 18px; }}
  .stage {{ position: relative; width: min(100%, {work_width}px);
            container-type: inline-size; overflow: hidden; border-radius: 8px;
            box-shadow: 0 12px 40px rgba(0,0,0,.5); }}
  .bg {{ display: block; width: 100%; height: auto; }}
  .bg-orig {{ position: absolute; inset: 0; width: 100%; height: 100%; visibility: hidden; }}
  .stage > .txt {{ position: absolute; }}
  .stage.show-orig .bg-clean {{ visibility: hidden; }}
  .stage.show-orig .bg-orig {{ visibility: visible; }}
  .txt {{ position: absolute; line-height: 1; white-space: pre;
          font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
          transform: translateY(-.08em); }}
  .edit .txt[contenteditable] {{ outline: 1.5px dashed rgba(0,170,255,.85);
          outline-offset: 3px; cursor: text; }}
  .edit .txt[contenteditable]:focus {{ outline-color: #ff3ec8; background: rgba(255,62,200,.12); }}
  .bar {{ position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
          display: flex; gap: 8px; background: #101024; color: #fff;
          padding: 8px 12px; border-radius: 10px; z-index: 10;
          box-shadow: 0 4px 18px rgba(0,0,0,.4); font-size: 13px; align-items: center; }}
  .bar button {{ background: #2962ff; border: 0; color: #fff; font: inherit;
          padding: 6px 12px; border-radius: 8px; cursor: pointer; }}
  .bar button.off {{ background: #444; }}
</style>
</head>
<body>
<div class="bar">
  <span id="info">{n_items} teks terekstraksi</span>
  <button id="bgBtn" class="off">🖼️ Background: Bersih</button>
  <button id="editBtn" class="off">✏️ Mode Edit</button>
  <button id="resetBtn" class="off">↺ Reset teks</button>
</div>
<div class="stage" id="stage">
  <img class="bg bg-clean" src="data:image/jpeg;base64,{b64}" alt="Background bersih (teks asli dihapus via inpainting)">
{bg_orig}{spans}
</div>
<script>
  const bgBtn = document.getElementById('bgBtn');
  const stage = document.getElementById('stage');
  if (bgBtn) bgBtn.onclick = () => {{
    const showOrig = stage.classList.toggle('show-orig');
    bgBtn.textContent = showOrig ? '🖼️ Background: Asli' : '🖼️ Background: Bersih';
    bgBtn.classList.toggle('off', !showOrig);
  }};
  const btn = document.getElementById('editBtn');
  const reset = document.getElementById('resetBtn');
  const spans = [...document.querySelectorAll('.txt')];
  const original = spans.map(s => s.textContent);
  btn.onclick = () => {{
    const on = document.body.classList.toggle('edit');
    btn.classList.toggle('off', !on);
    spans.forEach(s => s.contentEditable = on ? 'true' : 'false');
    btn.textContent = on ? '✅ Selesai Edit' : '✏️ Mode Edit';
  }};
  spans.forEach(s => s.contentEditable = 'false');
  reset.onclick = () => spans.forEach((s, i) => s.textContent = original[i]);
</script>
</body>
</html>
"""


def _jpeg_b64(image: np.ndarray, quality: int) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("gagal encode JPEG background")
    return base64.b64encode(buf.tobytes()).decode()


def generate_html(
    img: np.ndarray,
    items: list[TextItem],
    out_path: str | Path,
    *,
    inpainted: np.ndarray | None = None,
    quality: int = 78,
) -> Path:
    """Tulis file HTML: background + overlay span teks.

    Jika `inpainted` diberikan: dua layer background disematkan
    (bersih [default] & asli) dengan tombol pengalih — mendemonstrasikan
    penanganan teks ganda (opsional) secara interaktif.
    """
    H, W = img.shape[:2]
    b64 = _jpeg_b64(inpainted if inpainted is not None else img, quality)
    bg_orig = ""
    if inpainted is not None:
        bg_orig = ('  <img class="bg bg-orig" src="data:image/jpeg;base64,'
                   + _jpeg_b64(img, quality)
                   + '" alt="Background asli">\n')

    spans = []
    for it in items:
        left = it.x / W * 100
        top = it.y / H * 100
        width = it.w / W * 100
        font_cqw = it.font_px / W * 100
        weight = "800" if it.bold else "500"
        spans.append(
            f'  <span class="txt" style="left:{left:.3f}%;top:{top:.3f}%;'
            f'width:{width:.3f}%;color:{it.color};'
            f'font-size:{font_cqw:.3f}cqw;font-weight:{weight};">'
            f'{it.text}</span>')

    out = Path(out_path)
    out.write_text(_HTML_TEMPLATE.format(
        title=out.stem, work_width=W, n_items=len(items),
        b64=b64, bg_orig=bg_orig, spans="\n".join(spans)), encoding="utf-8")
    return out


# ---------------------------------------------------------------- pipeline
def process_slide(
    image_path: str | Path,
    out_path: str | Path,
    *,
    inpaint: bool = True,
) -> dict:
    """Pipeline lengkap satu slide → HTML. Return ringkasan + item."""
    img, items, scale = load_and_ocr(image_path)
    items = extract_styles(img, items)
    cleaned = inpaint_text(img, items) if inpaint else None
    generate_html(img, items, out_path, inpainted=cleaned)
    return {
        "slide": Path(image_path).name,
        "texts": len(items),
        "avg_conf": round(sum(i.conf for i in items) / len(items), 3) if items else 0,
        "work_size": f"{img.shape[1]}x{img.shape[0]}",
        "inpaint": inpaint,
        "items": [asdict(i) for i in items],
    }


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="path gambar slide (JPG/PNG)")
    ap.add_argument("-o", "--out", default=None, help="file HTML keluaran")
    ap.add_argument("--no-inpaint", action="store_true",
                    help="tanpa inpainting (background asli)")
    a = ap.parse_args()
    out = a.out or (Path(a.image).with_suffix(".html").name)
    info = process_slide(a.image, out, inpaint=not a.no_inpaint)
    print(f"{info['texts']} teks → {out} (inpaint={'ya' if info['inpaint'] else 'tidak'})")
