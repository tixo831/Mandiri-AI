"""Validasi parsing terhadap 6 pertanyaan evaluasi technical test.
Jalankan: python -m scripts.inspect_parse <path_pdf>
"""
import sys
from pipeline.parse import parse_pdf, parse_summary

CHECKS = [
    (7, ["POJK", "22 Tahun 2023"], "soal #1 peran Unit Pelindungan Nasabah"),
    (7, ["08.00", "20.00"], "soal #2 jam penagihan (21.00 tidak boleh)"),
    (4, ["Tambang", "157.186.029", "7,98"], "soal #3 kredit sektor tambang"),
    (4, ["Konstruksi", "108.148.636", "8,27"], "soal #3 kredit sektor konstruksi"),
    (6, ["KOMPOSISI DPK", "40,12", "20,57", "39,31"], "soal #4 komposisi DPK 2025"),
    (8, ["pengaduan"], "soal #5 alur penanganan pengaduan"),
    (9, ["WhatsApp"], "soal #6 saluran pengaduan"),
]

def main(path: str) -> int:
    sections = parse_pdf(path)
    print(parse_summary(sections), "\n")
    by_page = {}
    for s in sections:
        by_page.setdefault(s.page, []).append(s)
    ok = True
    for page, needles, label in CHECKS:
        blob = "\n".join(s.content for s in by_page.get(page, []))
        missing = [n for n in needles if n.lower() not in blob.lower()]
        status = "OK " if not missing else "MISS"
        if missing:
            ok = False
        print(f"[{status}] hal {page} — {label}" + (f" | kurang: {missing}" if missing else ""))
    print("\nSEMUA CEK LOLOS" if ok else "\nADA YANG KURANG — perlu parsing tambahan")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
