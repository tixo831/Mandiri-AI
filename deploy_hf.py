"""Deploy demo statis interaktif ke Hugging Face Spaces (24/7, gratis).
Pakai:  HF_TOKEN=hf_xxx python3 deploy_hf.py"""
import os, sys, time, urllib.request
from pathlib import Path
from huggingface_hub import HfApi

REPO_ID = "Tiko76/mandiri-ai"
FOLDER = Path(__file__).parent / "hf-space"
URL = f"https://tiko76-mandiri-ai.static.hf.space/index.html"

token = os.environ.get("HF_TOKEN")
if not token:
    sys.exit("Set HF_TOKEN=hf_xxx dulu.")

api = HfApi(token=token)
print("1/3 · buat Space (static, gratis)…")
api.create_repo(repo_id=REPO_ID, repo_type="space", space_sdk="static",
                private=False, exist_ok=True)
print("2/3 · unggah file…")
api.upload_folder(folder_path=str(FOLDER), repo_id=REPO_ID,
                  repo_type="space", commit_message="Demo statis interaktif: KB + BM25 retrieval + evaluasi resmi + Bagian B")
print("3/3 · tunggu propagasi…")
ok = False
for i in range(30):
    time.sleep(6)
    try:
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            if r.status == 200:
                ok = True
                break
    except Exception:
        pass
    print(f"   …({(i+1)*6}s)")
print("🎉 LIVE 24/7:", URL if ok else f"cek manual: {URL}")
