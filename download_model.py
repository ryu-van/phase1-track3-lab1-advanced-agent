"""
Download Qwen/Qwen2.5-1.5B-Instruct-GGUF (Q4_K_M quantization) from HuggingFace.
Saves to: models/qwen2.5-1.5b-instruct-q4_k_m.gguf

Usage:
    python download_model.py
"""
from __future__ import annotations
import os
import sys
import urllib.request
from pathlib import Path

REPO_ID = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
FILENAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
HF_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"
OUTPUT_DIR = Path(__file__).parent / "models"
OUTPUT_PATH = OUTPUT_DIR / FILENAME


def download_with_progress(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Try huggingface_hub first (faster, supports resuming)
    try:
        from huggingface_hub import hf_hub_download
        print(f"[download] Using huggingface_hub to download {FILENAME} ...")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=str(OUTPUT_DIR),
            local_dir_use_symlinks=False,
        )
        print(f"[download] Saved to: {path}")
        return
    except ImportError:
        print("[download] huggingface_hub not found, falling back to urllib ...")

    # Fallback: urllib with progress bar
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(downloaded / total_size * 100, 100)
            mb = downloaded / 1_048_576
            total_mb = total_size / 1_048_576
            sys.stdout.write(f"\r  {pct:.1f}%  {mb:.1f} / {total_mb:.1f} MB")
            sys.stdout.flush()

    print(f"[download] Downloading from:\n  {url}")
    print(f"[download] Destination: {dest}")
    urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
    print(f"\n[download] Done.")


if __name__ == "__main__":
    if OUTPUT_PATH.exists():
        size_mb = OUTPUT_PATH.stat().st_size / 1_048_576
        print(f"[download] Model already exists: {OUTPUT_PATH}  ({size_mb:.1f} MB)")
        sys.exit(0)

    download_with_progress(HF_URL, OUTPUT_PATH)
    print(f"[download] Model saved to: {OUTPUT_PATH}")
