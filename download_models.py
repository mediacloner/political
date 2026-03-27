#!/usr/bin/env python3
"""
Download and optionally quantize models to EXL2 format.

Usage:
  # Download pre-quantized EXL2 models from HuggingFace (fastest)
  python download_models.py --all

  # Download specific model
  python download_models.py --model gemma3
  python download_models.py --model deepseek
  python download_models.py --model qwen

  # List available models
  python download_models.py --list

Models are saved to ~/models/ by default.
EXL2 quants are sourced from community repos on HuggingFace (bartowski, turboderp, etc.)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

MODELS_DIR = Path.home() / "models"

# Pre-quantized EXL2 model sources (HuggingFace repos)
# Update these if better quants become available
MODEL_REGISTRY = {
    "gemma3": {
        "description": "Gemma 3 12B — US Delegation (4.5 bpw EXL2)",
        "hf_repo": "bartowski/google_gemma-3-12b-it-exl2",
        "revision": "4_5",  # branch/tag for bpw variant
        "local_name": "gemma-3-12b-exl2-4.5bpw",
        "vram_gb": 7.5,
    },
    "deepseek": {
        "description": "DeepSeek-R1-Distill-Qwen-14B — China Delegation (3.75 bpw EXL2)",
        "hf_repo": "turboderp/DeepSeek-R1-Distill-Qwen-14B-exl2",
        "revision": "3.75bpw",
        "local_name": "deepseek-r1-distill-qwen-14b-exl2-3.75bpw",
        "vram_gb": 8.0,
    },
    "qwen": {
        "description": "Qwen 2.5 14B Instruct — EU Judge (3.5 bpw EXL2)",
        "hf_repo": "turboderp/Qwen2.5-14B-Instruct-exl2",
        "revision": "3.5bpw",
        "local_name": "qwen-2.5-14b-instruct-exl2-3.5bpw",
        "vram_gb": 8.5,
    },
}


def check_hf_cli():
    try:
        subprocess.run(["huggingface-cli", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def download_model(key: str) -> None:
    info = MODEL_REGISTRY[key]
    dest = MODELS_DIR / info["local_name"]

    if dest.exists() and any(dest.iterdir()):
        print(f"  [{key}] already exists at {dest}, skipping")
        return

    dest.mkdir(parents=True, exist_ok=True)
    print(f"\n  [{key}] downloading {info['hf_repo']} (revision: {info['revision']})")
    print(f"          → {dest}")
    print(f"          VRAM: ~{info['vram_gb']} GB")

    if not check_hf_cli():
        # Fall back to Python huggingface_hub
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=info["hf_repo"],
                revision=info.get("revision"),
                local_dir=str(dest),
                ignore_patterns=["*.md", "*.txt", ".gitattributes"],
            )
        except ImportError:
            print("  ERROR: huggingface_hub not installed. Run: pip install huggingface-hub")
            sys.exit(1)
    else:
        cmd = [
            "huggingface-cli", "download",
            info["hf_repo"],
            "--revision", info["revision"],
            "--local-dir", str(dest),
            "--exclude", "*.md", "*.txt",
        ]
        subprocess.run(cmd, check=True)

    print(f"  [{key}] download complete")


def verify_models() -> None:
    print("\nModel verification:")
    total_vram = 0.0
    for key, info in MODEL_REGISTRY.items():
        dest = MODELS_DIR / info["local_name"]
        exists = dest.exists() and any(dest.iterdir())
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {key:10s} {info['local_name']} (~{info['vram_gb']} GB)")
        if exists:
            total_vram += info["vram_gb"]
    print(f"\n  Total VRAM if all loaded simultaneously: ~{total_vram} GB (RAM pre-load)")
    print(f"  VRAM per model during inference: max {max(i['vram_gb'] for i in MODEL_REGISTRY.values())} GB")


def main():
    parser = argparse.ArgumentParser(description="Download EXL2 models for Politics AI Swarm")
    parser.add_argument("--all", action="store_true", help="Download all 3 models")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), help="Download specific model")
    parser.add_argument("--list", action="store_true", help="List models and their status")
    parser.add_argument("--models-dir", default=str(MODELS_DIR), help=f"Models directory (default: {MODELS_DIR})")
    args = parser.parse_args()

    global MODELS_DIR
    MODELS_DIR = Path(args.models_dir)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        verify_models()
        return

    if args.all:
        for key in MODEL_REGISTRY:
            download_model(key)
    elif args.model:
        download_model(args.model)
    else:
        parser.print_help()
        print()
        verify_models()


if __name__ == "__main__":
    main()
