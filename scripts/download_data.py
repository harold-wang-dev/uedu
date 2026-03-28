"""Download datasets required for UEDU model training.

Requires the Kaggle API to be configured:
    pip install kaggle
    # Place kaggle.json in ~/.kaggle/kaggle.json

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --dataset suicide-detection
    python scripts/download_data.py --dataset asap
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "raw" / "kaggle"

DATASETS = {
    "suicide-detection": {
        "kaggle_id": "nikhileswarkomati/suicide-watch",
        "target_dir": DATA_DIR / "suicide-depression",
        "description": "Kaggle Suicide Detection (232K Reddit posts, CC0)",
        "url": "https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch",
    },
    "asap": {
        "kaggle_id": "lburleigh/asap-2-0",
        "target_dir": DATA_DIR / "asap-aes",
        "description": "ASAP 2.0 Student Essays (24.7K essays, academic use)",
        "url": "https://www.kaggle.com/datasets/lburleigh/asap-2-0",
    },
}


def download_dataset(name: str) -> bool:
    """Download a single dataset using the Kaggle API."""
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print(f"Available: {', '.join(DATASETS.keys())}")
        return False

    ds = DATASETS[name]
    target = ds["target_dir"]

    if target.exists() and any(target.iterdir()):
        print(f"[SKIP] {ds['description']} already exists at {target}")
        return True

    print(f"[DOWNLOAD] {ds['description']}")
    print(f"  Source: {ds['url']}")
    print(f"  Target: {target}")

    target.mkdir(parents=True, exist_ok=True)

    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", ds["kaggle_id"], "-p", str(target), "--unzip"],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            print(f"  [OK] Downloaded successfully")
            return True
        else:
            print(f"  [ERROR] {result.stderr.strip()}")
            return False
    except FileNotFoundError:
        print("  [ERROR] Kaggle CLI not found. Install it:")
        print("    pip install kaggle")
        print("    # Then place your API token at ~/.kaggle/kaggle.json")
        print(f"  Or download manually: {ds['url']}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download UEDU training datasets")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), help="Download a specific dataset")
    args = parser.parse_args()

    if args.dataset:
        download_dataset(args.dataset)
    else:
        print("UEDU Dataset Downloader")
        print("=" * 50)
        print(f"Data directory: {DATA_DIR}\n")

        results = {}
        for name in DATASETS:
            results[name] = download_dataset(name)
            print()

        print("Summary:")
        for name, ok in results.items():
            status = "OK" if ok else "FAILED"
            print(f"  [{status}] {DATASETS[name]['description']}")


if __name__ == "__main__":
    main()
