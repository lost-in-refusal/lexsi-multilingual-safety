#!/usr/bin/env python3
"""Cache exactly the checkpoint in configs/runtime.json, or verify it offline."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    # Standard HTTP succeeded on this Mac after an Xet transport failure.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import snapshot_download
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/runtime.json").read_text())
    snapshot = snapshot_download(config["model"], revision=config["model_revision"],
        local_files_only=args.offline,
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    print(f"Cached {config['model']} @ {config['model_revision']}")
    print(snapshot)


if __name__ == "__main__":
    main()
