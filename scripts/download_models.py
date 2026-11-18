"""Download the base models at the revisions in configs/models.json into $MODELS_DIR/<dir>. Weights are not part of this
repository; this uses the Hugging Face Hub (pip install huggingface_hub).
Usage: python scripts/download_models.py [policy reader reader_8b reader_other_family aligned_reader]   (default: policy reader)"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

CONFIG = json.loads((Path(__file__).resolve().parent.parent / "configs" / "models.json").read_text(encoding="utf-8"))


def main() -> None:
    root = Path(os.environ.get("MODELS_DIR") or sys.exit("MODELS_DIR is not set: source configs/paths.env first"))
    for key in sys.argv[1:] or ["policy", "reader"]:
        m = CONFIG[key]
        path = snapshot_download(m["repo"], revision=m["revision"], local_dir=root / m["dir"], allow_patterns=CONFIG["download_patterns"])
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        print(f"{m['repo']}@{m['revision'][:12]} -> {path} ({size / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
