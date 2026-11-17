"""Merge a reader-alignment LoRA into the reader's weights and save a full model directory that vLLM serves exactly as it
serves the original reader (only the weights differ). Every file of the base directory other than the weights and
config.json (tokenizer, chat template, image processor, generation config) is copied unchanged, so the prompt, template
and image handling stay the original reader's.
Usage: python -m reader.merge_lora --base <model_dir> --adapter <adapter_dir> --out <model_dir> [--device cuda]"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

WEIGHTS = (".safetensors", ".bin", ".pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.bfloat16).to(a.device)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    copied = []
    for f in sorted(base.iterdir()):
        if f.is_file() and not f.name.endswith(WEIGHTS) and f.name not in ("config.json", "model.safetensors.index.json"):
            shutil.copy2(f, out / f.name)
            copied.append(f.name)
    ad = Path(a.adapter) / "adapter_model.safetensors"
    meta = {"base": str(base), "adapter": a.adapter, "adapter_md5": hashlib.md5(ad.read_bytes()).hexdigest()[:12] if ad.exists() else None,
            "copied_from_base": copied, "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
            "seconds": round(time.time() - t0, 1)}
    (out / "merge_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print("MERGED", out, json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
