"""Reader alignment: LoRA on the reader (Qwen3-VL-4B-Instruct), trained to write each calibration screenshot as its true
text view (reader/align_data.py).
- Input: the saved PNG halved with LANCZOS (as reader.read_screen does at scale 0.5) and the reader prompt in the model's
  chat template, as the reader server receives it.
- Target: the label and the end token; the loss is on the target only.
- Hyperparameters: configs/lora.json, the same as policy training: rank 16, alpha 32, dropout 0.05, lr 1e-4, 2 epochs,
  one example a step, 16 accumulated, 3% warmup, cosine, no weight decay, clipping 1.0, the last checkpoint. LoRA on the
  language model's q/k/v/o/gate/up/down projections only; the vision encoder and its merger have other layer names
  (qkv, proj, linear_fc1/2) and stay frozen.
Usage: python -m reader.train_lora <examples.jsonl> --base <model_dir> --out <adapter_dir> [--seed 0] [--limit N] [--device cuda]
       (--limit N: train on the first N examples only, a smoke test)"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, get_cosine_schedule_with_warmup

from common.config import load
from reader.reader import READER_PROMPT

HP = load("lora")
SCALE = load("reader")["call"]["scale"]


def image_of(path: str) -> Image.Image:
    im = Image.open(path)
    im.load()
    return im.resize((round(im.width * SCALE), round(im.height * SCALE)), Image.LANCZOS)


def encode(proc, ex: dict) -> dict:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": READER_PROMPT}]}]
    prompt = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=[prompt], images=[image_of(ex["image"])], return_tensors="pt")
    target = proc.tokenizer(ex["target"] + "<|im_end|>", add_special_tokens=False, return_tensors="pt")["input_ids"]
    ids = torch.cat([enc["input_ids"], target], dim=1)
    labels = torch.cat([torch.full_like(enc["input_ids"], -100), target], dim=1)
    if "mm_token_type_ids" not in enc:
        raise SystemExit("the processor gave no mm_token_type_ids, which Qwen3-VL needs for its multimodal positions")
    mm = torch.cat([enc["mm_token_type_ids"], torch.zeros_like(target)], dim=1)  # the target is text
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "mm_token_type_ids": mm,
            "pixel_values": enc["pixel_values"], "image_grid_thw": enc["image_grid_thw"], "prompt_tokens": enc["input_ids"].shape[1]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("examples")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    proc = AutoProcessor.from_pretrained(a.base)
    rows = [json.loads(line) for line in open(a.examples, encoding="utf-8")]
    if a.limit:
        rows = rows[:a.limit]
    model = AutoModelForImageTextToText.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(a.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
                                             target_modules=HP["targets"], task_type="CAUSAL_LM"))
    wrapped = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    outside = [n for n in wrapped if "language_model" not in n]
    if outside or len(wrapped) != 7 * model.config.text_config.num_hidden_layers:
        raise SystemExit(f"LoRA on {len(wrapped)} layers, {len(outside)} outside the language model: {outside[:3]}")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=HP["weight_decay"])
    updates = math.ceil(len(rows) * HP["epochs"] / HP["accumulate"])
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(updates * HP["warmup"])), updates)
    model.train()
    log, t0, seen, total, dropped, longest = [], time.time(), 0, 0.0, 0, 0
    for epoch in range(HP["epochs"]):
        order = list(range(len(rows)))
        random.shuffle(order)
        for i, j in enumerate(order):
            b = encode(proc, rows[j])
            n = b["input_ids"].shape[1]
            longest = max(longest, n)
            last = epoch == HP["epochs"] - 1 and i == len(order) - 1
            if n > HP["max_tokens"]:
                dropped += 1
            else:
                inputs = {k: v.to(a.device) for k, v in b.items() if k != "prompt_tokens"}
                loss = model(**inputs).loss
                (loss / HP["accumulate"]).backward()
                total += loss.item()
            seen += 1
            if seen % HP["accumulate"] == 0 or last:
                torch.nn.utils.clip_grad_norm_(params, HP["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                log.append({"examples_seen": seen, "epoch": epoch, "loss": round(total / HP["accumulate"], 4), "lr": sched.get_last_lr()[0],
                            "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                total = 0.0
    model.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps({"hp": HP, "seed": a.seed, "base": a.base, "examples": a.examples, "limit": a.limit or None,
                                                     "used": len(rows) - dropped // HP["epochs"], "dropped_too_long_per_epoch": dropped // HP["epochs"],
                                                     "longest_tokens": longest, "lora_layers": len(wrapped), "updates": updates,
                                                     "minutes": round((time.time() - t0) / 60, 1), "log": log}, ensure_ascii=False, indent=1),
                                         encoding="utf-8")
    print("ADAPTER", out, "examples", len(rows), "dropped", dropped // HP["epochs"], flush=True)


if __name__ == "__main__":
    main()
