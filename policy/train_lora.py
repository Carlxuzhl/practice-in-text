"""LoRA fine-tuning of the policy on one GPU, on the examples of policy/collect_sft.py. Each example is the prompt as the
chat template gives it to the server (system, user, the generation prompt, thinking on) followed by what the policy
wrote back and its end token; the loss is on what it wrote only. Every round starts from the base model with a fresh
adapter, trained on everything kept so far. The hyperparameters are configs/lora.json, the same for every group; only
--seed differs between training seeds.
Usage: python -m policy.train_lora <examples.jsonl>... --base <model_dir> --out <adapter_dir> [--seed 0] [--device cuda]"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common.config import load

HP = load("lora")


def encode(tok, ex: dict) -> tuple[list[int], list[int]]:
    prompt = tok.apply_chat_template([{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}],
                                     add_generation_prompt=True, tokenize=False, enable_thinking=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(ex["assistant"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
    return p + t, [-100] * len(p) + t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("examples", nargs="+")
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(a.base)
    rows = [json.loads(line) for f in a.examples for line in open(f, encoding="utf-8")]
    data, dropped = [], 0
    for ex in rows:
        ids, labels = encode(tok, ex)
        if len(ids) > HP["max_tokens"]:
            dropped += 1
            continue
        data.append((ids, labels))
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, attn_implementation="sdpa").to(a.device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=HP["lora_r"], lora_alpha=HP["lora_alpha"], lora_dropout=HP["lora_dropout"],
                                             target_modules=HP["targets"], task_type="CAUSAL_LM"))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=HP["lr"], weight_decay=HP["weight_decay"])
    updates = math.ceil(len(data) * HP["epochs"] / HP["accumulate"])
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(updates * HP["warmup"])), updates)
    model.train()
    log, t0, seen, total = [], time.time(), 0, 0.0
    for epoch in range(HP["epochs"]):
        order = list(range(len(data)))
        random.shuffle(order)
        for i, j in enumerate(order):
            ids, labels = data[j]
            x = torch.tensor([ids], device=a.device)
            y = torch.tensor([labels], device=a.device)
            loss = model(input_ids=x, labels=y).loss
            (loss / HP["accumulate"]).backward()
            total += loss.item()
            seen += 1
            if seen % HP["accumulate"] == 0 or (epoch == HP["epochs"] - 1 and i == len(order) - 1):
                torch.nn.utils.clip_grad_norm_(params, HP["grad_clip"])
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                log.append({"examples_seen": seen, "epoch": epoch, "loss": round(total / HP["accumulate"], 4), "lr": sched.get_last_lr()[0],
                            "minutes": round((time.time() - t0) / 60, 1)})
                print(json.dumps(log[-1]), flush=True)
                total = 0.0
    model.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "train_meta.json").write_text(json.dumps({"hp": HP, "seed": a.seed, "base": a.base, "examples": a.examples, "used": len(data),
                                                     "dropped_too_long": dropped, "updates": updates, "minutes": round((time.time() - t0) / 60, 1),
                                                     "log": log}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("ADAPTER", out, "examples", len(data), "dropped", dropped, flush=True)


if __name__ == "__main__":
    main()
