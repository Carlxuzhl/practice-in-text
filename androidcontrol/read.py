"""Read sampled AndroidControl screenshots with a served reader, with the reader settings of the MobileGym study
(configs/reader.json: half-size screenshots, greedy decoding, repetition penalty 1.05, at most 4,096 output tokens) and
its prompt (reader.READER_PROMPT, in Chinese; an English translation failed a parsing check on AndroidControl). Every call
goes into vision_ledger.jsonl with its purpose: ac_smoke, ac_original (the original reader) or ac_aligned (the reader
aligned on the alignment set). Reads are resumable: a key already in reads.jsonl of the same directory is not read again.
Usage: python -m androidcontrol.read <out dir> <manifest.jsonl> --purpose ac_original --reader-tag NAME
           [--set align|heldout] [--limit N] [--workers 12]
       (manifest rows: key, set, png path; --limit N: N rows evenly spaced, for a smoke test)"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import threading
import time
from pathlib import Path

from common.config import load, reader_url
from reader import reader as R


def main() -> None:
    call = load("reader")["call"]
    setting = {"scale": call["scale"], "temperature": call["temperature"], "rep_penalty": call["repetition_penalty"], "max_tokens": call["max_tokens"]}
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("manifest")
    ap.add_argument("--purpose", required=True, choices=["ac_smoke", "ac_original", "ac_aligned"])
    ap.add_argument("--reader-tag", required=True)
    ap.add_argument("--set", default="", choices=["", "align", "heldout"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.manifest, encoding="utf-8")]
    if a.set:
        rows = [r for r in rows if r["set"] == a.set]
    if a.limit:
        rows = [rows[i * len(rows) // a.limit] for i in range(min(a.limit, len(rows)))]
    run = Path(a.out_dir)
    run.mkdir(parents=True, exist_ok=True)
    meta_path = run / f"meta_{a.purpose}_{a.set or 'all'}.json"
    meta = {"purpose": a.purpose, "reader_tag": a.reader_tag, "reader_url": reader_url(), "set": a.set or None,
            "limit": a.limit or None, "manifest": a.manifest, "setting": setting, "prompt": R.READER_PROMPT,
            "parser": [R.PARSER_VERSION, R.parser_digest()], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "todo": len(rows)}
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    done = set()
    if (run / "reads.jsonl").exists():
        done = {json.loads(line)["key"] for line in open(run / "reads.jsonl", encoding="utf-8")}
    todo = [r for r in rows if r["key"] not in done]
    lock = threading.Lock()
    out = open(run / "reads.jsonl", "a", encoding="utf-8")
    ledger = open(run / "vision_ledger.jsonl", "a", encoding="utf-8")
    t0, n, failed = time.time(), 0, 0

    def one(r):
        png = Path(r["png"]).read_bytes()
        res = R.read_screen(png, timeout=call["timeout_s"], scale=setting["scale"], repetition_penalty=setting["rep_penalty"],
                            temperature=setting["temperature"], seed=call["seed"], max_tokens=setting["max_tokens"])
        return r, res

    with cf.ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(one, r): r for r in todo}
        for fut in cf.as_completed(futs):
            try:
                r, res = fut.result()
            except Exception as e:  # a failed call is booked as failed and read again on the next run
                failed += 1
                with lock:
                    ledger.write(json.dumps({"purpose": a.purpose, "reader": a.reader_tag, **setting, "t": time.time(),
                                             "key": futs[fut]["key"], "failed": f"{type(e).__name__}: {e}"[:300]}) + "\n")
                    ledger.flush()
                print(f"failed: {type(e).__name__}: {e}", flush=True)
                continue
            with lock:
                out.write(json.dumps({"key": r["key"], "set": r["set"], "raw": res["content"], "bad_lines": res["bad_lines"],
                                      "finish": res["finish"], "usage": res["usage"], "seconds": res["seconds"]}, ensure_ascii=False) + "\n")
                out.flush()
                ledger.write(json.dumps({"purpose": a.purpose, "reader": a.reader_tag, **setting, "t": time.time(), "key": r["key"],
                                         "seconds": res["seconds"], "usage": res["usage"], "finish": res["finish"]}) + "\n")
                ledger.flush()
                n += 1
                if n % 100 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {n}/{len(todo)} read, {time.time() - t0:.0f} s", flush=True)
    out.close()
    ledger.close()
    meta.update({"finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "read_now": n, "failed": failed, "already": len(done)})
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"READ {n} failed {failed} already {len(done)} run {run}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
