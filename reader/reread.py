"""Read saved screenshots again, offline, with a served reader (another model, the aligned reader, or another
temperature). No environment and no policy: each saved PNG (runs/<run>/shots/<stem>_<step>.png, kept by
policy/episodes.py --png) is read once more and paired with the true list the step recorded, exactly as the original
read was. With temperature > 0 each read gets its own sampling seed (from the episode and the step), so the same screen
read in two episodes is sampled independently.

Writes WORK_DIR/runs/<time>_reread_<tag>/ with meta.json, rows.jsonl (the source rows), steps/<stem>.jsonl (step, v0,
reader_raw, reader_finish) and vision_ledger.jsonl, so that the analysis tools read it as they read the source runs.

Usage: python -m reader.reread <tag> <run_dir>... --reader-tag NAME [--temperature 0] [--scale 0.5] [--rep-penalty 1.05]
       [--workers 12] [--limit N] [--episodes FILE --key KEY]
--limit N: only N of the screenshots, evenly spaced in the sorted list (a smoke test).
--episodes/--key: only the steps of the episodes listed under KEY in FILE (e.g. calibration_set.json and unused), before --limit."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import threading
import time
from pathlib import Path

from common.config import load, reader_url, runs_dir
from common.records import episodes_from, parse_stem
from reader import reader as R


def main() -> None:
    call = load("reader")["call"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--reader-tag", required=True, help="a name for the reader, recorded with every read")
    ap.add_argument("--temperature", type=float, default=call["temperature"])
    ap.add_argument("--scale", type=float, default=call["scale"])
    ap.add_argument("--rep-penalty", type=float, default=call["repetition_penalty"])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--key", default="")
    a = ap.parse_args()
    only = episodes_from(a.episodes, a.key) if a.episodes else None
    out = runs_dir() / f"{time.strftime('%Y%m%d_%H%M%S')}_reread_{a.tag}"
    (out / "steps").mkdir(parents=True)
    todo, rows = [], []
    for run in map(Path, a.runs):
        rows += [json.loads(line) for line in (run / "rows.jsonl").open(encoding="utf-8")]
        for f in sorted((run / "steps").glob("*.jsonl")):
            for rec in map(json.loads, f.open(encoding="utf-8")):
                if "reader_raw" in rec:
                    if only is not None and parse_stem(f.stem) not in only:
                        continue
                    todo.append((run, f.stem, rec["step"], rec["v0"]))
    if a.limit:
        todo = [todo[i * len(todo) // a.limit] for i in range(min(a.limit, len(todo)))]
    meta = {"run_id": out.name, "purpose": "audit", "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": ["reread", a.tag, *a.runs],
            "config": {"cond": "reread", "reader": {"tag": a.reader_tag, "scale": a.scale, "rep_penalty": a.rep_penalty, "temperature": a.temperature,
                                                    "purpose": "audit"}, "limit": a.limit or None,
                       "episodes": [a.episodes, a.key, len(only)] if only is not None else None, "reader_url": reader_url(),
                       "reader_prompt": R.READER_PROMPT, "parser": [R.PARSER_VERSION, R.parser_digest()],
                       "source_runs": [Path(r).name for r in a.runs], "reads": len(todo)}}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lock = threading.Lock()
    results: dict[tuple[str, int], dict] = {}
    ledger = open(out / "vision_ledger.jsonl", "w", encoding="utf-8")

    def one(item):
        run, stem, step, v0 = item
        png = (run / "shots" / f"{stem}_{step}.png").read_bytes()
        seed = int(hashlib.sha256(f"{stem}|{step}|audit".encode()).hexdigest()[:8], 16) if a.temperature > 0 else 0
        r = R.read_screen(png, scale=a.scale, repetition_penalty=a.rep_penalty, temperature=a.temperature, seed=seed,
                          max_tokens=call["max_tokens"], timeout=call["timeout_s"])
        tid, iseed, _ = parse_stem(stem)
        with lock:
            ledger.write(json.dumps({"purpose": "audit", "reader": a.reader_tag, "scale": a.scale, "rep_penalty": a.rep_penalty, "temperature": a.temperature,
                                     "t": time.time(), "id": tid, "seed": iseed, "sample": 0, "step": step, "seconds": r["seconds"],
                                     "usage": r["usage"], "finish": r["finish"], "raw": r["content"]}, ensure_ascii=False) + "\n")
            ledger.flush()
            results[(stem, step)] = {"step": step, "v0": v0, "reader_raw": r["content"], "bad_lines": r["bad_lines"], "reader_finish": r["finish"],
                                     "reader_usage": r["usage"], "reader_seconds": r["seconds"], "reader_seed": seed}

    with cf.ThreadPoolExecutor(a.workers) as pool:
        for fut in cf.as_completed([pool.submit(one, item) for item in todo]):
            fut.result()
    ledger.close()
    by_stem: dict[str, list] = {}
    for (stem, step), rec in results.items():
        by_stem.setdefault(stem, []).append(rec)
    for stem, recs in by_stem.items():
        with open(out / "steps" / f"{stem}.jsonl", "w", encoding="utf-8") as f:
            for rec in sorted(recs, key=lambda x: x["step"]):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["summary"] = {"reads": len(results), "cut": sum(r["reader_finish"] == "length" for r in results.values())}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"RUN_DIR {out} reads {len(results)}")


if __name__ == "__main__":
    main()
