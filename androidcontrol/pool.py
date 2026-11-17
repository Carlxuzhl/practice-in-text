"""Which AndroidControl screenshots can be sampled, before any sample is drawn: every screen of the official training
split in the four shards, with its app, the number of elements the tree gives (extract.extract, before the pixel
checks) and its coverage (extract rule 6). A screen is left out, with the first reason that applies:
  'launcher'     a home screen (icons and their names are one node);
  'development'  a screen looked at while the extractor was written and reviewed (development_screens.json);
  'empty'        the tree gives no element;
  'coverage'     coverage under extract.COVERAGE_MIN (0.6), or none (nothing but pictures).
CPU only, no reader calls; --parts processes per shard, each taking every n-th episode (speed only).
Usage: python -m androidcontrol.pool [--parts 4]   -> WORK_DIR/runs/<time>_ac_pool/{meta.json, pool.jsonl, summary.json}"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one thread a process on a shared machine

from androidcontrol import data as L  # noqa: E402
from androidcontrol import extract as X  # noqa: E402
from common.runs import new_run  # noqa: E402

HERE = Path(__file__).resolve().parent


def development() -> set:
    return {tuple(x) for x in json.loads((HERE / "development_screens.json").read_text(encoding="utf-8"))["screens"]}


def shard_job(args) -> list:
    shard, part, parts, dev, split = args
    rows, t0 = [], time.time()
    for i, rec in enumerate(L.records(shard)):
        if i % parts != part:
            continue
        e = L.episode(rec)
        eid = e["episode_id"][0]
        if split.get(eid) != "train":
            continue
        for k, tree in enumerate(e.get("accessibility_trees", [])):
            ws = L.forest(tree)
            app = L.app_of(ws)
            w, h = e["screenshot_widths"][k], e["screenshot_heights"][k]
            row = {"shard": shard.name, "episode_id": eid, "step": k, "app": app, "size": [w, h]}
            if not app or "launcher" in app.lower():
                row["out"] = "launcher"
            elif (eid, k) in dev:
                row["out"] = "development"
            else:
                els, _ = X.extract(ws, w, h)
                row["elements"] = len(els)
                if not els:
                    row["out"] = "empty"
                else:
                    cov = X.coverage(ws, els, bytes(e["screenshots"][k]), w, h)
                    row["coverage"] = cov
                    if cov is None or cov < X.COVERAGE_MIN:
                        row["out"] = "coverage"
            rows.append(row)
    print(f"{time.strftime('%H:%M:%S')} {shard.name} part {part}: {len(rows)} screens, {time.time() - t0:.0f} s", flush=True)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", type=int, default=4)
    a = ap.parse_args()
    missing = [p.name for p in L.shards() if not p.exists()]
    if missing:
        raise SystemExit(f"shards not downloaded: {missing} (androidcontrol/download.sh)")
    split = {i: k for k, ids in json.loads((L.raw_dir() / "splits.json").read_text(encoding="utf-8")).items() for i in ids}
    dev = development()
    run = new_run("ac_pool", {"shards": L.SHARD_NAMES, "data_md5": L.MD5, "extractor": X.VERSION, "coverage_min": X.COVERAGE_MIN,
                              "development_screens": len(dev), "rule": __doc__})
    with mp.get_context("fork").Pool(len(L.shards()) * a.parts) as pool:
        parts = pool.map(shard_job, [(p, k, a.parts, dev, split) for p in L.shards() for k in range(a.parts)])
    rows = [r for part in parts for r in part]
    with (run / "pool.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = [r for r in rows if "out" not in r]
    eps_by_app = collections.defaultdict(set)
    for r in ok:
        eps_by_app[r["app"]].add(r["episode_id"])

    def apps_with(n: int) -> dict:
        apps = {a for a, es in eps_by_app.items() if len(es) >= n}
        sel = [r for r in ok if r["app"] in apps]
        return {"apps": len(apps), "episodes": len({r["episode_id"] for r in sel}), "screens": len(sel)}
    summary = {"screens": len(rows), "episodes": len({r["episode_id"] for r in rows}),
               "left_out": dict(collections.Counter(r.get("out", "kept") for r in rows)),
               "eligible": {"screens": len(ok), "episodes": len({r["episode_id"] for r in ok}), "apps": len(eps_by_app)},
               "eligible_apps_with_at_least_episodes": {str(n): apps_with(n) for n in (1, 2, 3, 5, 8, 10)},
               "top_apps": sorted(((a, len(es)) for a, es in eps_by_app.items()), key=lambda x: -x[1])[:25]}
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "top_apps"}))
    print("RUN_DIR", run)


if __name__ == "__main__":
    main()
