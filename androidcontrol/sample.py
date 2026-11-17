"""Draw the AndroidControl sample, by a rule fixed before sampling, from the screens androidcontrol.pool found usable
(official training split of shards 0-3; no home screens, no screens seen in development or review, none without
elements, none under coverage 0.6):
 1. Apps: only apps with at least MIN_EPISODES usable episodes, so that an element can recur across episodes.
 2. Order: episodes by sha256("ac-sample-v1|<episode_id>"); an episode goes whole into one set or none.
 3. Alignment set: episodes in that order until the set holds at least ALIGN screens; held-out set: the episodes that
    follow, until it holds at least HELDOUT. An episode whose app already holds CAP_SHARE of a set's target is skipped
    (used by neither set). A screen counts once its text view (with the pixel checks) has an element; screens whose
    text view is empty are left out.
 4. Each screen: the screenshot (PNG as stored), its text view, and the training label (reader.align_data.line_of per
    element, in the layout order), read back with the reader's parser to check that it gives the text view again (any
    difference stops the build). A held-out screen is new if its label is no alignment screen's label.
CPU only, no reader calls.
Usage: python -m androidcontrol.sample <pool run dir> [--parts 4]   -> WORK_DIR/runs/<time>_ac_sample/{meta.json, manifest.jsonl,
       align_examples.jsonl, png/<episode>_<step>.png, summary.json}"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

from androidcontrol import data as L  # noqa: E402
from androidcontrol import extract as X  # noqa: E402
from common.runs import new_run  # noqa: E402
from reader.align_data import line_of  # noqa: E402
from reader.reader import parse_reader  # noqa: E402

MIN_EPISODES = 5
ALIGN, HELDOUT = 2000, 1000
CAP_SHARE = 0.08
MARGIN = 1.4  # text views are computed for this many times the screens needed, before the exact walk
PARTS = 4  # processes per shard, each taking every PARTS-th record (speed only; the sample does not depend on it)


def order_key(eid: int) -> str:
    return hashlib.sha256(f"ac-sample-v1|{eid}".encode()).hexdigest()


def shard_job(args) -> list:
    shard, part, parts, wanted, png_dir = args
    out = []
    for i, rec in enumerate(L.records(shard)):
        if i % parts != part:
            continue
        e = L.episode(rec)
        eid = e["episode_id"][0]
        for k in wanted.get(eid, []):
            ws = L.forest(e["accessibility_trees"][k])
            w, h = e["screenshot_widths"][k], e["screenshot_heights"][k]
            png = bytes(e["screenshots"][k])
            els, _, dropped = X.text_view(ws, png, w, h)
            key = f"{eid}_{k}"
            path = png_dir / f"{key}.png"
            path.write_bytes(png)
            out.append({"key": key, "episode_id": eid, "step": k, "shard": shard.name, "size": [w, h], "png": str(path),
                        "elements": els, "left_out": len(dropped)})
    return out


def label_of(els: list) -> str:
    target = "\n".join(line_of(e) for e in els)
    back = parse_reader(target)[0]["elements"]
    want = []
    for i, e in enumerate(els, 1):
        d = {k: v for k, v in e.items() if k != "id"}
        if d["role"] == "switch":
            d.setdefault("on", False)
        if d["role"] == "mark":
            d.setdefault("filled", False)
        if d["role"] in ("text", "field"):
            d.setdefault("text", "")
        if d.get("empty") is False:
            del d["empty"]
        want.append({"id": i, **d})
    if back != want:
        raise SystemExit(f"label does not parse back: {target[:300]!r}")
    return target


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool")
    ap.add_argument("--parts", type=int, default=PARTS, help="processes per shard (speed only)")
    a = ap.parse_args()
    pool_dir = Path(a.pool)
    rows = [json.loads(l) for l in (pool_dir / "pool.jsonl").open(encoding="utf-8")]
    ok = [r for r in rows if "out" not in r]
    eps = collections.defaultdict(set)
    for r in ok:
        eps[r["app"]].add(r["episode_id"])
    apps = {a for a, s in eps.items() if len(s) >= MIN_EPISODES}
    screens = collections.defaultdict(list)
    app_of = {}
    for r in ok:
        if r["app"] in apps:
            screens[r["episode_id"]].append(r)
            app_of[r["episode_id"]] = r["app"]
    order = sorted(screens, key=order_key)
    need, cand, total = (ALIGN + HELDOUT) * MARGIN, [], 0
    for eid in order:
        cand.append(eid)
        total += len(screens[eid])
        if total >= need:
            break
    run = new_run("ac_sample", {"pool": pool_dir.name, "extractor": X.VERSION, "min_episodes": MIN_EPISODES, "align": ALIGN,
                                "heldout": HELDOUT, "cap_share": CAP_SHARE, "rule": __doc__})
    png_dir = run / "png"
    png_dir.mkdir()
    by_shard = collections.defaultdict(dict)
    for eid in cand:
        for r in screens[eid]:
            by_shard[r["shard"]].setdefault(eid, []).append(r["step"])
    jobs = [(L.raw_dir() / s, k, a.parts, by_shard[s], png_dir) for s in sorted(by_shard) for k in range(a.parts)]
    with mp.get_context("fork").Pool(len(jobs)) as p:
        done = {r["key"]: r for part in p.map(shard_job, jobs) for r in part}
    cov = {f"{r['episode_id']}_{r['step']}": r.get("coverage") for r in ok}
    sets = {"align": [], "heldout": []}
    per_app = {"align": collections.Counter(), "heldout": collections.Counter()}
    skipped, which = [], "align"
    for eid in cand:
        if which == "align" and len(sets["align"]) >= ALIGN:
            which = "heldout"
        if which == "heldout" and len(sets["heldout"]) >= HELDOUT:
            break
        got = [done[f"{eid}_{r['step']}"] for r in sorted(screens[eid], key=lambda r: r["step"])]
        got = [g for g in got if g["elements"]]
        target = ALIGN if which == "align" else HELDOUT
        if per_app[which][app_of[eid]] + len(got) > CAP_SHARE * target:
            skipped.append(eid)
            continue
        for g in got:
            sets[which].append({**g, "set": which, "app": app_of[eid], "coverage": cov[g["key"]]})
        per_app[which][app_of[eid]] += len(got)
    if len(sets["align"]) < ALIGN or len(sets["heldout"]) < HELDOUT:
        raise SystemExit(f"not enough screens: {len(sets['align'])} / {len(sets['heldout'])}; raise MARGIN")
    align_labels = set()
    with (run / "manifest.jsonl").open("w", encoding="utf-8") as man, (run / "align_examples.jsonl").open("w", encoding="utf-8") as ex:
        for which in ("align", "heldout"):
            for g in sets[which]:
                g["target"] = label_of(g["elements"])
                if which == "align":
                    align_labels.add(g["target"])
                    ex.write(json.dumps({"image": g["png"], "target": g["target"], "key": g["key"]}, ensure_ascii=False) + "\n")
                else:
                    g["new_screen"] = g["target"] not in align_labels
                man.write(json.dumps(g, ensure_ascii=False) + "\n")
    used = {g["key"] for w in sets.values() for g in w}
    for p in png_dir.iterdir():
        if p.stem not in used:
            p.unlink()
    summary = {w: {"screens": len(v), "episodes": len({g["episode_id"] for g in v}), "apps": len({g["app"] for g in v}),
                   "elements": sum(len(g["elements"]) for g in v),
                   "roles": dict(collections.Counter(e["role"] for g in v for e in g["elements"])),
                   "largest_app_share": round(max(per_app[w].values()) / len(v), 3)} for w, v in sets.items()}
    summary["heldout"]["new_screens"] = sum(g["new_screen"] for g in sets["heldout"])
    summary["apps_eligible"] = len(apps)
    summary["episodes_skipped_for_cap"] = len(skipped)
    summary["apps_in_both"] = len({g["app"] for g in sets["align"]} & {g["app"] for g in sets["heldout"]})
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary))
    print("RUN_DIR", run)


if __name__ == "__main__":
    main()
