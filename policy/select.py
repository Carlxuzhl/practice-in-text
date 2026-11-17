"""Picking episodes by screenshot budget, and run views for policy/collect_sft.py.

calibration-set: the calibration set is the first 2,000 reads of the start policy's real-pipeline episodes on the
  calibration instances, in a fixed order (seed, then template), whole episodes up to and with the one that reaches
  2,000; its first part (direct training's first practice round) is the first 1,000 by the same rule. The other episodes
  are "unused": collected, used by no method, and kept for checks (the acceptance test of an aligned reader, the new
  episodes of the recurrence analysis). Writes calibration_set.json with the lists calibration, first_part,
  second_part (the rest of the calibration set) and unused.
  With fewer than 2,000 reads in all, run the top-up seeds (configs/experiment.json) and pass them with --seeds.
budget: whole episodes of one run in the order (sample, then template), up to and with the one that reaches the budget
  (direct training's second practice round keeps the episodes of its first 1,000 reads).
view: a run directory holding only some episodes of other runs (rows filtered, step files linked), so that
  collect_sft.py takes exactly those.

Usage: python -m policy.select calibration-set --runs RUN... --out calibration_set.json [--seeds 1008,...]
       python -m policy.select budget --run RUN --seed 1018 --samples 6 --budget 1000 --out kept.json
       python -m policy.select view --runs RUN... --episodes FILE --key KEY --out DIR"""
from __future__ import annotations

import argparse
import collections
import json
import shutil
from pathlib import Path

from common.config import load
from common.records import episode_key, episodes_from, rows


def reads_of(run: Path) -> collections.Counter:
    """Reads per episode (template, seed, sample) from the run's ledger."""
    return collections.Counter((x["id"], int(x["seed"]), int(x.get("sample") or 0))
                               for x in map(json.loads, (Path(run) / "vision_ledger.jsonl").open(encoding="utf-8")))


def take_reads(order: list[tuple], reads: collections.Counter, budget: int) -> list[tuple]:
    """Whole episodes in the given order, up to and with the one that reaches the budget."""
    out, total = [], 0
    for ep in order:
        if total >= budget:
            break
        out.append(ep)
        total += reads[ep]
    return out


def calibration_set(runs: list[Path], templates: list[str], seeds: list[int], budget: int, half: int) -> dict:
    reads = collections.Counter()
    by_ep = {}
    for run in runs:
        reads.update(reads_of(run))
        for r in rows(run):
            if episode_key(r) in by_ep:
                raise SystemExit(f"episode {episode_key(r)} is in two runs")
            by_ep[episode_key(r)] = r
    order = [(t, s, 0) for s in seeds for t in templates]
    missing = [ep for ep in order if ep not in by_ep]
    if missing:
        raise SystemExit(f"{len(missing)} calibration episodes are in no run, e.g. {missing[:3]}")
    if sum(reads[ep] for ep in order) < budget:
        raise SystemExit(f"only {sum(reads[ep] for ep in order)} reads, under {budget}: run the next top-up seed and add it to --seeds")
    cal, first = take_reads(order, reads, budget), take_reads(order, reads, half)
    unused = [ep for ep in order if ep not in set(cal)]
    info = lambda eps: [{"id": t, "seed": s, "sample": k, "reads": reads[(t, s, k)], "success": bool(by_ep[(t, s, k)].get("official_success"))}
                        for t, s, k in eps]
    return {"calibration": info(cal), "first_part": info(first), "second_part": info(cal[len(first):]), "unused": info(unused),
            "runs": [Path(r).name for r in runs],
            "reads": {"calibration": sum(reads[e] for e in cal), "first_part": sum(reads[e] for e in first),
                      "second_part": sum(reads[e] for e in cal[len(first):]), "unused": sum(reads[e] for e in unused), "all": sum(reads.values())}}


def view(out: Path, runs: list[Path], keep: set) -> Path:
    """A run directory holding only the episodes in keep; its meta.json is the first run's, with the runs it was made from."""
    if out.exists():
        shutil.rmtree(out)
    (out / "steps").mkdir(parents=True)
    metas = [json.loads((Path(r) / "meta.json").read_text(encoding="utf-8")) for r in runs]
    if len({m["config"]["policy_system"] for m in metas}) != 1:
        raise SystemExit("the runs of a view must share the policy's system prompt")
    multi = metas[0]["config"].get("policy", {}).get("samples", 1) > 1
    found = set()
    with open(out / "rows.jsonl", "w", encoding="utf-8") as f:
        for run in map(Path, runs):
            for r in rows(run):
                key = episode_key(r)
                if key not in keep:
                    continue
                found.add(key)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                stem = f'{r["id"]}_{r["seed"]}' + (f'_k{r.get("sample", 0)}' if multi else "")
                (out / "steps" / f"{stem}.jsonl").symlink_to((run / "steps" / f"{stem}.jsonl").resolve())
    if found != set(keep):
        raise SystemExit(f"view {out.name}: {len(set(keep) - found)} episodes are in no run")
    (out / "meta.json").write_text(json.dumps({**metas[0], "view_of": [Path(r).name for r in runs], "episodes": len(found)}, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
    return out


def main() -> None:
    exp = load("experiment")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    c = sub.add_parser("calibration-set")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--seeds", default=",".join(map(str, exp["seeds"]["calibration"])))
    c.add_argument("--out", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--run", required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--samples", type=int, required=True)
    b.add_argument("--budget", type=int, default=exp["budget"]["direct_round_2"])
    b.add_argument("--out", required=True)
    v = sub.add_parser("view")
    v.add_argument("--runs", nargs="+", required=True)
    v.add_argument("--episodes", required=True)
    v.add_argument("--key", required=True)
    v.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.what == "calibration-set":
        doc = calibration_set([Path(r) for r in a.runs], exp["templates"], [int(s) for s in a.seeds.split(",")],
                              exp["budget"]["screenshots"], exp["budget"]["first_part"])
        Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({"reads": doc["reads"], "episodes": {k: len(doc[k]) for k in ("calibration", "first_part", "second_part", "unused")}}))
    elif a.what == "budget":
        reads = reads_of(Path(a.run))
        order = [(t, a.seed, k) for k in range(a.samples) for t in exp["templates"]]
        have = {episode_key(r) for r in rows(Path(a.run))}
        if set(order) - have:
            raise SystemExit(f"{len(set(order) - have)} of the {len(order)} episodes (sample, then template) have no row in {a.run}")
        keep = take_reads(order, reads, a.budget)
        doc = {"kept": [{"id": t, "seed": s, "sample": k, "reads": reads[(t, s, k)]} for t, s, k in keep], "run": Path(a.run).name,
               "reads_used": sum(reads[e] for e in keep), "reads_collected": sum(reads.values())}
        Path(a.out).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps({k: doc[k] for k in ("reads_used", "reads_collected")} | {"episodes": len(keep)}))
    else:
        print(view(Path(a.out), [Path(r) for r in a.runs], episodes_from(a.episodes, a.key)))


if __name__ == "__main__":
    main()
