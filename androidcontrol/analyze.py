"""The two pre-specified AndroidControl measurements (CPU only, no reader calls), from the sample (androidcontrol.sample)
and the reads (androidcontrol.read) in <sample>/reads_original and <sample>/reads_aligned.
1. Recurrence, on the original reader's reads of the alignment set, with the method of reader/recurrence.py: an
   element's identity is app, role, text, icon name, state and box to 10 units, counted once per episode at its first
   read; screens whose output was cut or had over 60 extras are left out. The control fits the rate-matched noise
   model (practice/noise.py) on the same reads and applies it with independent draws (keyed by the episode and the
   screen's content). q = the share of misread elements misread again in another episode; difference = reader minus
   control, in points; 95% interval from 1,000 app resamples. Line: at least 20 points and a lower bound above 0, for
   "any" error; class 1 (missed) is reported beside it.
2. Alignment, on the held-out set: the element error rate = (missed + wrong text + wrong state) / true elements, for the
   original and the aligned reader; an unparsable screenshot counts all its elements as missed. Line: a relative drop
   of at least 30% and at most 1% unparsable screenshots. New screens (labels in no alignment example) are reported
   apart, descriptively.
Also counts the reader calls by purpose from the ledgers.
Usage: python -m androidcontrol.analyze <sample run dir> [recurrence|alignment|all] [--suffix S]
       -> <sample>/results_<part><suffix>.json   (--suffix: read the reads_original<S> and reads_aligned<S> folders)"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from common.config import load
from practice import noise as N
from reader import recurrence as RC
from reader.error_rate import STATUS_Y2, ident, misreads, outcomes
from reader.reader import parse_reader
from textview.textview import render

CFG = load("reader")
LINES = {"recurrence_points_at_least": CFG["recurrence"]["threshold_points"], "recurrence_low_above": 0,
         "drop_at_least": CFG["acceptance"]["misread_reduction_at_least"], "unparsable_at_most": CFG["acceptance"]["unparsable_at_most"]}


def app_id(pkg: str) -> str:
    """The noise model groups extras by the part of an id before its first dot; a package name keeps its dots as _."""
    return pkg.replace(".", "_")


def screens_of(sample: Path, reads: str, which: str) -> list:
    rows = [json.loads(line) for line in (sample / "manifest.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if r["set"] == which]
    got = {}
    for line in (sample / reads / "reads.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        got[r["key"]] = r
    missing = [r["key"] for r in rows if r["key"] not in got]
    if missing:
        raise SystemExit(f"{reads}: {len(missing)} of {len(rows)} {which} screenshots not read")
    out = []
    for r in rows:
        g = got[r["key"]]
        out.append({"key": r["key"], "episode": r["episode_id"], "step": r["step"], "app": r["app"], "id": app_id(r["app"]) + ".screen",
                    "truth": r["elements"], "read": parse_reader(g["raw"])[0]["elements"], "cut": g["finish"] == "length",
                    "new": r.get("new_screen")})
    return sorted(out, key=lambda s: (s["episode"], s["step"]))


def recurrence_on(screens: list) -> dict:
    """Recurrence of the reader's misreads on these screens against independent errors fitted on the same reads."""
    model = N.Calibrated(N.fit(screens))
    out = []
    for s in screens:
        truth, n = s["truth"], len(s["truth"])
        content = hashlib.sha256(render({"elements": truth}).encode()).hexdigest()[:16]
        reader, extras = outcomes(truth, s["read"])
        sim = model.apply({"elements": truth}, f'{s["episode"]}|{content}', app=app_id(s["app"]))["elements"]
        control, _ = outcomes(truth, sim)
        els = [{"ident": ident(s["app"], t), "feat": N.features(t, n), "reader": a, "control": b}
               for t, a, b in zip(truth, reader, control) if t["bbox"][3] > STATUS_Y2]
        out.append({"episode": s["episode"], "step": s["step"], "cut": s["cut"], "extras": extras, "content": content, "els": els})
    kept = [s for s in out if not s["cut"] and s["extras"] <= RC.MAX_EXTRAS]
    eps = {s["episode"] for s in kept}
    occ = {who: RC.occurrences(kept, who, eps) for who in ("reader", "control")}
    res = {"screens": len(screens), "screens_used": len(kept), "left_out_cut": sum(s["cut"] for s in out),
           "left_out_extras": sum(1 for s in out if not s["cut"] and s["extras"] > RC.MAX_EXTRAS), "episodes": len(eps),
           "apps": len({s["app"] for s in screens}), "model_digest": model.digest}
    for cls in ("any", "1_miss", "3_text"):
        a, b = RC.persistence(RC.per_element(occ["reader"], cls)), RC.persistence(RC.per_element(occ["control"], cls))
        res[cls] = {"reader": a, "independent": b,
                    "difference_points": None if a["q"] is None or b["q"] is None else round(100 * (a["q"] - b["q"]), 1)}
    for cls in ("any", "1_miss"):
        res[cls]["difference_95_by_app"] = RC.boot_q_diff(occ["reader"], occ["control"], cls)
    return res


def recurrence(sample: Path, suf: str = "") -> dict:
    res = recurrence_on(screens_of(sample, "reads_original" + suf, "align"))
    d, ci = res["any"]["difference_points"], res["any"]["difference_95_by_app"]["interval"]
    res["line"] = {"points_at_least_20": d is not None and d >= LINES["recurrence_points_at_least"],
                   "low_above_0": ci is not None and ci[0] > LINES["recurrence_low_above"]}
    res["line"]["met"] = all(res["line"].values())
    return res


def alignment(sample: Path, suf: str = "") -> dict:
    before, after = screens_of(sample, "reads_original" + suf, "heldout"), screens_of(sample, "reads_aligned" + suf, "heldout")
    if [s["key"] for s in before] != [s["key"] for s in after]:
        raise SystemExit("the held-out reads do not pair")
    b, a = misreads(before), misreads(after)
    res = {"original": b, "aligned": a, "relative_drop": round(1 - a["rate"] / b["rate"], 4),
           "unparsable_share": round(a["unparsable_screens"] / a["screens"], 4)}
    res["line"] = {"drop_at_least_30pct": res["relative_drop"] >= LINES["drop_at_least"],
                   "unparsable_at_most_1pct": res["unparsable_share"] <= LINES["unparsable_at_most"]}
    res["line"]["met"] = all(res["line"].values())
    nb, na = [s for s in before if s["new"]], [s for s in after if s["new"]]
    res["new_screens"] = {"screens": len(nb), "original": misreads(nb), "aligned": misreads(na)}
    if nb:
        res["new_screens"]["relative_drop"] = round(1 - res["new_screens"]["aligned"]["rate"] / res["new_screens"]["original"]["rate"], 4)
    return res


def ledger(sample: Path) -> dict:
    calls = collections.Counter()
    for f in sample.glob("reads_*/vision_ledger.jsonl"):
        for line in f.open(encoding="utf-8"):
            r = json.loads(line)
            calls[(r["purpose"], "failed" if r.get("failed") else "answered")] += 1
    return {f"{p} {k}": v for (p, k), v in sorted(calls.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample")
    ap.add_argument("part", nargs="?", default="all", choices=["recurrence", "alignment", "all"])
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sample = Path(a.sample)
    res = {"lines": LINES, "reads_suffix": a.suffix, "reader_calls": ledger(sample)}
    if a.part in ("recurrence", "all"):
        res["recurrence"] = recurrence(sample, a.suffix)
    if a.part in ("alignment", "all"):
        res["alignment"] = alignment(sample, a.suffix)
    out = sample / f"results_{a.part}{a.suffix}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False)[:3000])
    print("written:", out)


if __name__ == "__main__":
    main()
