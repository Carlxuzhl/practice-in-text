"""The element error rate, and the acceptance test of an aligned reader.

Per true element below the status bar, the element is misread when the reader misses it, gives it the wrong text or
gives it the wrong state (classes 1_miss, 3_text, 5_state of outcomes()). Every screenshot counts; one the parser cannot
read at all counts all its elements as missed. The element error rate is misread / elements.

Acceptance test (configs/reader.json "acceptance"): on the same screenshots, the aligned reader's error rate is at least 30%
lower than the original reader's (relative), and at most 1% of its screenshots are unparsable.

Usage:
  python -m reader.error_rate --runs RUN...                         error rate of the reads in these runs
  python -m reader.error_rate --before RUN... --after RUN [--episodes calibration_set.json --key unused] [--out report.json]
      before: the runs whose own reads are the original reader's (e.g. the calibration-set collection);
      after: a reader/reread.py run of the same screenshots with the aligned reader; screens are paired by
      (template, seed, sample, step) and must have the same true list."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from common.config import load
from common.records import episodes_from, screens_of
from reader import match as M

STATUS_Y2 = load("reader")["recurrence"]["status_bar_y2"]  # elements ending above this line are the status bar


def ident(app: str, e: dict) -> tuple:
    """An element's identity across episodes: app, role, text, icon name, switch or mark state, box to 10 units."""
    state = e.get("on") if e["role"] == "switch" else e.get("filled") if e["role"] == "mark" else None
    return (app, e["role"], M.norm(M.text_of(e)), (e.get("name") or "").lower(), None if state is None else bool(state),
            tuple(int(round(v / 10)) for v in e["bbox"]))


def outcomes(truth: list[dict], read: list[dict]) -> tuple[list[tuple[dict, tuple]], int]:
    """Per true element: the error classes that apply, each with whether it went wrong, and how the reader wrote it.
    Classes: 1_miss left out; 2_shape a coloured text written as a shape; 3_text text misread or merged; 5_state switch or
    mark in the wrong state (2, 3 and 5 only among elements not left out); 4_renamed an icon renamed (reported apart);
    "any" = 1, 2, 3 or 5. Also returns the number of extra reader elements."""
    m = M.match(truth, read)
    out = []
    for t, (kind, j) in zip(truth, m["outcomes"]):
        r = read[j] if j is not None else None
        has_text = bool(M.norm(M.text_of(t)))
        f = {"1_miss": kind == "miss"}
        if kind != "miss":
            if has_text and M.coloured(t):
                f["2_shape"] = kind == "shape"
            if has_text and kind in ("kept", "merged"):
                f["3_text"] = kind == "merged" or M.norm(M.text_of(r)) != M.norm(M.text_of(t))
            if t["role"] in ("switch", "mark") and kind == "kept" and r["role"] == t["role"]:
                k = "on" if t["role"] == "switch" else "filled"
                f["5_state"] = bool(r.get(k)) != bool(t.get(k))
            if t["role"] == "icon" and t.get("name") and kind == "kept":
                f["4_renamed"] = not (r["role"] == "icon" and M.same_name(r.get("name"), t["name"]))
        f["any"] = any(f.get(c, False) for c in ("1_miss", "2_shape", "3_text", "5_state"))
        sig = ("miss",) if kind == "miss" else (kind, r["role"], M.norm(M.text_of(r)), (r.get("name") or "").lower(),
                                                 r.get("on"), r.get("filled"))
        out.append((f, sig))
    return out, len(m["extras"])


def misreads(screens: list[dict]) -> dict:
    """The element error rate over screens {"truth", "read", "cut"}."""
    n = wrong = 0
    by = collections.Counter()
    for s in screens:
        outs, _ = outcomes(s["truth"], s["read"])
        for t, (f, _sig) in zip(s["truth"], outs):
            if t["bbox"][3] <= STATUS_Y2:
                continue
            n += 1
            bad = [c for c in ("1_miss", "3_text", "5_state") if f.get(c)]
            wrong += bool(bad)
            by.update(bad)
    return {"elements": n, "misread": wrong, "rate": round(wrong / n, 4) if n else None, "by_class": dict(by),
            "unparsable_screens": sum(1 for s in screens if not s["read"]), "cut_screens": sum(1 for s in screens if s["cut"]), "screens": len(screens)}


def key(s: dict) -> tuple:
    return s["id"], s["seed"], s["sample"], s["step"]


def compare(before_runs: list[Path], after_run: Path, only: set | None = None) -> dict:
    """The acceptance test: the same screenshots read by the original reader (before) and the aligned one (after)."""
    lines = load("reader")["acceptance"]
    before = {key(s): s for s in screens_of(before_runs, only=only)}
    after = {key(s): s for s in screens_of([after_run])}
    if set(before) != set(after) or any(before[k]["truth"] != after[k]["truth"] for k in after):
        raise SystemExit(f"the screenshots do not pair: before {len(before)}, after {len(after)}")
    keys = sorted(after)
    b, a = misreads([before[k] for k in keys]), misreads([after[k] for k in keys])
    reduction = 1 - a["rate"] / b["rate"] if b["rate"] else None  # None: the original reader made no error to reduce
    unparsable = a["unparsable_screens"] / a["screens"]
    res = {"after_run": Path(after_run).name, "screenshots": len(keys), "before": b, "after": a,
           "relative_reduction": None if reduction is None else round(reduction, 4), "unparsable_share": round(unparsable, 4), "lines": lines}
    res["checks"] = {"misread_down_at_least_30pct": reduction is not None and reduction >= lines["misread_reduction_at_least"],
                     "unparsable_at_most_1pct": unparsable <= lines["unparsable_at_most"]}
    res["passed"] = all(res["checks"].values())
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=[])
    ap.add_argument("--before", nargs="+", default=[])
    ap.add_argument("--after", default="")
    ap.add_argument("--episodes", default="", help="JSON file listing episodes (e.g. calibration_set.json)")
    ap.add_argument("--key", default="", help="the list in --episodes to use (e.g. unused)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    only = episodes_from(a.episodes, a.key) if a.episodes else None
    if a.runs:
        res = misreads(screens_of([Path(r) for r in a.runs], only=only))
    elif a.before and a.after:
        res = compare([Path(r) for r in a.before], Path(a.after), only)
    else:
        raise SystemExit("give --runs, or --before and --after")
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
