"""Descriptive breakdown of the AndroidControl alignment result: made after the pre-specified result was known, so
nothing here is a test (no line, no interval), only where the element error rate falls and where it does not. CPU only,
no reader calls. Element outcomes as in androidcontrol.analyze (misread = missed, wrong text or wrong state; elements
in the status bar left out). On the held-out set, for the original and the aligned reader, the error rate and its
relative drop, split by:
 1. role: text, icon, image, mark, field, switch.
 2. size: the box's share of the screen (coordinates normalized to 0-1000), in thirds of the held-out elements; the
    cut points are reported.
 3. the app: how many screens of the held-out screen's app the alignment set holds: 0, 1-9, 10-29, 30 or more.
 4. the element: whether an element with the same identity (app, role, text, icon name, state, box to 10 units) is in
    the alignment set's text views.
 5. per app with at least 100 held-out elements: the spread of the relative drop (quartiles, how many apps reach 30%),
    and the five apps with the smallest and the five with the largest drop.
 6. the missed elements: the share where the reader wrote something whose box overlaps the element's box (it wrote
    something else there) rather than nothing at all; overall and by role.
Each group also gives its error classes (missed, wrong text, wrong state) before and after.
Usage: python -m androidcontrol.breakdown <sample run dir> [--suffix S]   -> <sample>/breakdown<suffix>.json"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

from androidcontrol import analyze as AA
from reader.error_rate import STATUS_Y2, ident, outcomes
from reader.match import overlaps

CLASSES = ("1_miss", "3_text", "5_state")
ROLES = ("text", "icon", "image", "mark", "field", "switch")
APP_GROUPS = ((0, 0, "0"), (1, 9, "1-9"), (10, 29, "10-29"), (30, 10 ** 9, "30+"))
MIN_APP_ELEMENTS = 100


def elements(screens: list) -> list:
    """Per counted true element, in a fixed order: its screen, app, the element, its error classes, and (when missed)
    whether the reader wrote anything overlapping its box."""
    out = []
    for s in screens:
        outs, _ = outcomes(s["truth"], s["read"])
        for t, (f, _sig) in zip(s["truth"], outs):
            if t["bbox"][3] <= STATUS_Y2:
                continue
            bad = [c for c in CLASSES if f.get(c)]
            wrote = bool(f.get("1_miss")) and any(overlaps(t["bbox"], r["bbox"]) for r in s["read"])
            out.append({"key": s["key"], "app": s["app"], "el": t, "bad": bad, "wrote_there": wrote})
    return out


def summary(pairs: list) -> dict:
    """pairs: [(original record, aligned record)] of one group; the same elements under both readers."""
    n = len(pairs)
    b = sum(1 for o, _ in pairs if o["bad"])
    a = sum(1 for _, x in pairs if x["bad"])
    res = {"elements": n, "original_misread": b, "aligned_misread": a,
           "original_rate": round(b / n, 4) if n else None, "aligned_rate": round(a / n, 4) if n else None,
           "relative_drop": round(1 - a / b, 4) if b else None}
    for c in CLASSES:
        res[c] = {"original": sum(1 for o, _ in pairs if c in o["bad"]), "aligned": sum(1 for _, x in pairs if c in x["bad"])}
    return res


def area(e: dict) -> float:
    x1, y1, x2, y2 = e["bbox"]
    return max(0, x2 - x1) * max(0, y2 - y1) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sample, suf = Path(a.sample), a.suffix
    before = AA.screens_of(sample, "reads_original" + suf, "heldout")
    after = AA.screens_of(sample, "reads_aligned" + suf, "heldout")
    if [s["key"] for s in before] != [s["key"] for s in after]:
        raise SystemExit("the held-out reads do not pair")
    eb, ea = elements(before), elements(after)
    if len(eb) != len(ea) or any(x["key"] != y["key"] or x["el"] != y["el"] for x, y in zip(eb, ea)):
        raise SystemExit("the held-out elements do not pair")
    pairs = list(zip(eb, ea))
    align = AA.screens_of(sample, "reads_original" + suf, "align")  # only the true lists and the apps are used
    seen = {ident(s["app"], t) for s in align for t in s["truth"] if t["bbox"][3] > STATUS_Y2}
    align_screens = collections.Counter(s["app"] for s in align)

    res = {"note": "descriptive, made after the pre-specified result was known; no line", "reads_suffix": suf,
           "all": summary(pairs)}
    res["by_role"] = {r: summary([p for p in pairs if p[0]["el"]["role"] == r]) for r in ROLES}
    other = [p for p in pairs if p[0]["el"]["role"] not in ROLES]
    if other:
        res["by_role"]["other"] = summary(other)

    areas = sorted(area(o["el"]) for o, _ in pairs)
    cut1, cut2 = areas[len(areas) // 3], areas[2 * len(areas) // 3]
    res["by_size"] = {"cut_points_screen_share": [round(cut1, 6), round(cut2, 6)],
                      "small": summary([p for p in pairs if area(p[0]["el"]) < cut1]),
                      "medium": summary([p for p in pairs if cut1 <= area(p[0]["el"]) < cut2]),
                      "large": summary([p for p in pairs if area(p[0]["el"]) >= cut2])}

    res["by_app_screens_in_alignment"] = {}
    for lo, hi, name in APP_GROUPS:
        group = [p for p in pairs if lo <= align_screens.get(p[0]["app"], 0) <= hi]
        res["by_app_screens_in_alignment"][name] = {"apps": len({p[0]["app"] for p in group}),
                                                    "screens": len({p[0]["key"] for p in group}), **summary(group)}

    res["by_element_seen_in_alignment"] = {
        "seen": summary([p for p in pairs if ident(p[0]["app"], p[0]["el"]) in seen]),
        "not_seen": summary([p for p in pairs if ident(p[0]["app"], p[0]["el"]) not in seen])}

    per_app = collections.defaultdict(list)
    for p in pairs:
        per_app[p[0]["app"]].append(p)
    apps = {a: summary(ps) for a, ps in per_app.items() if len(ps) >= MIN_APP_ELEMENTS}
    drops = sorted((s["relative_drop"], a) for a, s in apps.items() if s["relative_drop"] is not None)
    q = statistics.quantiles([d for d, _ in drops], n=4)
    keep = lambda a: {"app": a, "elements": apps[a]["elements"], "screens_in_alignment": align_screens.get(a, 0),
                      "original_rate": apps[a]["original_rate"], "aligned_rate": apps[a]["aligned_rate"],
                      "relative_drop": apps[a]["relative_drop"]}
    res["per_app"] = {"apps": len(drops), "min_elements": MIN_APP_ELEMENTS,
                      "drop_quartiles": [round(x, 4) for x in q],
                      "apps_reaching_30pct": sum(1 for d, _ in drops if d >= 0.30),
                      "apps_worse_after": sum(1 for d, _ in drops if d < 0),
                      "smallest_drop": [keep(a) for _, a in drops[:5]],
                      "largest_drop": [keep(a) for _, a in drops[-5:][::-1]]}

    def wrote(recs: list) -> dict:
        missed = [r for r in recs if "1_miss" in r["bad"]]
        return {"missed": len(missed), "wrote_something_there": sum(r["wrote_there"] for r in missed),
                "share": round(sum(r["wrote_there"] for r in missed) / len(missed), 4) if missed else None}

    res["missed_elements"] = {}
    for name, recs in (("original", eb), ("aligned", ea)):
        res["missed_elements"][name] = {"all": wrote(recs),
                                        **{r: wrote([x for x in recs if x["el"]["role"] == r]) for r in ROLES}}
    out = sample / f"breakdown{suf}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["all"], ensure_ascii=False))
    print("written:", out)


if __name__ == "__main__":
    main()
