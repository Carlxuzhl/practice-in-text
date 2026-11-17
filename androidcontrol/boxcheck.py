"""How much of the AndroidControl results comes from box widths (descriptive, made after the pre-specified results were
known; no line). Many texts the reader is counted as missing are written with the right text but a box of another
width: the label's box spans the row while the text is short, or the other way round, and the matcher (a text more
than reader.match.FAR from the label's box is another element) then counts the element as missed. CPU only, no reader
calls.
A box-width miss: a true element with text that is counted as missed (1_miss), for which one of the reader's extra
elements (reader.match.match extras; each used once, the most overlapping first) has the same or a similar text
(difflib ratio >= reader.match.SAME_LABEL) and a box overlapping the label's box; "label wider" when the label's box is
wider than the reader's, else "reader wider". Such a reader element is given the label's box, and everything is
computed again from those reads, the control's noise model included.
Reports:
 1. held-out set, original and aligned reader: missed text elements, box-width misses among them, and the element error
    rate and relative drop with the boxes so changed (a wrong-text error if the texts differ);
 2. the recurrence of androidcontrol.analyze (same algorithm, control fitted again) on the original reader's alignment-set
    reads so changed.
Usage: python -m androidcontrol.boxcheck <sample run dir> [--suffix S]   -> <sample>/boxcheck<suffix>.json"""
from __future__ import annotations

import argparse
import collections
import difflib
import json
from pathlib import Path

from androidcontrol import analyze as AA
from reader import match as M
from reader.error_rate import STATUS_Y2, outcomes

CLASSES = ("1_miss", "3_text", "5_state")


def normalized(truth: list, read: list) -> tuple[list, collections.Counter]:
    """The reader's list with each box-width miss's reader element given the label's box; and how many, by kind."""
    outs, _ = outcomes(truth, read)
    extras = set(M.match(truth, read)["extras"])
    new = [dict(r) for r in read]
    kinds = collections.Counter()
    for t, (f, _sig) in zip(truth, outs):
        nt = M.norm(M.text_of(t))
        if not (f.get("1_miss") and nt and t["bbox"][3] > STATUS_Y2):
            continue
        best = None
        for j in extras:
            r = read[j]
            nr = M.norm(M.text_of(r))
            if not nr or not M.overlaps(t["bbox"], r["bbox"]):
                continue
            if nr == nt or difflib.SequenceMatcher(None, nt, nr).ratio() >= M.SAME_LABEL:
                o = M.iou(t["bbox"], r["bbox"])
                if best is None or o > best[0]:
                    best = (o, j)
        if best is not None:
            j = best[1]
            extras.discard(j)
            r = read[j]
            kinds["label wider" if t["bbox"][2] - t["bbox"][0] > r["bbox"][2] - r["bbox"][0] else "reader wider"] += 1
            new[j]["bbox"] = list(t["bbox"])
    return new, kinds


def rate(screens: list, normalize: bool) -> dict:
    n = wrong = text_missed = 0
    kinds = collections.Counter()
    for s in screens:
        read = s["read"]
        if normalize:
            read, k = normalized(s["truth"], read)
            kinds.update(k)
        outs, _ = outcomes(s["truth"], read)
        for t, (f, _sig) in zip(s["truth"], outs):
            if t["bbox"][3] <= STATUS_Y2:
                continue
            n += 1
            wrong += any(f.get(c) for c in CLASSES)
            text_missed += bool(f.get("1_miss")) and bool(M.norm(M.text_of(t)))
    res = {"elements": n, "misread": wrong, "rate": round(wrong / n, 4), "text_missed": text_missed}
    if normalize:
        res["box_width_misses"] = dict(kinds)
    return res


def recurrence_summary(screens: list) -> dict:
    """analyze.recurrence_on on the given screens (the control fitted again on these reads), in this report's keys."""
    r = AA.recurrence_on(screens)
    res = {"screens_used": r["screens_used"], "model_digest": r["model_digest"]}
    for cls in ("any", "1_miss"):
        a, b = r[cls]["reader"], r[cls]["independent"]
        res[cls] = {"reader_q": a["q"], "independent_q": b["q"], "reader_p": a["p"], "independent_p": b["p"],
                    "difference_points": r[cls]["difference_points"], "difference_95_by_app": r[cls]["difference_95_by_app"]}
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    sample, suf = Path(a.sample), a.suffix
    before = AA.screens_of(sample, "reads_original" + suf, "heldout")
    after = AA.screens_of(sample, "reads_aligned" + suf, "heldout")
    res = {"note": "descriptive, made after the pre-specified results were known; no line", "reads_suffix": suf}
    for name, screens in (("original", before), ("aligned", after)):
        res[name] = {"as_scored": rate(screens, False), "box_width_misses_counted_as_read": rate(screens, True)}
    for key in ("as_scored", "box_width_misses_counted_as_read"):
        b, a = res["original"][key]["rate"], res["aligned"][key]["rate"]
        res[f"relative_drop_{key}"] = round(1 - a / b, 4)
    align = AA.screens_of(sample, "reads_original" + suf, "align")
    res["recurrence_as_scored"] = recurrence_summary(align)
    kinds = collections.Counter()
    changed = []
    for s in align:
        read, k = normalized(s["truth"], s["read"])
        kinds.update(k)
        changed.append(dict(s, read=read))
    res["recurrence_box_width_misses_counted_as_read"] = {"box_width_misses": dict(kinds), **recurrence_summary(changed)}
    out = sample / f"boxcheck{suf}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False))
    print("written:", out)


if __name__ == "__main__":
    main()
