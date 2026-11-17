"""Training examples for reader alignment (CPU only, no model).

Each calibration screenshot (every read step of the calibration-set episodes listed in calibration_set.json, with the
PNG the collection kept) is paired with the true text view the step recorded, written in the reader's own output format: one
line per element in the format of reader.READER_PROMPT, in the layout order, so that reader.parse_reader reads it back
as the list the practice environment shows. Fields the parser cannot express (round, raster) are left out, and boxes are
clipped to 0-1000 as the parser clips them (an element whose clipped box is empty is not on the screen and is left out;
counted).

Before writing, every label is read back with the same parser and compared with the true list (ids renumbered, those
two fields dropped); any mismatch stops the build. Also writes the first N screenshots in episode order (the
calibration set's order, steps in order) for the data-efficiency variant.

Usage: python -m reader.align_data --calibration-set calibration_set.json [--out DIR] [--expect 2004] [--first 500]
       (writes train.jsonl, train_first<N>.jsonl and align_data.json; DIR defaults to WORK_DIR/data/align)"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path

from common.config import runs_dir, work_dir
from common.records import parse_stem
from reader.reader import parse_reader
from textview.textview import reading_order

DROPPED_FIELDS = ("round", "raster")  # the parser has no word for them


def clipped(e: dict) -> dict | None:
    """The element with its box clipped to 0-1000 (the prompt's coordinates; parse_reader clips the same way), or None
    when nothing of it is left on the screen."""
    x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in e["bbox"])
    if x2 <= x1 or y2 <= y1:
        return None
    return {**e, "bbox": [x1, y1, x2, y2]}


def visible(truth: list[dict]) -> list[dict]:
    return [c for c in (clipped(e) for e in reading_order([dict(e) for e in truth])) if c is not None]


def line_of(e: dict) -> str:
    """One element as the reader prompt asks for it."""
    role = e["role"]
    words: list[str] = [role]
    if role in ("text", "field"):
        words.append(json.dumps(e.get("text", ""), ensure_ascii=False))
    if role == "icon" and e.get("name"):
        words.append(e["name"])
    if role == "switch":
        words.append("on" if e.get("on") else "off")
    if role == "mark":
        words.append("filled" if e.get("filled") else "empty")
    if e.get("color"):
        words.append(e["color"])
    if e.get("bg"):
        words.append(f"on-{e['bg']}")
    if e.get("bold"):
        words.append("bold")
    if e.get("dim"):
        words.append("dim")
    if role == "field" and e.get("empty"):
        words.append("empty")
    x1, y1, x2, y2 = e["bbox"]
    return " ".join(words) + f" ({x1},{y1},{x2},{y2})"


def target_of(truth: list[dict]) -> str:
    """The training label of a screen: its true text view in the reader's output format."""
    return "\n".join(line_of(e) for e in visible(truth))


def expected(truth: list[dict]) -> list[dict]:
    """What the parser should give back: the list in layout order, ids 1..n, without the fields it cannot express."""
    out = []
    for i, e in enumerate(visible(truth), 1):
        d = {k: v for k, v in e.items() if k not in DROPPED_FIELDS and k != "id"}
        if d["role"] == "switch":
            d.setdefault("on", False)
        if d["role"] == "mark":
            d.setdefault("filled", False)
        if d["role"] in ("text", "field"):
            d.setdefault("text", "")
        for k in ("bold", "dim", "empty"):
            if d.get(k) is False:
                del d[k]
        out.append({"id": i, **d})
    return out


def check(truth: list[dict]) -> tuple[bool, dict]:
    """Whether the label of a screen reads back as its true list."""
    target = target_of(truth)
    got, bad = parse_reader(target)
    exp = expected(truth)
    ok = not bad and got["elements"] == exp
    return ok, {"bad_lines": bad, "lens": [len(exp), len(got["elements"])],
                "first_diffs": [(a, b) for a, b in zip(exp, got["elements"]) if a != b][:2]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration-set", required=True, help="calibration_set.json (python -m policy.select calibration-set)")
    ap.add_argument("--out", default="")
    ap.add_argument("--expect", type=int, default=0, help="stop unless there are exactly this many screenshots")
    ap.add_argument("--first", type=int, default=500)
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else work_dir() / "data" / "align"
    cal = json.loads(Path(a.calibration_set).read_text(encoding="utf-8"))
    order = [(e["id"], e["seed"], e.get("sample", 0)) for e in cal["calibration"]]
    want = set(order)
    found: dict[tuple, list[dict]] = collections.defaultdict(list)
    for run in cal["runs"]:
        run_dir = runs_dir() / run
        for f in sorted((run_dir / "steps").glob("*.jsonl")):
            ep = parse_stem(f.stem)
            if ep not in want:
                continue
            for rec in map(json.loads, f.open(encoding="utf-8")):
                if "reader_raw" not in rec:
                    continue
                png = run_dir / "shots" / f"{f.stem}_{rec['step']}.png"
                found[ep].append({"image": str(png), "run": run, "episode": f.stem, "id": ep[0], "seed": ep[1], "sample": ep[2],
                                  "step": rec["step"], "truth": rec["v0"], "after_shot_differs": rec.get("v0_after_shot") != rec["v0"]})
    rows, problems, first_bad = [], collections.Counter(), []
    for ep in order:
        for r in sorted(found[ep], key=lambda x: x["step"]):
            if not os.path.exists(r["image"]):
                raise SystemExit(f"missing screenshot {r['image']}")
            ok, info = check(r["truth"])
            if not ok:
                problems["mismatch"] += 1
                if len(first_bad) < 5:
                    first_bad.append({"episode": r["episode"], "step": r["step"], **info})
            problems["dropped_fields"] += sum(1 for e in r["truth"] for k in DROPPED_FIELDS if k in e)
            problems["boxes_clipped"] += sum(1 for e in r["truth"] if any(v < 0 or v > 1000 for v in e["bbox"]))
            problems["elements_off_screen"] += len(r["truth"]) - len(expected(r["truth"]))
            rows.append({k: r[k] for k in ("image", "run", "episode", "id", "seed", "sample", "step", "after_shot_differs")}
                        | {"target": target_of(r["truth"])})
    rep = {"screenshots": len(rows), "episodes": len(order), "mismatches": problems["mismatch"], "first_mismatches": first_bad,
           "fields_left_out": problems["dropped_fields"], "boxes_clipped": problems["boxes_clipped"],
           "elements_off_screen_left_out": problems["elements_off_screen"], "after_shot_differs": sum(r["after_shot_differs"] for r in rows),
           "target_lines": sum(r["target"].count("\n") + 1 for r in rows)}
    if problems["mismatch"] or (a.expect and len(rows) != a.expect):
        print(json.dumps(rep, ensure_ascii=False, indent=1))
        raise SystemExit("a label does not read back as its true list, or the count is not the expected one")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train.jsonl", rows), (f"train_first{a.first}.jsonl", rows[:a.first])):
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part)
        (out_dir / name).write_text(text, encoding="utf-8")
        rep[name] = {"rows": len(part), "md5": hashlib.md5(text.encode()).hexdigest()[:12],
                     "episodes": len({r["episode"] for r in part}), "templates": len({r["id"] for r in part})}
    (out_dir / "align_data.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
