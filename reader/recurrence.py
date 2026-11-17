"""Do a reader's misreads recur on the same elements across episodes? (CPU only; it reads saved reader output.)

Data: the reads of runs whose steps hold both the true text view and the reader's output: the calibration-set
collection (the original reader's own reads) or a reader/reread.py run of the same screenshots with another reader or
temperature. Episodes are split into "fit" episodes (the calibration set: calibration_set.json, key calibration) and "new" episodes
(collected but unused, key unused).

Elements: every true element below the status bar, on screens whose output was not cut and had at most 60 extras. An
element's identity across episodes: app, role, text, icon name, switch or mark state, box to 10 units
(error_rate.ident). Per element and episode, the first read counts. Error classes as in error_rate.outcomes.

Control: independent errors at the same rates. The rate-matched noise model (practice/noise.py) is applied to the same
true lists, with draws keyed by the episode and the screen's content, so that its errors have the reader's rates but
fall on elements independently. Give the fitted model with --control (the paper's 4B reader at temperature 0 used the
model fitted for rate-matched practice), or let it be fitted here on the fit episodes' reads.

A. Recurrence on the fit episodes: over the elements read in two or more episodes, q = the chance that another
   episode's read is wrong given that this one is wrong, r = the same given that this one is right (each element weighs
   the same), for the reader and for the control. The difference of q (reader minus control, in points) is the
   reported quantity, with a 95% interval from resampling apps 1,000 times. "same_way": among pairs of wrong reads of
   an element, the share written identically.
B. Predictable on new episodes: the error rate of new reads by the element's record on the fit episodes (always wrong,
   never wrong, mixed, not seen), the share of new errors on elements seen (or seen wrong) before, and the AUC of
   predicting each new read's error from its features alone (the fit-episode rate of its feature group) and from
   features plus the element's own record.

Usage: python -m reader.recurrence --runs RUN... --episodes calibration_set.json [--fit-key calibration]
                                   [--new-key unused]
                                   [--control noise_model.json] [--out report.json]"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing as mp
import random
import statistics
from pathlib import Path

from common.config import load
from common.records import episodes_from, screens_of
from practice import noise as N
from reader import match as M
from reader.error_rate import STATUS_Y2, ident, outcomes
from textview.textview import render

CFG = load("reader")["recurrence"]
MAX_EXTRAS, BOOT, WORKERS, THRESHOLD = CFG["max_extras"], CFG["bootstrap"], CFG["workers"], CFG["threshold_points"]
CLASSES = ("any", "1_miss", "2_shape", "3_text", "5_state", "4_renamed")
_MODEL: N.Calibrated | None = None


def _init(model: dict) -> None:
    global _MODEL
    _MODEL = N.Calibrated(model)


def screen_job(s: dict) -> dict:
    """Outcomes of one screen for the reader and for the control (independent draws keyed by episode and content)."""
    app, truth = M.app_of(s["id"]), s["truth"]
    reader, extras = outcomes(truth, s["read"])
    content = hashlib.sha256(render({"elements": truth}).encode()).hexdigest()[:16]
    sim = _MODEL.apply({"elements": truth}, f'{s["id"]}|{s["seed"]}|{s["sample"]}|{content}')["elements"]
    control, _ = outcomes(truth, sim)
    n = len(truth)
    els = [{"ident": ident(app, t), "feat": N.features(t, n), "reader": a, "control": b}
           for t, a, b in zip(truth, reader, control) if t["bbox"][3] > STATUS_Y2]
    return {"episode": (s["id"], s["seed"], s["sample"]), "step": s["step"], "cut": s["cut"], "extras": extras,
            "content": content, "els": els}


# ---------------------------------------------------------------- reads per element

def occurrences(screens: list[dict], who: str, episodes: set, distinct_screens: bool = False) -> dict:
    """{ident: [(flags, sig, feat), ...]}, the first read of the element in each episode, in episode order; with
    distinct_screens, a read on a true screen already counted for the element is skipped."""
    occ: dict = collections.defaultdict(dict)
    seen: dict = collections.defaultdict(set)
    for s in screens:  # sorted by episode, then step
        if s["episode"] not in episodes:
            continue
        for e in s["els"]:
            d = occ[e["ident"]]
            if s["episode"] in d or (distinct_screens and s["content"] in seen[e["ident"]]):
                continue
            d[s["episode"]] = (*e[who], e["feat"])
            seen[e["ident"]].add(s["content"])
    return {k: list(v.values()) for k, v in occ.items()}


def per_element(occ: dict, cls: str) -> list[tuple[int, int, collections.Counter]]:
    """(reads, wrong reads, how the wrong reads were written) per element read twice or more."""
    out = []
    for vals in occ.values():
        vals = [(f[cls], sig) for f, sig, _ in vals if cls in f]
        if len(vals) >= 2:
            k = sum(v for v, _ in vals)
            out.append((len(vals), k, collections.Counter(sig for v, sig in vals if v)))
    return out


def persistence(rows: list[tuple[int, int, collections.Counter]]) -> dict:
    mean = lambda xs: round(statistics.fmean(xs), 4) if xs else None
    q = [(k - 1) / (n - 1) for n, k, _ in rows if k >= 1]
    r = [k / (n - 1) for n, k, _ in rows if n - k >= 1]
    same = [sum(x * (x - 1) for x in c.values()) / (k * (k - 1)) for n, k, c in rows if k >= 2]
    wrong3 = [(n, k) for n, k, _ in rows if n >= 3 and k >= 1]
    return {"elements": len(rows), "reads": sum(n for n, _, _ in rows), "p": mean([k / n for n, k, _ in rows]),
            "q": mean(q), "r": mean(r), "same_way": mean(same),
            "elements_wrong_ever_3plus": len(wrong3),
            "share_always_wrong": mean([k == n for n, k in wrong3]), "share_wrong_once": mean([k == 1 for n, k in wrong3])}


def boot_q_diff(occ_reader: dict, occ_control: dict, cls: str) -> dict:
    """95% interval of q(reader) - q(control) in points, resampling apps (the elements of one app move together)."""
    per_app = collections.defaultdict(lambda: [0.0, 0, 0.0, 0])  # reader sum, reader elements, control sum, control elements
    for key in occ_reader:
        for idx, occ in ((0, occ_reader), (2, occ_control)):
            vals = [f[cls] for f, _, _ in occ.get(key, []) if cls in f]
            n, k = len(vals), sum(vals)
            if n >= 2 and k >= 1:
                per_app[key[0]][idx] += (k - 1) / (n - 1)
                per_app[key[0]][idx + 1] += 1
    apps, rng, diffs = sorted(per_app), random.Random(0), []
    for _ in range(BOOT):
        pick = [per_app[rng.choice(apps)] for _ in apps]
        a, b = sum(p[1] for p in pick), sum(p[3] for p in pick)
        if a and b:
            diffs.append(sum(p[0] for p in pick) / a - sum(p[2] for p in pick) / b)
    diffs.sort()
    if not diffs:  # no element was misread twice in any resample
        return {"apps": len(apps), "interval": None}
    return {"apps": len(apps), "interval": [round(100 * diffs[int(0.025 * len(diffs))], 1), round(100 * diffs[int(0.975 * len(diffs)) - 1], 1)]}


# ---------------------------------------------------------------- B: new episodes

def auc(scores: list[float], labels: list[bool]) -> float | None:
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    npos = sum(labels)
    nneg = len(labels) - npos
    if not npos or not nneg:
        return None
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        rank_sum += sum(1 for k in range(i, j + 1) if pairs[k][1]) * ((i + j) / 2 + 1)
        i = j + 1
    return (rank_sum - npos * (npos + 1) / 2) / (npos * nneg)


def feature_rates(train: dict, cls: str):
    tabs = [collections.defaultdict(lambda: [0, 0]) for _ in N.BACKOFF]
    for vals in train.values():
        for f, _, feat in vals:
            if cls in f:
                for li, lv in enumerate(N.BACKOFF):
                    a = tabs[li][tuple(feat[i] for i in lv)]
                    a[0] += f[cls]
                    a[1] += 1
    total = [sum(a[0] for a in tabs[-1].values()), sum(a[1] for a in tabs[-1].values())]

    def rate(feat) -> float:
        for li, lv in enumerate(N.BACKOFF):
            a = tabs[li].get(tuple(feat[i] for i in lv))
            if a and a[1] >= N.MIN_COUNT:
                return a[0] / a[1]
        a = tabs[-1].get((feat[0],))
        return a[0] / a[1] if a and a[1] else total[0] / max(1, total[1])
    return rate


def new_instances(train: dict, test: dict, cls: str) -> dict:
    rate = feature_rates(train, cls)
    record = {k: (sum(f[cls] for f, _, _ in v if cls in f), sum(cls in f for f, _, _ in v)) for k, v in train.items()}
    buckets = collections.defaultdict(lambda: [0, 0])
    feat_s, mem_s, labels, keys = [], [], [], []
    for key, vals in test.items():
        k_tr, n_tr = record.get(key, (0, 0))
        for f, _, feat in vals:
            if cls not in f:
                continue
            y = f[cls]
            b = "not seen" if n_tr == 0 else "always wrong" if k_tr == n_tr else "never wrong" if k_tr == 0 else "mixed"
            buckets[b][0] += y
            buckets[b][1] += 1
            fr = rate(feat)
            feat_s.append(fr)
            mem_s.append((k_tr + fr) / (n_tr + 1))
            labels.append(y)
            keys.append(key)
    wrong = sum(labels)
    on_seen = sum(y for y, k in zip(labels, keys) if record.get(k, (0, 0))[1] > 0)
    on_seen_wrong = sum(y for y, k in zip(labels, keys) if record.get(k, (0, 0))[0] > 0)
    a_f, a_m = auc(feat_s, labels), auc(mem_s, labels)
    by_app = collections.defaultdict(list)
    for i, k in enumerate(keys):
        by_app[k[0]].append(i)
    apps, rng, diffs = sorted(by_app), random.Random(0), []
    for _ in range(200):
        idx = [i for a in (rng.choice(apps) for _ in apps) for i in by_app[a]]
        x, y = auc([feat_s[i] for i in idx], [labels[i] for i in idx]), auc([mem_s[i] for i in idx], [labels[i] for i in idx])
        if x is not None and y is not None:
            diffs.append(y - x)
    diffs.sort()
    return {"reads": len(labels), "wrong": wrong, "error_rate": round(wrong / max(1, len(labels)), 4),
            "by_fit_record": {b: {"reads": v[1], "error_rate": round(v[0] / v[1], 4) if v[1] else None, "wrong": v[0]}
                              for b, v in sorted(buckets.items())},
            "share_of_errors_on_elements_seen": round(on_seen / max(1, wrong), 4),
            "share_of_errors_on_elements_seen_wrong": round(on_seen_wrong / max(1, wrong), 4),
            "auc_features": None if a_f is None else round(a_f, 4), "auc_features_and_record": None if a_m is None else round(a_m, 4),
            "auc_gain": None if a_f is None or a_m is None else round(a_m - a_f, 4),
            "auc_gain_95_by_app_200": [round(diffs[int(0.025 * len(diffs))], 4), round(diffs[int(0.975 * len(diffs)) - 1], 4)] if diffs else None}


# ---------------------------------------------------------------- main

def analyse(screens: list[dict], fit_eps: set, new_eps: set, control: dict) -> dict:
    with mp.Pool(WORKERS, initializer=_init, initargs=(control,)) as pool:
        done = pool.map(screen_job, screens, chunksize=8)
    kept = [s for s in done if not s["cut"] and s["extras"] <= MAX_EXTRAS]
    report: dict = {"reads": len(done), "screens_used": len(kept),
                    "screens_left_out": {"cut": sum(s["cut"] for s in done),
                                         "over_60_extras": sum((not s["cut"]) and s["extras"] > MAX_EXTRAS for s in done)},
                    "control_digest": N.Calibrated(control).digest}
    A = {}
    for distinct in (False, True):
        occ = {who: occurrences(kept, who, fit_eps, distinct) for who in ("reader", "control")}
        part = {}
        for cls in CLASSES:
            part[cls] = {who: persistence(per_element(occ[who], cls)) for who in ("reader", "control")}
            qa, qb = part[cls]["reader"]["q"], part[cls]["control"]["q"]
            part[cls]["q_difference_points"] = None if qa is None or qb is None else round(100 * (qa - qb), 1)
        for cls in ("any", "1_miss"):
            part[cls]["q_difference_95_by_app"] = boot_q_diff(occ["reader"], occ["control"], cls)
        A["distinct_screens" if distinct else "all_pairs"] = part
    report["A_recurrence_fit_episodes"] = A
    report["B_new_episodes"] = {who: {cls: new_instances(occurrences(kept, who, fit_eps), occurrences(kept, who, new_eps), cls)
                                      for cls in ("any", "1_miss")} for who in ("reader", "control")}
    diff = A["all_pairs"]["any"]["q_difference_points"] or 0
    report["line"] = {"q_difference_any_at_least": THRESHOLD, "passed": diff >= THRESHOLD}
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--episodes", required=True, help="JSON file listing the episodes, e.g. calibration_set.json")
    ap.add_argument("--fit-key", default="calibration")
    ap.add_argument("--new-key", default="unused")
    ap.add_argument("--control", default="", help="a fitted rate-matched noise model; fitted on the fit episodes if left out")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    fit_eps, new_eps = episodes_from(a.episodes, a.fit_key), episodes_from(a.episodes, a.new_key)
    screens = screens_of([Path(r) for r in a.runs])
    screens.sort(key=lambda s: ((s["id"], s["seed"], s["sample"]), s["step"]))
    if a.control:
        control = json.loads(Path(a.control).read_text(encoding="utf-8"))
    else:
        control = N.fit([s for s in screens if (s["id"], s["seed"], s["sample"]) in fit_eps])
    report = analyse(screens, fit_eps, new_eps, control)
    report["runs"] = [Path(r).name for r in a.runs]
    text = json.dumps(report, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    a_any = report["A_recurrence_fit_episodes"]["all_pairs"]["any"]
    print(json.dumps({"reads": report["reads"], "screens_used": report["screens_used"], "q_reader": a_any["reader"]["q"],
                      "q_control": a_any["control"]["q"], "difference_points": a_any["q_difference_points"],
                      "interval": a_any["q_difference_95_by_app"]["interval"], "line": report["line"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
