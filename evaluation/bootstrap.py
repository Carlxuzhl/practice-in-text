"""Paired bootstrap over task templates for differences in full-task success.

- Per instance (template, instance seed), a group's success is the mean over its training seeds; an episode that ended
  in an error is a failure. A group's rate is the mean over templates of the per-template mean over instances.
- Main interval: 20,000 resamples of the templates (random.Random(0)); 95%.
- Robust interval: 20,000 resamples (random.Random(1)) of the templates and, per seed group, of its training seeds. The
  same policies seen through another reader (or in text) share one seed group, so that such a comparison isolates the
  reader and not the policies.
- Wording, for a difference d = X - Y in points: if an interval excludes zero, "clearly higher/lower" (|d| >= 5) or
  "slightly higher/lower" (|d| < 5); if it holds zero, "points toward X/Y" (|d| >= 5) or "no measurable difference"
  (|d| < 5). When the main and the robust interval give different wordings, the weaker one is used.
Both intervals use one random stream for all comparisons of a spec: every resample draws the templates, then one seed
list per seed group in the spec's group order. The intervals therefore depend on the spec's groups and their seed
counts, not on which comparisons are asked for.

Spec (JSON):
  {"instances": "configs/instances/confirmation.json",
   "seed_groups": ["clean", "direct", ...],          optional; default: the groups' names, sorted
   "arms": {"aligned": {"group": "clean", "seeds": {"0": ["<run dir>", ...], "1": [...]}}, ...},
   "comparisons": {"main": ["aligned", "direct"], ...}}
Run directories are relative to WORK_DIR/runs unless absolute; together, the runs of one arm and training seed must
hold every instance exactly once.
Usage: python -m evaluation.bootstrap SPEC.json [--out report.json] [--boot 20000]
       python -m evaluation.bootstrap --self-test"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from common.config import REPO, runs_dir
from common.records import rows

BOOT = 20000
LEVEL = {"clearly": 3, "slightly": 2, "points toward": 1, "no measurable difference": 0}


def outcomes(runs: list[Path], want: set) -> dict[tuple, int]:
    rs = [r for run in runs for r in rows(run)]
    keys = [(r["id"], int(r["seed"])) for r in rs]
    if set(keys) != want or len(keys) != len(want):
        raise SystemExit(f"runs {[Path(r).name for r in runs]} do not have the {len(want)} instances once each")
    return {(r["id"], int(r["seed"])): int(bool(r.get("official_success"))) for r in rs}


def rates(per_seed: dict[int, dict], templates: list[str], seeds: list[int]) -> dict[int, dict[str, float]]:
    """training seed -> template -> success over the instance seeds."""
    return {s: {t: sum(o[(t, i)] for i in seeds) / len(seeds) for t in templates} for s, o in per_seed.items()}


def arm_rate(tr: dict[int, dict[str, float]], pick: list[str], use: list[int]) -> float:
    return sum(sum(tr[s][t] for s in use) / len(use) for t in pick) / len(pick)


def wording(d: float, lo: float, hi: float) -> tuple[str, int]:
    """(level, sign): sign +1 means X above Y, -1 Y above X, 0 no direction."""
    if lo > 0:
        return ("clearly" if d >= 5 else "slightly"), 1
    if hi < 0:
        return ("clearly" if -d >= 5 else "slightly"), -1
    return ("no measurable difference", 0) if abs(d) < 5 else ("points toward", 1 if d > 0 else -1)


def weaker(a: tuple[str, int], b: tuple[str, int]) -> tuple[str, int]:
    if a == b:
        return a
    if a[1] and b[1] and a[1] != b[1]:
        return "no measurable difference", 0
    return min(a, b, key=lambda w: LEVEL[w[0]])


def phrase(x: str, y: str, w: tuple[str, int]) -> str:
    level, sign = w
    if level in ("clearly", "slightly"):
        return f"{x} {level} {'higher' if sign > 0 else 'lower'} than {y}"
    if level == "points toward":
        return f"points toward {x if sign > 0 else y}"
    return "no measurable difference"


def compare(tr: dict, groups: dict[str, str], order: list[str], templates: list[str], pairs: dict[str, tuple[str, str]],
            boot: int = BOOT) -> dict:
    """tr: arm -> training seed -> template -> success; groups: arm -> seed group; order: the seed groups' draw order."""
    seeds_of = {a: sorted(tr[a]) for a in tr}
    group_seeds = {}
    for a in sorted(tr):
        g = groups[a]
        if g in group_seeds and group_seeds[g] != seeds_of[a]:
            raise SystemExit(f"the arms of seed group {g} have different training seeds")
        group_seeds.setdefault(g, seeds_of[a])
    if sorted(order) != sorted(group_seeds):
        raise SystemExit(f"seed_groups {order} do not match the arms' groups {sorted(group_seeds)}")
    point = {k: 100 * (arm_rate(tr[x], templates, seeds_of[x]) - arm_rate(tr[y], templates, seeds_of[y])) for k, (x, y) in pairs.items()}
    ci: dict = {}
    for label, rng, robust in (("main", random.Random(0), False), ("robust", random.Random(1), True)):
        draws = {k: [] for k in pairs}
        for _ in range(boot):
            pick = [rng.choice(templates) for _ in templates]
            drawn = {g: [rng.choice(group_seeds[g]) for _ in group_seeds[g]] for g in order}
            use = {a: (drawn[groups[a]] if robust else seeds_of[a]) for a in tr}
            for k, (x, y) in pairs.items():
                draws[k].append(arm_rate(tr[x], pick, use[x]) - arm_rate(tr[y], pick, use[y]))
        for k, xs in draws.items():
            xs.sort()
            ci.setdefault(k, {})[label] = [100 * xs[int(0.025 * boot)], 100 * xs[int(0.975 * boot) - 1]]
    out = {}
    for k, (x, y) in pairs.items():
        wm, wr = wording(point[k], *ci[k]["main"]), wording(point[k], *ci[k]["robust"])
        w = weaker(wm, wr)
        out[k] = {"X": x, "Y": y, "difference": round(point[k], 1), "95_main": [round(v, 1) for v in ci[k]["main"]],
                  "95_robust": [round(v, 1) for v in ci[k]["robust"]], "wording_main": phrase(x, y, wm), "wording_robust": phrase(x, y, wr),
                  "wording": phrase(x, y, w), "level": w[0], "direction": w[1], "intervals_agree": wm == wr}
    return out


def run_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else runs_dir() / path


def report(spec: dict, boot: int = BOOT) -> dict:
    inst_path = Path(spec["instances"])
    inst = json.loads((inst_path if inst_path.is_absolute() else REPO / inst_path).read_text(encoding="utf-8"))
    templates, seeds = sorted(inst["templates"]), inst["seeds"]
    want = {(t, s) for t in templates for s in seeds}
    tr, groups = {}, {}
    for arm, a in spec["arms"].items():
        per_seed = {int(s): outcomes([run_path(r) for r in runs], want) for s, runs in a["seeds"].items()}
        tr[arm] = rates(per_seed, templates, seeds)
        groups[arm] = a.get("group", arm)
    order = spec.get("seed_groups") or sorted(set(groups.values()))
    pairs = {k: tuple(v) for k, v in spec["comparisons"].items()}
    res = compare(tr, groups, order, templates, pairs, boot)
    pct = {a: round(100 * arm_rate(tr[a], templates, sorted(tr[a])), 1) for a in tr}
    per_seed = {f"{a}_s{s}": round(100 * sum(tr[a][s].values()) / len(templates), 1) for a in tr for s in sorted(tr[a])}
    return {"success_percent": pct, "per_seed_percent": per_seed, "comparisons": res, "seed_groups": order, "resamples": boot}


def self_test() -> dict:
    """The machinery on a made-up group: a group against itself gives 0 and no measurable difference; the wording rules."""
    templates = [f"app.T{i}" for i in range(10)]
    rng = random.Random(7)
    tr = {"A": {s: {t: rng.random() for t in templates} for s in (0, 1)}}
    tr["B"] = tr["A"]
    res = compare(tr, {"A": "g", "B": "g2"}, ["g", "g2"], templates, {"A-B": ("B", "A")}, boot=2000)["A-B"]
    assert res["difference"] == 0 and res["level"] == "no measurable difference", res
    w = [wording(6, 1, 11), wording(3, 0.5, 6), wording(2, -1, 5), wording(-7, -12, 1), wording(-8, -13, -2)]
    assert [x[0] for x in w] == ["clearly", "slightly", "no measurable difference", "points toward", "clearly"], w
    assert [x[1] for x in w] == [1, 1, 0, -1, -1], w
    v = [weaker(("clearly", 1), ("slightly", 1)), weaker(("slightly", 1), ("no measurable difference", 0)), weaker(("slightly", 1), ("slightly", -1))]
    assert [x[0] for x in v] == ["slightly", "no measurable difference", "no measurable difference"], v
    return {"self": res, "wording": w, "weaker": v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", nargs="?")
    ap.add_argument("--out", default="")
    ap.add_argument("--boot", type=int, default=BOOT)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        print(json.dumps(self_test(), indent=1))
        return
    if not a.spec:
        raise SystemExit("give a spec, or --self-test")
    rep = report(json.loads(Path(a.spec).read_text(encoding="utf-8")), a.boot)
    text = json.dumps(rep, ensure_ascii=False, indent=1)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
