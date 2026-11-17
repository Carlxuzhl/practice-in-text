"""Write the instance files configs/instances/*.json from configs/experiment.json (they are also shipped).
Each is {"templates": [...], "seeds": [...]}: the episode runner runs every template with every seed.
Usage: python -m policy.manifests [--check]   (--check: only verify that the shipped files match)"""
from __future__ import annotations

import argparse
import json

from common.config import CONFIGS, load


def specs() -> dict[str, dict]:
    exp = load("experiment")
    t, s = exp["templates"], exp["seeds"]
    out = {
        "demonstrations": {"templates": t, "seeds": s["demonstrations"]},
        "calibration": {"templates": t, "seeds": s["calibration"]},
        "calibration_top_up_0": {"templates": t, "seeds": [s["calibration_top_up_from"]]},
        "first_round_evaluation": {"templates": t, "seeds": s["first_round_evaluation"]},
        "confirmation": {"templates": t, "seeds": s["confirmation"]},
    }
    for r, seed in s["practice_round"].items():
        out[f"practice_round_{r}"] = {"templates": t, "seeds": [seed]}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    bad = []
    for name, spec in specs().items():
        path = CONFIGS / "instances" / f"{name}.json"
        text = json.dumps(spec, indent=1) + "\n"
        if a.check:
            if not path.exists() or json.loads(path.read_text(encoding="utf-8")) != spec:
                bad.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print(path.relative_to(CONFIGS.parent), len(spec["templates"]), "templates x", len(spec["seeds"]), "seeds")
    if bad:
        raise SystemExit(f"instance files out of date: {bad}")


if __name__ == "__main__":
    main()
