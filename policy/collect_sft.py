"""Training examples of one practice round. An episode is kept when the judge accepted it and the policy itself said it
was done (official success). Every step of a kept episode becomes one example, exactly as the policy saw it (the run's
system prompt and the user message of that step) and as it answered (its thinking and its action). A step is left out
when its answer could not be parsed into an action, the action was invalid, or the answer ran into the token limit.
Usage: python -m policy.collect_sft <out.jsonl> <run_dir>...   (prints the counts as one line of JSON)"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

INVALID = "无效动作"  # "invalid action": the history entry policy/episodes.py writes for an action it could not carry out


def collect(out_path: Path, runs: list[Path]) -> dict:
    stats = collections.Counter()
    with Path(out_path).open("w", encoding="utf-8") as out:
        for run in map(Path, runs):
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))["config"]
            system, multi = meta["policy_system"], meta.get("policy", {}).get("samples", 1) > 1
            for row in map(json.loads, (run / "rows.jsonl").open(encoding="utf-8")):
                stats["episodes"] += 1
                if not row.get("official_success"):
                    continue
                stats["kept episodes"] += 1
                name = f'{row["id"]}_{row["seed"]}' + (f'_k{row.get("sample", 0)}' if multi else "")
                for rec in map(json.loads, (run / "steps" / f"{name}.jsonl").open(encoding="utf-8")):
                    if not rec.get("policy_user") or rec.get("action") is None or rec.get("summary", "").startswith(INVALID) \
                            or rec.get("policy_finish") == "length":
                        stats["steps left out"] += 1
                        continue
                    out.write(json.dumps({"system": system, "user": rec["policy_user"], "assistant": rec["policy_raw"], "run": run.name,
                                          "id": row["id"], "seed": row["seed"], "sample": row.get("sample", 0), "step": rec["step"]},
                                         ensure_ascii=False) + "\n")
                    stats["examples"] += 1
    return dict(stats)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    print(json.dumps(collect(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]])))


if __name__ == "__main__":
    main()
