"""Reading run directories. A run (policy/episodes.py, reader/reread.py) holds
  meta.json               provenance and settings
  rows.jsonl              one row per episode: id (template), seed, sample, official_success, steps, stop, ...
  steps/<stem>.jsonl      one record per step; stem is <template>_<seed>, or <template>_<seed>_k<sample> with --samples
  shots/                  screenshots (PNG with --png, else JPEG)
  vision_ledger.jsonl     one line per reader call
In a step record, "v0" is the true text view of the page (the list of visible elements) and "reader_raw" what the
reader wrote for the screenshot of that step."""
from __future__ import annotations

import json
from pathlib import Path


def rows(run: Path) -> list[dict]:
    return [json.loads(line) for line in (Path(run) / "rows.jsonl").open(encoding="utf-8")]


def parse_stem(stem: str) -> tuple[str, int, int]:
    """(template, seed, sample) of a step file's stem."""
    parts = stem.rsplit("_k", 1)
    tid_seed, k = (parts[0], int(parts[1])) if len(parts) == 2 and parts[1].isdigit() else (stem, 0)
    tid, seed = tid_seed.rsplit("_", 1)
    return tid, int(seed), k


def episode_key(row: dict) -> tuple[str, int, int]:
    return row["id"], int(row["seed"]), int(row.get("sample") or 0)


def screens_of(runs: list[Path], only: set | None = None) -> list[dict]:
    """Every read step of the runs: the true list the text environment shows for it, the reader's list, and whether the
    reader's output was cut by its length limit. only: the (template, seed, sample) of the episodes to use."""
    from reader.reader import parse_reader

    out = []
    for run in map(Path, runs):
        for f in sorted((run / "steps").glob("*.jsonl")):
            for rec in map(json.loads, f.open(encoding="utf-8")):
                if "reader_raw" not in rec:
                    continue
                tid, seed, k = parse_stem(f.stem)
                if only is not None and (tid, seed, k) not in only:
                    continue
                out.append({"episode": f"{run.name}/{f.stem}", "id": tid, "seed": seed, "sample": k, "step": rec["step"],
                            "truth": rec["v0"], "read": parse_reader(rec["reader_raw"])[0]["elements"],
                            "cut": rec.get("reader_finish") == "length"})
    return out


def episodes_from(path: str | Path, key: str) -> set[tuple[str, int, int]]:
    """The (template, seed, sample) listed under key in a JSON file such as calibration_set.json."""
    listed = json.loads(Path(path).read_text(encoding="utf-8"))[key]
    return {(e["id"], int(e["seed"]), int(e.get("sample", 0))) for e in listed}
