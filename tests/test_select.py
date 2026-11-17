"""Picking whole episodes by screenshot budget, and run views."""
import collections
import json

from policy import select as SEL
from tests.helpers import write_run


def test_take_reads_whole_episodes():
    reads = collections.Counter({"a": 3, "b": 4, "c": 5})
    assert SEL.take_reads(["a", "b", "c"], reads, 7) == ["a", "b"]
    assert SEL.take_reads(["a", "b", "c"], reads, 8) == ["a", "b", "c"]
    assert SEL.take_reads(["a", "b", "c"], reads, 1) == ["a"]


def test_calibration_set_and_view(tmp_path):
    templates, seeds = ["app.A", "app.B"], [1008, 1009]
    rows, ledger, steps = [], [], {}
    for s in seeds:
        for t in templates:
            rows.append({"id": t, "seed": s, "sample": 0, "official_success": t == "app.A"})
            ledger += [{"id": t, "seed": s, "sample": 0}] * 3
            steps[f"{t}_{s}"] = [{"step": 1}]
    run = write_run(tmp_path, "cal", rows, steps, ledger)
    doc = SEL.calibration_set([run], templates, seeds, budget=7, half=4)
    assert [(e["id"], e["seed"]) for e in doc["calibration"]] == [("app.A", 1008), ("app.B", 1008), ("app.A", 1009)]
    assert [(e["id"], e["seed"]) for e in doc["first_part"]] == [("app.A", 1008), ("app.B", 1008)]
    assert [(e["id"], e["seed"]) for e in doc["unused"]] == [("app.B", 1009)]
    assert doc["reads"]["calibration"] == 9
    view = SEL.view(tmp_path / "view", [run], {("app.A", 1008, 0), ("app.B", 1008, 0)})
    kept = [json.loads(line) for line in (view / "rows.jsonl").open()]
    assert len(kept) == 2 and (view / "steps" / "app.A_1008.jsonl").is_symlink()
