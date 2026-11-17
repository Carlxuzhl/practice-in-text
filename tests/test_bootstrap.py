"""The paired bootstrap: its self-test, and a report on small made-up runs."""
import json
import random

from evaluation import bootstrap as B
from tests.helpers import write_run


def test_self_test():
    B.self_test()


def test_report_on_made_up_runs(tmp_path):
    templates, seeds = [f"app.T{i}" for i in range(8)], [3000, 3001, 3002]
    inst = tmp_path / "instances.json"
    inst.write_text(json.dumps({"templates": templates, "seeds": seeds}))
    rng = random.Random(1)

    def run(name: str, p: float) -> str:
        rows = [{"id": t, "seed": s, "official_success": rng.random() < p} for t in templates for s in seeds]
        return str(write_run(tmp_path, name, rows))

    spec = {"instances": str(inst),
            "arms": {"good": {"group": "g", "seeds": {"0": [run("good0", 0.9)], "1": [run("good1", 0.9)]}},
                     "bad": {"group": "b", "seeds": {"0": [run("bad0", 0.2)], "1": [run("bad1", 0.2)]}}},
            "comparisons": {"main": ["good", "bad"]}}
    rep = B.report(spec, boot=2000)
    m = rep["comparisons"]["main"]
    assert m["difference"] > 30 and m["95_main"][0] > 0 and m["level"] == "clearly" and m["direction"] == 1
    assert m["wording"] == "good clearly higher than bad"
    assert rep["seed_groups"] == ["b", "g"]
