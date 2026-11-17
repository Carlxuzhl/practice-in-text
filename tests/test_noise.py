"""Practice-environment noise: deterministic for a given key, fitted from (true list, reader list) pairs."""
import random

from practice import noise as N
from tests.helpers import el, screen_a


def corrupt(truth: list[dict], rng: random.Random) -> list[dict]:
    out = []
    for e in truth:
        if rng.random() < 0.2:
            continue
        e = dict(e)
        if e.get("text") and rng.random() < 0.2:
            e["text"] = e["text"][:-1] + "x"
        if e["role"] == "switch" and rng.random() < 0.3:
            e["on"] = not e["on"]
        e["bbox"] = [v + rng.randint(-3, 3) for v in e["bbox"]]
        out.append(e)
    if rng.random() < 0.5:
        out.append(el(0, "text", (10, 500, 200, 540), text="Settings x"))
    return out


def screens(n: int = 60) -> list[dict]:
    rng = random.Random(0)
    out = []
    for i in range(n):
        truth = screen_a()
        out.append({"episode": f"run/app.Task_{i % 12}", "id": "app.Task", "seed": i % 12, "sample": 0, "step": i,
                    "truth": truth, "read": corrupt(truth, rng), "cut": i % 17 == 0})
    return out


def test_generic_is_deterministic():
    g = N.Generic()
    obs = {"elements": screen_a()}
    a, b = g.apply(obs, "app.Task|1|0", 3), g.apply(obs, "app.Task|1|0", 3)
    assert a == b
    assert [e["id"] for e in a["elements"]] == list(range(1, len(a["elements"]) + 1))
    differ = sum(g.apply(obs, f"app.Task|{s}|0", 3) != a for s in range(2, 12))
    assert differ > 0


def test_calibrated_fit_and_apply():
    model = N.fit(screens())
    assert model["version"] == N.VERSION and model["screens"] == 60
    c = N.Calibrated(model)
    obs = {"elements": screen_a()}
    a, b = c.apply(obs, "app.Task|1|0|abc"), c.apply(obs, "app.Task|1|0|abc")
    assert a == b
    assert any(c.apply(obs, f"app.Task|{s}|0|abc") != a for s in range(2, 30))
    for e in a["elements"]:
        assert 0 <= e["bbox"][0] < e["bbox"][2] <= 1000 and 0 <= e["bbox"][1] < e["bbox"][3] <= 1000


def test_fit_check_runs():
    res = N.fit_check(screens(120))
    assert set(res["main_classes"]) == {"1_miss", "2_shape", "5_state_wrong"}
    assert res["train_screens"] + res["test_screens"] == 120
