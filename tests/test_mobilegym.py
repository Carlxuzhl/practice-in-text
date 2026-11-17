"""Checks that need the MobileGym checkout (MOBILEGYM_DIR) but no running simulator and no GPU: every practiced template
is in MobileGym's train split and has a program oracle, and decisions map onto the shared action space."""
import os
from pathlib import Path

import pytest

if not os.environ.get("MOBILEGYM_DIR") or not (Path(os.environ["MOBILEGYM_DIR"]) / "bench_env").is_dir():
    pytest.skip("MOBILEGYM_DIR is not set to a MobileGym checkout", allow_module_level=True)

from common import sim  # noqa: E402
from common.config import load  # noqa: E402


def test_templates_are_train_and_have_oracles():
    from policy.oracles import ORACLES
    templates = load("experiment")["templates"]
    assert sorted(ORACLES) == sorted(templates)
    for t in templates:
        sim.assert_train(t)


def test_to_action():
    from policy import episodes as EP
    from bench_env.env.base import ActionType
    obs = {"elements": [{"id": 1, "role": "text", "text": "a", "bbox": [100, 200, 300, 260]},
                        {"id": 2, "role": "field", "text": "", "bbox": [0, 300, 1000, 360]}]}
    a, why = EP.to_action({"action": "tap", "id": 1}, obs)
    assert a.action_type == ActionType.CLICK and a.data["point"] == [200, 230] and not why
    a, _ = EP.to_action({"action": "type", "id": "2", "text": "hi", "clear": True}, obs)
    assert a.action_type == ActionType.TYPE and a.data == {"value": "hi", "point": [500, 330], "clear": True}
    assert EP.to_action({"action": "tap", "id": 9}, obs) == (None, "no such element")
    assert EP.to_action(None, obs) == (None, "unparsable")
    a, _ = EP.to_action({"action": "swipe", "from": [500, 1200], "to": [500, -5]}, obs)
    assert a.data["point1"] == [500, 1000] and a.data["point2"] == [500, 0]
    assert EP.describe({"action": "tap", "id": 1}, obs, "") == "tap a"
    assert EP.describe({"action": "tap", "id": 9}, obs, "no such element").endswith("\uff08no such element\uff09")
