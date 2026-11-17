"""Rendering the text view for the policy, icon turns and the layout order."""
from policy import model as pm
from textview import textview as tv
from tests.helpers import screen_a


def test_render_lines():
    lines = tv.render({"elements": screen_a()}).splitlines()
    assert lines[1] == "[2] icon chevron-left black (20,60,70,100)"
    assert lines[2] == '[3] text "Settings" black bold (400,60,600,100)'
    assert lines[4] == "[5] switch off lightgray (820,150,940,190)"
    assert lines[7] == '[8] field "Search" empty (40,300,960,350)'
    assert lines[8] == "[9] mark filled blue (40,400,80,440)"
    assert lines[10] == '[11] text "Submit" white on-blue (300,900,700,960)'


def test_turned():
    assert tv.turned("chevron-right", 180) == "chevron-left"
    assert tv.turned("chevron-right", 90) == "chevron-down"
    assert tv.turned("arrow-up-down", 90) == "arrow-up-down"
    assert tv.turned("plus", 90) == "plus"


def test_reading_order():
    els = [{"id": 1, "role": "text", "bbox": [500, 100, 600, 120]}, {"id": 2, "role": "text", "bbox": [10, 101, 90, 121]},
           {"id": 3, "role": "text", "bbox": [10, 10, 90, 30]}]
    out = tv.reading_order(els)  # rows of 6 px of an 800 px screen: y 100 and 101 share a row, 102 would not
    assert [e["bbox"][:2] for e in out] == [[10, 10], [10, 101], [500, 100]] and [e["id"] for e in out] == [1, 2, 3]


def test_policy_message_and_decision():
    msg = pm.user_message("T", "S", [f"h{i}" for i in range(12)], 3, 30, [f"n{i}" for i in range(12)])
    assert "h3" not in msg and "h4" in msg and "n1\n" not in msg and "n2" in msg
    assert pm.parse_decision('<think>a {"x": 1}</think>\n{"action": "tap", "id": 3}') == {"action": "tap", "id": 3}
    assert pm.parse_decision("no json here") is None
    assert pm.parse_decision('{"id": 3}') is None
