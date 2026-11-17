"""Matching a reader's list against the true text view, and the element error rate built on it."""
from reader import error_rate as ER
from reader import match as M
from tests.helpers import el, screen_a


def read_of(truth: list[dict]) -> list[dict]:
    return [dict(e) for e in truth]


def test_perfect_read():
    truth = screen_a()
    m = M.match(truth, read_of(truth))
    assert [k for k, _ in m["outcomes"]] == ["kept"] * len(truth)
    assert m["extras"] == []
    r = ER.misreads([{"truth": truth, "read": read_of(truth), "cut": False}])
    assert r["misread"] == 0 and r["elements"] == len(truth) - 1  # the status-bar clock is not counted


def test_miss_misread_state_merge_shape():
    truth = screen_a()
    read = read_of(truth)
    del read[3]  # "Dark mode" left out
    read[3] = {**read[3], "on": True}  # the first switch read in the wrong state
    read = [e for e in read if e.get("text") != "Remember me"]
    read.append(el(99, "text", (60, 395, 420, 445), text="* Remember me"))  # misread text at the same spot
    read = [e for e in read if e.get("text") != "Submit"]
    read.append(el(98, "mark", (300, 900, 700, 960), filled=True, color="blue"))  # the button written as a shape
    m = M.match(truth, read)
    kinds = {t.get("text") or t.get("name") or t["role"]: k for t, (k, _) in zip(truth, m["outcomes"])}
    assert kinds["Dark mode"] == "miss"
    assert kinds["Submit"] == "shape"
    flags = {t["id"]: f for t, (f, _) in zip(truth, ER.outcomes(truth, read)[0])}
    assert flags[4]["1_miss"] and flags[4]["any"]  # Dark mode
    assert flags[10]["3_text"]  # Remember me
    assert flags[5]["5_state"] and not flags[7]["5_state"]  # the two switches
    assert flags[11]["2_shape"]  # Submit


def test_merged_texts():
    truth = [el(1, "text", (10, 100, 100, 130), text="Total"), el(2, "text", (110, 100, 200, 130), text="12.50")]
    read = [el(1, "text", (10, 100, 200, 130), text="Total12.50")]
    m = M.match(truth, read)
    assert [k for k, _ in m["outcomes"]] == ["merged", "merged"]


def test_unparsable_screen_counts_all_missed():
    truth = screen_a()
    r = ER.misreads([{"truth": truth, "read": [], "cut": False}])
    assert r["misread"] == r["elements"] and r["unparsable_screens"] == 1


def test_rates_shape():
    truth = screen_a()
    r = M.rates([(truth, read_of(truth)), (truth, [])])
    assert r["screens"] == 2 and r["1_miss"] == 0.5


def test_ident_rounds_boxes():
    a = el(1, "text", (101, 200, 304, 240), text="Hello ", color="black")
    b = el(1, "text", (99, 204, 296, 236), text="hello", color="gray")
    assert ER.ident("app", a) == ER.ident("app", b)
