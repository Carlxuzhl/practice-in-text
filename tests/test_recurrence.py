"""Recurrence statistics on hand-made reads."""
import collections

from reader import recurrence as RC


def test_persistence_q_and_r():
    # element 1: wrong in 3 of 3 episodes; element 2: wrong in 1 of 2; element 3: never wrong in 2
    rows = [(3, 3, collections.Counter({("miss",): 3})), (2, 1, collections.Counter({("miss",): 1})), (2, 0, collections.Counter())]
    p = RC.persistence(rows)
    assert p["elements"] == 3
    assert p["q"] == round((1.0 + 0.0) / 2, 4)  # (k-1)/(n-1) over elements wrong at least once
    assert p["r"] == round((1.0 + 0.0) / 2, 4)  # k/(n-1) over elements right at least once
    assert p["same_way"] == 1.0


def test_auc():
    assert RC.auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert RC.auc([0.5, 0.5], [False, True]) == 0.5
    assert RC.auc([0.1], [True]) is None


def test_occurrences_first_read_per_episode():
    f_wrong, f_right = {"any": True}, {"any": False}
    screens = [
        {"episode": ("a.T", 1, 0), "content": "c1", "els": [{"ident": "x", "reader": (f_wrong, ("miss",)), "feat": ()}]},
        {"episode": ("a.T", 1, 0), "content": "c2", "els": [{"ident": "x", "reader": (f_right, ("kept",)), "feat": ()}]},
        {"episode": ("a.T", 2, 0), "content": "c1", "els": [{"ident": "x", "reader": (f_right, ("kept",)), "feat": ()}]},
    ]
    occ = RC.occurrences(screens, "reader", {("a.T", 1, 0), ("a.T", 2, 0)})
    assert [f["any"] for f, _, _ in occ["x"]] == [True, False]
    occ = RC.occurrences(screens, "reader", {("a.T", 1, 0), ("a.T", 2, 0)}, distinct_screens=True)
    assert len(occ["x"]) == 1
