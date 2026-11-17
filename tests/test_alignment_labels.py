"""Alignment labels: every screen's label must read back, through the reader's own parser, as its true text view in the
layout order."""
from reader import align_data as AD
from tests.helpers import el, screen_a


def test_label_reads_back():
    truth = screen_a()
    ok, info = AD.check(truth)
    assert ok, info
    lines = AD.target_of(truth).splitlines()
    assert lines[0] == 'text "10:00" black (40,5,120,30)'
    assert 'switch on green (820,220,940,260)' in lines
    assert 'field "Search" empty (40,300,960,350)' in lines
    assert 'text "Submit" white on-blue (300,900,700,960)' in lines


def test_layout_order_and_dropped_fields():
    truth = [el(1, "text", (500, 100, 600, 120), text="right"), el(2, "text", (10, 101, 100, 121), text="left"),
             el(3, "mark", (10, 200, 40, 230), filled=False, round=True), el(4, "icon", (10, 300, 40, 330), raster=True)]
    exp = AD.expected(truth)
    assert [e.get("text") for e in exp[:2]] == ["left", "right"]  # same row: left to right
    assert all("round" not in e and "raster" not in e for e in exp)
    assert AD.check(truth)[0]


def test_clipped_and_offscreen():
    truth = [el(1, "text", (-20, 100, 300, 140), text="cut"), el(2, "icon", (990, 100, 1100, 140), name="plus"),
             el(3, "text", (1005, 200, 1100, 240), text="gone")]
    exp = AD.expected(truth)
    assert [e["bbox"] for e in exp] == [[0, 100, 300, 140], [990, 100, 1000, 140]]
    assert AD.check(truth)[0]


def test_quotes_and_unicode_in_text():
    truth = [el(1, "text", (10, 10, 200, 40), text='say "hi" \\ \u00a5 12.50'), el(2, "field", (10, 60, 300, 90), text="")]
    assert AD.check(truth)[0]
