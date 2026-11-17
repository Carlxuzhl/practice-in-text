"""The AndroidControl text view on a made-up accessibility forest (no data, no screenshot): what a node can say but
not draw (description, resource name, tooltip, interaction flags) stays out; the keyboard, system bars and web content
are not read; roles come from the class names. Plus the wire-format decoder and the sampling helpers."""
import json

import pytest

pytest.importorskip("numpy")
from androidcontrol import data as L  # noqa: E402
from androidcontrol import extract as X  # noqa: E402
from androidcontrol import pool as P  # noqa: E402


def test_extractor_self_test():
    X.test()


def test_varint_round_trip():
    for v in (0, 1, 127, 128, 300, 2 ** 31, 2 ** 40):
        got, i = L._varint(X._enc_varint(v), 0)
        assert got == v
    assert L._signed(2 ** 64 - 5) == -5


def test_development_screens_listed():
    dev = P.development()
    assert len(dev) == 118 and all(isinstance(e, int) and isinstance(k, int) for e, k in dev)


def test_icon_font_and_class_names():
    assert X.icon_font("\ue001\ue002") and not X.icon_font("abc")
    assert X.short("android.widget.TextView") == "TextView"
    assert X.SWITCH.search("SwitchCompat") and X.FIELD.search("AutoCompleteTextView") and X.CONTAINER.search("LinearLayout")


def test_md5_table_covers_the_shards():
    assert set(L.SHARD_NAMES) <= set(L.MD5) and {"splits.json", "test_subsplits.json"} <= set(L.MD5)
    json.dumps(L.MD5)
