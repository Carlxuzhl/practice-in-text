"""AndroidControl episodes, read without tensorflow or generated protobuf code.

- records(): the GZIP-compressed TFRecord framing (length, length CRC, data, data CRC); the CRCs are not checked, the
  shard's md5 is (MD5, the public bucket's md5Hash).
- episode(): one tf.train.Example (Features.feature map; BytesList / FloatList / Int64List) as a dict.
- forest(): one android_env AndroidAccessibilityForest, with the field numbers of android_env/proto/a11y at commit
  dacb11c418e010b56a2ebcf46c7adef5e26a5010 (2024-05-08, the last change to those files before the dataset's release).

forest() decodes only what the text view may use plus the window structure, so the extractor cannot read what is not
on the screen: content_description, view_id_resource_name, tooltip_text, the interaction flags (is_clickable,
is_focusable, is_editable, ...), labels between nodes and the action lists are skipped by field number. The app of a
node (package_name) is kept under a separate key, only to group screens by app, as the template id groups them in
MobileGym; it never enters a text view.

The data stays where androidcontrol/download.sh puts it: $ANDROIDCONTROL_DIR/raw (default WORK_DIR/androidcontrol/raw).
It is not redistributed with this code."""
from __future__ import annotations

import gzip
import hashlib
import os
import struct
from pathlib import Path

from common.config import work_dir


def data_dir() -> Path:
    return Path(os.environ.get("ANDROIDCONTROL_DIR") or work_dir() / "androidcontrol")


def raw_dir() -> Path:
    return data_dir() / "raw"


# md5 of each file, as the public bucket gives it (md5Hash); the study used shards 0-3 of the 20
MD5 = {"splits.json": "d179f4dceb50c70dc31de1376bdd2b92", "test_subsplits.json": "15afd73ebe7149d1dff1c1a24b5ce9dd",
       "android_control-00000-of-00020": "42e84ff97e5a88ffa20f96a694d33ab4",
       "android_control-00001-of-00020": "604814a8b2a691bdcccaefac9fcc3d1f",
       "android_control-00002-of-00020": "991a213084d73203afe655bdb8a649ce",
       "android_control-00003-of-00020": "d621ede670fc9890de6cd5df6e3f89f4"}
SHARD_NAMES = [f"android_control-0000{i}-of-00020" for i in range(4)]


def shards() -> list[Path]:
    return [raw_dir() / name for name in SHARD_NAMES]


def check_md5(names=None) -> dict[str, bool]:
    """Whether each downloaded file has the bucket's md5."""
    out = {}
    for name in names or MD5:
        p = raw_dir() / name
        out[name] = p.exists() and hashlib.md5(p.read_bytes()).hexdigest() == MD5[name]
    return out


WINDOW_TYPES = {0: "unknown", 1: "application", 2: "input_method", 3: "system", 4: "accessibility_overlay",
                5: "split_screen_divider"}
# AndroidAccessibilityNodeInfo fields the text view may use (number: name)
NODE_TEXT = {3: "class_name", 5: "hint_text", 7: "text"}
NODE_FLAG = {13: "is_checked", 20: "is_password", 23: "is_visible_to_user"}
NODE_INT = {1: "unique_id", 11: "window_id", 27: "depth", 30: "drawing_order"}
NODE_BOX, NODE_CHILDREN, NODE_APP = 2, 25, 6


def _varint(buf, i: int) -> tuple[int, int]:
    shift = value = 0
    while True:
        b = buf[i]
        i += 1
        value |= (b & 0x7F) << shift
        if b < 0x80:
            return value, i
        shift += 7


def _signed(v: int) -> int:
    return v - (1 << 64) if v >= 1 << 63 else v


def fields(buf) -> dict[int, list]:
    """Wire-format decoding: {field number: [values]}; varints as int, length-delimited values as memoryview."""
    buf = memoryview(buf)
    out: dict[int, list] = {}
    i, n = 0, len(buf)
    while i < n:
        key, i = _varint(buf, i)
        num, wire = key >> 3, key & 7
        if wire == 0:
            v, i = _varint(buf, i)
        elif wire == 2:
            ln, i = _varint(buf, i)
            v = buf[i:i + ln]
            i += ln
        elif wire == 5:
            v = buf[i:i + 4]
            i += 4
        elif wire == 1:
            v = buf[i:i + 8]
            i += 8
        else:
            raise ValueError(f"unsupported wire type {wire}")
        out.setdefault(num, []).append(v)
    return out


def _packed(values: list) -> list[int]:
    out = []
    for v in values:
        if isinstance(v, int):
            out.append(_signed(v))
            continue
        i = 0
        while i < len(v):
            x, i = _varint(v, i)
            out.append(_signed(x))
    return out


def records(path: Path):
    with gzip.open(path, "rb") as f:
        while True:
            head = f.read(12)
            if not head:
                return
            if len(head) < 12:
                raise ValueError(f"{path}: truncated record header")
            (length,) = struct.unpack("<Q", head[:8])
            data = f.read(length)
            if len(data) < length or len(f.read(4)) < 4:
                raise ValueError(f"{path}: truncated record")
            yield data


def episode(data: bytes) -> dict:
    """{feature name: list}: bytes_list values as memoryview, int64_list as int, float_list as float."""
    feats = fields(fields(data)[1][0])
    out = {}
    for entry in feats.get(1, []):
        e = fields(entry)
        key = bytes(e[1][0]).decode("utf-8")
        f = fields(e[2][0]) if 2 in e else {}
        if 1 in f:
            out[key] = fields(f[1][0]).get(1, [])
        elif 3 in f:
            out[key] = _packed(fields(f[3][0]).get(1, []))
        elif 2 in f:
            raw = fields(f[2][0]).get(1, [])
            out[key] = [x for v in raw for x in struct.unpack(f"<{len(v) // 4}f", v)]
        else:
            out[key] = []
    return out


def _rect(v) -> list[int]:
    f = fields(v)
    return [_signed(f.get(k, [0])[0]) for k in (1, 2, 3, 4)]  # left, top, right, bottom


def node(v) -> dict:
    f = fields(v)
    n = {"box": _rect(f[NODE_BOX][0]) if NODE_BOX in f else None}
    for num, name in NODE_TEXT.items():
        n[name] = bytes(f[num][0]).decode("utf-8", "replace") if num in f else ""
    for num, name in NODE_FLAG.items():
        n[name] = bool(f[num][0]) if num in f else False
    for num, name in NODE_INT.items():
        n[name] = _signed(f[num][0]) if num in f else 0
    n["children"] = _packed(f.get(NODE_CHILDREN, []))
    n["_app"] = bytes(f[NODE_APP][0]).decode("utf-8", "replace") if NODE_APP in f else ""
    return n


def forest(data) -> list[dict]:
    """Windows in the order stored, each {type, layer, id, box, active, focused, nodes}."""
    out = []
    for w in fields(data).get(1, []):
        f = fields(w)
        tree = fields(f[11][0]).get(1, []) if 11 in f else []
        out.append({"type": WINDOW_TYPES.get(f.get(6, [0])[0], "unknown"), "layer": _signed(f.get(4, [0])[0]),
                    "id": _signed(f.get(3, [0])[0]), "box": _rect(f[1][0]) if 1 in f else None,
                    "active": bool(f.get(8, [0])[0]), "focused": bool(f.get(9, [0])[0]),
                    "nodes": [node(v) for v in tree]})
    return out


def app_of(windows: list[dict]) -> str:
    """The app on screen: the package of the topmost application window's nodes (grouping only)."""
    apps = [w for w in windows if w["type"] == "application" and w["nodes"]]
    if not apps:
        return ""
    top = max(apps, key=lambda w: w["layer"])
    return next((n["_app"] for n in top["nodes"] if n["_app"]), "")
