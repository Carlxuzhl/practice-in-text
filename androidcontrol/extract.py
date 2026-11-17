"""Text view of an AndroidControl screen: only what is on the screen, in the reader's six roles, in the layout order,
with no colour and no icon names (the tree gives neither).

Source: the step's accessibility forest as androidcontrol.data.forest decodes it; content descriptions, resource names,
tooltips, interaction flags, labels between nodes and action lists are never decoded. The tree gives no colour, no icon
name and, for a text that the screen shortens or clips, the whole text; rule 5 checks every element against the
screenshot. Rules (extractor version AC4):
 1. Windows: the status and navigation bars (system windows) and the keyboard (input-method window) are left out; the
    keyboard's keys carry no text in the tree. Every other window is read, and a window of a higher layer hides what
    it covers (rule 2).
    1b. Nothing inside a WebView is read: web pages expose text meant only for screen readers (such as "3.5 out of 5
        stars" over drawn stars), which the tree cannot tell from drawn text (MobileGym's text view reads no ARIA
        text). A WebView still hides what is drawn under it (rule 2d).
 2. A node is a candidate only if is_visible_to_user is set. Its box is clipped to the screen and to its window; a
    strip covered by a higher window that spans the box's full width is cut off; the node is dropped if more than half
    of what is left is covered by higher windows, or if a side of it is under 3 px of a 360-px-wide screen
    (MobileGym's minimum).
    2b. The same holds for bars of its own window drawn over it (_edge_bars: a bottom navigation bar, an app bar);
        only bars that hold something drawn count, not an empty transparent layer.
    2c. A text or field less than 8 such px high shows no line of text and is dropped.
    2d. An element more than half covered by other elements of its window drawn over it is dropped (a banner that a
        list has scrolled over).
    2e. Nothing is kept in the bottom 3% of the screen, where the system draws its gesture bar.
 3. Roles from class_name only (the class is never written out):
    - field: EditText and the auto-complete text views. Its text is what is typed; with nothing typed, its hint text,
      flagged empty (a placeholder); with neither it is left out (MobileGym leaves out boxes that show nothing).
    - switch: any class with "Switch" in its name; on = is_checked. A label on the same node becomes a text element,
      and the switch gets a box at the right end of the node.
    - mark: CheckBox, RadioButton, CompoundButton and their AppCompat / Material versions; filled = is_checked. A label
      on the same node becomes a text element, and the mark gets a square box at the left end of the node.
    - icon or image: ImageView, ImageButton and web images (Image). A node box over 64 px of a 360-px-wide screen on
      either side is an image, else an icon; under 8 such px on a side it is left out (MobileGym's limits for bitmaps).
      Never a name; the text of a web image (its alt text) is not drawn and is dropped.
    - text: any other node with text, except containers (layouts, lists, scroll and web views, dialogs, pagers, tab
      widgets, spinners, toolbars, cards), whose text labels their content, and except nodes with a descendant that
      has text (the visible text is in the descendants).
    3b. A text inside a field's box with the field's own text is that field, not a second element.
    3c. A toolbar action (ActionMenuItemView) without text is an icon; a text of icon-font glyphs only (Unicode private
        use) is an icon.
 4. Box on 0-1000 by the screenshot's width and height; the layout order of textview.reading_order (rows of 6 px of an
    800-px-high screen by top edge, then left to right), as in MobileGym. Exact duplicates are kept once.
 5. Checks against the screenshot (pixel_check; the elements they drop are listed with the reason, never shown):
    'blank' (an icon, image, mark or switch with nothing drawn inside its box but the background), 'no ink' (a text
    with nothing, or no line of anything, drawn in its box), 'other ink' (far fewer pieces of ink than the text has
    characters: what is drawn there is something else, such as an icon button whose name is not shown; an icon-sized
    one becomes an icon), 'cut' (the text ends in an ellipsis on the screen but not in the tree, or the ink is both too
    narrow and in too few pieces for the whole text: shortened or clipped; also a text whose box touches the left or
    right edge of the screen with ink in its outermost columns). A text is measured without what other elements draw
    inside its box. 'other ink' and the width test of 'cut' are run only on mostly-Latin text, measured with the font
    DejaVu Sans (AC_FONT, default the Debian/Ubuntu path of fonts-dejavu-core).
 6. For sampling, not part of the text view: coverage(), the share of what the screenshot draws that lies in the boxes
    of the tree's elements (before the pixel checks; images left out on both sides). Screens under COVERAGE_MIN (0.6)
    are not used: their trees miss much of the screen (behind a pop-up, web content, self-drawn content).
Usage: python -m androidcontrol.extract test"""
from __future__ import annotations

import collections
import json
import os
import re
import sys

from androidcontrol import data as L
from reader.match import norm
from textview.textview import reading_order

VERSION = "AC4"
LEFT_OUT = {"system", "input_method"}
CONTAINER = re.compile(r"(Layout|LayoutCompat|ViewGroup|RecyclerView|ListView|GridView|ScrollView|ViewPager2?|WebView|"
                       r"Dialog|Spinner|TabWidget|TabHost|Toolbar|CardView|ComposeView|ViewFlipper|ViewAnimator|ViewSwitcher)$")
FIELD = re.compile(r"(EditText|AutoCompleteTextView|SearchAutoComplete)$")
SWITCH = re.compile(r"Switch")  # AC4: Switch, SwitchCompat, SwitchMaterial and library switches such as SwitchButton
MARK = {"CheckBox", "RadioButton", "CompoundButton", "AppCompatCheckBox", "AppCompatRadioButton", "MaterialCheckBox",
        "MaterialRadioButton"}
PICTURE = re.compile(r"(ImageView|ImageButton|^Image)$")
ACTION_ITEM = "ActionMenuItemView"  # a toolbar action; with no text it shows its icon

COVERAGE_MIN = 0.6  # screens whose coverage() is under this are not sampled (rule 6)


def icon_font(text: str) -> bool:
    """Text drawn only with private-use code points: an icon font's glyph (MobileGym counts it as an icon)."""
    return bool(text) and all(0xE000 <= ord(c) <= 0xF8FF or 0xF0000 <= ord(c) for c in text if not c.isspace())


def _web_nodes(nodes: dict, parent: dict) -> set:
    """Rule 1b: the nodes inside a WebView. Web pages expose text meant only for screen readers (like
    "3.5 out of 5 stars" over drawn stars), which the tree cannot tell from drawn text, so web content is not read."""
    webs = {u for u, n in nodes.items() if short(n["class_name"]) == "WebView"}
    out = set()
    for u in nodes:
        p, seen = u, set()
        while p is not None and p not in seen:
            if p in webs:
                out.add(u)
                break
            seen.add(p)
            p = parent.get(p)
    return out
GRID = 16  # points per side for the covered share


def short(cls: str) -> str:
    return cls.rsplit(".", 1)[-1]


def _inter(a, b) -> list:
    return [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]


def _covered_share(b, covers) -> float:
    if not covers:
        return 0.0
    hit = 0
    for i in range(GRID):
        x = b[0] + (i + 0.5) * (b[2] - b[0]) / GRID
        for j in range(GRID):
            y = b[1] + (j + 0.5) * (b[3] - b[1]) / GRID
            hit += any(c[0] <= x < c[2] and c[1] <= y < c[3] for c in covers)
    return hit / GRID ** 2


def visible_box(box, window_box, screen, covers, px: float) -> list | None:
    """Rule 2: the node's box as it shows, in screen pixels, or None."""
    b = _inter(box, screen)
    if window_box:
        b = _inter(b, window_box)
    if b[2] <= b[0] or b[3] <= b[1]:
        return None
    for c in covers:
        if c[0] <= b[0] and c[2] >= b[2]:  # spans the box's full width
            if c[1] <= b[1] and c[3] >= b[3]:
                return None
            if c[1] <= b[1] < c[3]:
                b[1] = c[3]
            elif c[1] < b[3] <= c[3]:
                b[3] = c[1]
    if b[2] - b[0] < 3 * px or b[3] - b[1] < 3 * px:
        return None
    return None if _covered_share(b, covers) > 0.5 else b


def _norm(b, w: int, h: int) -> list | None:
    x1, y1, x2, y2 = (max(0, min(1000, round(v))) for v in (b[0] * 1000 / w, b[1] * 1000 / h, b[2] * 1000 / w, b[3] * 1000 / h))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def _with_text_below(nodes: dict) -> set:
    """unique_ids of the nodes that have a descendant with text."""
    parent = {}
    for uid, n in nodes.items():
        for ch in n["children"]:
            parent.setdefault(ch, uid)
    out = set()
    for uid, n in nodes.items():
        if not n["text"].strip():
            continue
        p, seen = parent.get(uid), set()
        while p is not None and p not in seen:
            seen.add(p)
            out.add(p)
            p = parent.get(p)
    return out


def _parent_map(nodes: dict) -> dict:
    parent = {}
    for uid, n in nodes.items():
        for ch in n["children"]:
            parent.setdefault(ch, uid)
    return parent


def _root_path(uid: int, parent: dict) -> list:
    path, seen = [uid], {uid}
    while path[-1] in parent and parent[path[-1]] not in seen:
        path.append(parent[path[-1]])
        seen.add(path[-1])
    return path[::-1]


def _drawn_over(b_uid: int, a_uid: int, parent: dict, nodes: dict) -> bool:
    """Whether node b is drawn over node a: neither contains the other, and where their branches split, b's branch
    has the higher drawing order."""
    pa, pb = _root_path(a_uid, parent), _root_path(b_uid, parent)
    i = 0
    while i < min(len(pa), len(pb)) and pa[i] == pb[i]:
        i += 1
    if i == 0 or i == len(pa) or i == len(pb) or pa[i] not in nodes or pb[i] not in nodes:
        return False
    return nodes[pb[i]]["drawing_order"] > nodes[pa[i]]["drawing_order"]


def _edge_bars(w: dict, nodes: dict, height: int) -> list:
    """Rule 2b: bars at the top or bottom of a window (at least 90% of its width, 2-15% of the screen height,
    within 3% of the screen height of the window's bottom or in the top 12% of the screen), like a bottom navigation
    bar or an app bar; content drawn under one is hidden where the bar covers it."""
    wb = w["box"] or [0, 0, 1, height]
    bottom = min(wb[3], height)
    out = []
    for uid, n in nodes.items():
        b = n["box"]
        if not b or not n["is_visible_to_user"]:
            continue
        if b[2] - b[0] >= 0.9 * (wb[2] - wb[0]) and 0.02 * height <= b[3] - b[1] <= 0.15 * height and \
                (b[3] >= bottom - 0.03 * height or b[1] <= 0.12 * height):
            out.append(uid)
    return out


def _not_hidden_by_elements(raw: list, windows: list[dict], px: float) -> list:
    """Rule 2d: an element more than half covered by elements of its window that are drawn over it (later at
    the point where their branches of the tree split) is hidden by them, like a banner that a list scrolled over."""
    by_window = {}
    for w in windows:
        nodes = {n["unique_id"]: n for n in w["nodes"]}
        by_window[id(w)] = (nodes, _parent_map(nodes))
    out = []
    for i, (e, src) in enumerate(raw):
        nodes, parent = by_window[src["wid"]]
        b = e["bbox"]
        over = [f["bbox"] for j, (f, fs) in enumerate(raw) if j != i and fs["wid"] == src["wid"] and fs["uid"] != src["uid"]
                and min(b[2], f["bbox"][2]) > max(b[0], f["bbox"][0]) and min(b[3], f["bbox"][3]) > max(b[1], f["bbox"][1])
                and _drawn_over(fs["uid"], src["uid"], parent, nodes)]
        if not over or _covered_share(b, over) <= 0.5:
            out.append((e, src))
    return out


def extract(windows: list[dict], width: int, height: int) -> tuple[list[dict], list[dict]]:
    """The text view (elements in the layout order, ids from 1) and, per element, where it came from (for checks and
    overlays only; never part of the text view)."""
    px = width / 360
    screen = [0, 0, width, height]
    raw = []
    for w in windows:
        if w["type"] in LEFT_OUT:
            continue
        covers = [o["box"] for o in windows if o is not w and o["box"] and o["layer"] > w["layer"]]
        nodes = {n["unique_id"]: n for n in w["nodes"]}
        has_text_below = _with_text_below(nodes)
        parent = _parent_map(nodes)
        web = _web_nodes(nodes, parent)
        drawn = set()  # ancestors of nodes that draw something by the role rules (rule 2b: only such bars hide content)
        for u, n in nodes.items():
            c = short(n["class_name"])
            if n["is_visible_to_user"] and n["box"] and u not in web and (
                    FIELD.search(c) or SWITCH.search(c) or c in MARK or PICTURE.search(c) or c == ACTION_ITEM or
                    (n["text"].strip() and not CONTAINER.search(c) and u not in has_text_below)):
                p, seen = parent.get(u), set()
                while p is not None and p not in seen:
                    seen.add(p)
                    drawn.add(p)
                    p = parent.get(p)
        bars = [u for u in _edge_bars(w, nodes, height) if u in drawn]
        for n in w["nodes"]:
            if n["unique_id"] in web and short(n["class_name"]) == "WebView" and n["is_visible_to_user"] and n["box"]:
                vb = visible_box(n["box"], w["box"], screen, covers, px)  # unread, but it still hides what is under it
                if vb is not None:
                    raw.append(({"role": "_hides", "bbox": vb}, {"class": "WebView", "window": w["type"], "layer": w["layer"],
                                                                "uid": n["unique_id"], "wid": id(w)}))
            if not n["is_visible_to_user"] or not n["box"] or n["unique_id"] in web:
                continue
            over = [nodes[u]["box"] for u in bars if u != n["unique_id"] and _drawn_over(u, n["unique_id"], parent, nodes)]
            b = visible_box(n["box"], w["box"], screen, covers + over, px)
            if b is None:
                continue
            cls, text = short(n["class_name"]), n["text"].strip()
            src = {"class": cls, "window": w["type"], "layer": w["layer"], "uid": n["unique_id"], "wid": id(w)}
            cy = (b[1] + b[3]) / 2
            if FIELD.search(cls):
                hint = n["hint_text"].strip()
                if text and text != hint:
                    raw.append(({"role": "field", "text": text, "bbox": b}, src))
                elif hint:
                    raw.append(({"role": "field", "text": hint, "empty": True, "bbox": b}, src))
            elif SWITCH.search(cls):
                dw, dh = min((b[2] - b[0]) / 2, 52 * px), min(b[3] - b[1], 32 * px)
                sw = [b[2] - dw, cy - dh / 2, b[2], cy + dh / 2] if text else b
                if text:
                    raw.append(({"role": "text", "text": text, "bbox": [b[0], b[1], b[2] - dw, b[3]]}, src))
                raw.append(({"role": "switch", "on": n["is_checked"], "bbox": sw}, src))
            elif cls in MARK:
                d = min(b[3] - b[1], 32 * px)
                mk = [b[0], cy - d / 2, b[0] + d, cy + d / 2] if text else b
                raw.append(({"role": "mark", "filled": n["is_checked"], "bbox": mk}, src))
                if text:
                    raw.append(({"role": "text", "text": text, "bbox": [b[0] + d, b[1], b[2], b[3]]}, src))
            elif PICTURE.search(cls) or (cls == ACTION_ITEM and not text) or icon_font(text):
                fw, fh = n["box"][2] - n["box"][0], n["box"][3] - n["box"][1]
                if fw >= 8 * px and fh >= 8 * px:
                    raw.append(({"role": "image" if fw > 64 * px or fh > 64 * px else "icon", "bbox": b}, src))
            elif text and not CONTAINER.search(cls) and n["unique_id"] not in has_text_below:
                raw.append(({"role": "text", "text": text, "bbox": b}, src))
    raw = [x for x in _not_hidden_by_elements(raw, windows, px) if x[0]["role"] != "_hides"]
    seen, elements, sources = set(), [], []
    for e, src in raw:
        nb = _norm(e["bbox"], width, height)
        if nb is None or nb[1] >= 970:  # rule 2e: the strip of the system's gesture bar
            continue
        if e["role"] in ("text", "field") and e["bbox"][3] - e["bbox"][1] < 8 * px:  # rule 2c: too low to show a line
            continue
        e = {**e, "bbox": nb}
        key = json.dumps(e, sort_keys=True, ensure_ascii=False)
        if key in seen:
            continue
        seen.add(key)
        elements.append({**e, "_src": len(sources)})
        sources.append(src)
    # rule 3b: a text drawn inside a field with the field's own text is that field, not a second element
    fields = [e for e in elements if e["role"] == "field"]
    def in_field(e: dict) -> bool:
        cx, cy = (e["bbox"][0] + e["bbox"][2]) / 2, (e["bbox"][1] + e["bbox"][3]) / 2
        return any(norm(e["text"]) == norm(f["text"]) and f["bbox"][0] <= cx <= f["bbox"][2] and f["bbox"][1] <= cy <= f["bbox"][3]
                   for f in fields)
    elements = [e for e in elements if not (e["role"] == "text" and in_field(e))]
    ordered = reading_order(elements)
    return ([{k: v for k, v in e.items() if k != "_src"} for e in ordered],
            [{k: v for k, v in sources[e["_src"]].items() if k != "wid"} for e in ordered])


# ---------------------------------------------------------------- rule 5: checks against the screenshot

INK = 40          # a pixel is ink if a channel differs from the background by more than this
FAINT = 20        # ... and anything drawn at all if by more than this (a text with fewer such pixels shows nothing)
BLANK_DIFF = 30   # an icon or image with fewer than max(12, 0.3% of its box) pixels this far from the background of
BLANK_SHARE = 0.003  # its box border shows nothing but a background (a plain panel, or nothing drawn)
LATIN = set(map(chr, range(32, 127))) | set(map(chr, range(160, 256))) | set("\u20b9\u20ac\u00a3\u2013\u2014\u2022\u2026\u2018\u2019\u201c\u201d\u00b7")
_FONT = {}


def _measure_font():
    from PIL import ImageFont
    if "f" not in _FONT:
        _FONT["f"] = ImageFont.truetype(os.environ.get("AC_FONT", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), 100)
    return _FONT["f"]


def _runs(flags) -> list:
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i))
            start = None
    return out


def _ring_background(crop):
    import numpy as np
    ring = np.concatenate([crop[0], crop[-1], crop[:, 0], crop[:, -1]])
    return np.median(ring, axis=0)


def _text_background(crop):
    """The color behind a text: the most common color (to 16 levels a channel) of the middle of its box, which is
    the button's or chip's color when the text sits on one, else the page's."""
    import numpy as np
    h, w = crop.shape[:2]
    mid = crop[int(0.25 * h):max(int(0.75 * h), int(0.25 * h) + 1), int(0.1 * w):max(int(0.9 * w), int(0.1 * w) + 1)].reshape(-1, 3)
    q = (mid >> 4).astype(np.int32)
    key = q[:, 0] * 256 + q[:, 1] * 16 + q[:, 2]
    vals, counts = np.unique(key, return_counts=True)
    top = vals[counts.argmax()]
    return mid[key == top].mean(axis=0)


def _drop_edge_shapes(ink):
    """Removes ink that is not glyphs: shapes attached to the left or right edge of the box and at least half its
    height (the round ends of a button or chip), and shapes along the top or bottom edge over 80% of its width."""
    import numpy as np
    from collections import deque
    h, w = ink.shape
    seen = np.zeros_like(ink)
    out = ink.copy()
    starts = [(y, x) for y in range(h) for x in (0, w - 1) if ink[y, x]] + [(y, x) for x in range(w) for y in (0, h - 1) if ink[y, x]]
    for sy, sx in starts:
        if seen[sy, sx]:
            continue
        q, comp = deque([(sy, sx)]), []
        seen[sy, sx] = True
        while q:
            y, x = q.popleft()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and ink[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    q.append((ny, nx))
        ys, xs = [c[0] for c in comp], [c[1] for c in comp]
        side = (min(xs) == 0 or max(xs) == w - 1) and max(ys) - min(ys) + 1 >= 0.5 * h
        flat = (min(ys) == 0 or max(ys) == h - 1) and max(xs) - min(xs) + 1 >= 0.8 * w
        if side or flat:
            out[ys, xs] = False
    return out


def count_components(mask, min_area: int = 3) -> int:
    """Connected pieces of ink (4-neighbours, at least min_area pixels), by runs per row and union-find."""
    import numpy as np
    parent, area, prev = [], [], []
    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for row in mask:
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        cur, j = [], 0
        for s0, e0 in zip(starts.tolist(), ends.tolist()):
            lab = len(parent)
            parent.append(lab)
            area.append(e0 - s0)
            while j < len(prev) and prev[j][1] <= s0:
                j += 1
            k = j
            while k < len(prev) and prev[k][0] < e0:
                ra, rb = find(lab), find(prev[k][2])
                if ra != rb:
                    parent[rb] = ra
                    area[ra] += area[rb]
                k += 1
            cur.append((s0, e0, lab))
        prev = cur
    return sum(1 for i in range(len(parent)) if parent[i] == i and area[i] >= min_area)


def _drop_long_strokes(ink):
    """Clears the rows of horizontal strokes longer than half the box (strike-through, underline, a divider) and the
    rows next to them: no run of glyphs is unbroken that long, and such a stroke would join the letters into one
    piece (the glyphs it crosses fall into an upper and a lower piece instead)."""
    import numpy as np
    out = ink.copy()
    limit = 0.5 * ink.shape[1]
    for y in range(ink.shape[0]):
        d = np.diff(np.concatenate(([0], ink[y].astype(np.int8), [0])))
        if any(b - a >= limit for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])):
            out[max(0, y - 1):y + 2] = False  # the stroke's row and its anti-aliased neighbours
    return out


def text_lines(ink, keep: float = 0.35) -> list:
    """Lines of text in an ink mask: runs of rows with ink in at least 3% of the columns (descender-only rows fall
    below that, so lines separate), keeping runs at least 35% as tall as the tallest; (top, bottom, left, right)."""
    import numpy as np
    rows = ink.sum(axis=1) >= max(1, 0.03 * ink.shape[1])
    runs = _runs(rows)
    if not runs:
        return []
    tall = max(e - s for s, e in runs)
    out = []
    for s, e in runs:
        if e - s >= keep * tall:
            cols = np.where(ink[s:e].any(axis=0))[0]
            out.append((s, e, int(cols[0]), int(cols[-1]) + 1))
    return out


def ends_in_dots(ink, line) -> bool:
    """Whether a line ends in three small marks on its baseline: an ellipsis."""
    import numpy as np
    s, e, x0, x1 = line
    h = e - s
    sub = ink[s:e, max(x0, x1 - int(1.8 * h)):x1]
    comps = _runs(sub.any(axis=0))
    if len(comps) < 3:
        return False
    marks = []
    for c0, c1 in comps[-3:]:
        rows = np.where(sub[:, c0:c1].any(axis=1))[0]
        marks.append((c1 - c0, rows[0], rows[-1] + 1))
    gaps = [comps[-2][0] - comps[-3][1], comps[-1][0] - comps[-2][1]]
    return all(w <= 0.3 * h and top >= 0.45 * h and bot - top <= 0.3 * h for w, top, bot in marks) and max(gaps) <= 0.6 * h


def pixel_check(e: dict, im, width: int, height: int, others: list | tuple = ()) -> tuple[str | None, dict]:
    """Rule 5: why the element is not what the screen shows (None if it is), with the measurements.
    - icon or image: 'blank' if fewer than max(12, 0.3% of its box) pixels differ from the background of its box
      border by more than 12 (a plain panel, or nothing drawn);
    - text or field: 'no ink' if fewer than 10 pixels differ from the color behind the text by more than 20, or if
      what is drawn does not make one line; 'other ink' if the ink has fewer than 0.25 pieces per character (what
      is drawn there is something else, such as the icon of a button whose name is not shown); 'cut' if its last
      line ends in an ellipsis that its text does not have, or if both its text, set in one line, would be over
      1/0.6 times as wide (relative to its height) as the ink of its lines and the ink has fewer than 0.7 pieces per
      character (the tree gives the whole text where the screen shows only part of it). Both are needed because an
      icon drawn in the text's box (a launcher label, a clock before a time) makes the first alone fire, and
      touching letters make the second alone fire. The last two are tested only on text in Latin script.
    The ink of a text: pixels that differ from the color behind it (the box border's or the middle's, whichever
    leaves less ink) by more than 40, or by half the text's own contrast if that is less (light gray text); without
    the rows of horizontal strokes longer than half the box (strike-through, underline) and without shapes on the
    box's left or right edge at least half its height (the round ends of a button or chip)."""
    import numpy as np
    x1, y1, x2, y2 = (round(e["bbox"][0] * width / 1000), round(e["bbox"][1] * height / 1000),
                      round(e["bbox"][2] * width / 1000), round(e["bbox"][3] * height / 1000))
    crop = im[y1:y2, x1:x2]
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return "no ink", {}
    if e["role"] in ("icon", "image", "mark", "switch"):  # a mark or switch must be drawn too (a chip is not)
        hh, ww = crop.shape[:2]
        inner = crop[int(0.1 * hh):max(int(0.9 * hh), int(0.1 * hh) + 1), int(0.1 * ww):max(int(0.9 * ww), int(0.1 * ww) + 1)]
        drawn = int((np.abs(inner - _ring_background(crop)).max(axis=2) > BLANK_DIFF).sum())  # a shadow or border on the
        return ("blank" if drawn < max(12, BLANK_SHARE * hh * ww) else None), {"drawn": drawn}  # box's edge is not an icon
    if e["role"] not in ("text", "field") or not e.get("text"):
        return None, {}
    diffs = [np.abs(crop - bg).max(axis=2) for bg in (_text_background(crop), _ring_background(crop))]
    diff = min(diffs, key=lambda d: int((d > INK).sum()))  # text is the smaller part of its box
    if int((diff > FAINT).sum()) < 10:
        return "no ink", {"drawn": int((diff > FAINT).sum())}
    level = min(INK, max(FAINT, 0.5 * float(np.percentile(diff, 99))))  # light gray text: half its own contrast
    raw_ink = _drop_long_strokes(diff > level)
    cx, cy = (e["bbox"][0] + e["bbox"][2]) / 2, (e["bbox"][1] + e["bbox"][3]) / 2
    for o in others:  # what other elements draw inside this box is not this text (unless one holds the whole box)
        if o[0] <= cx <= o[2] and o[1] <= cy <= o[3]:
            continue
        ox1, oy1 = round(o[0] * width / 1000) - x1, round(o[1] * height / 1000) - y1
        ox2, oy2 = round(o[2] * width / 1000) - x1, round(o[3] * height / 1000) - y1
        if ox2 > 0 and oy2 > 0 and ox1 < raw_ink.shape[1] and oy1 < raw_ink.shape[0]:
            raw_ink[max(0, oy1):max(0, oy2), max(0, ox1):max(0, ox2)] = False
    for side, at_edge in ((0, e["bbox"][0] <= 1), (-1, e["bbox"][2] >= 999)):  # cut off by the screen's edge
        if at_edge and int(raw_ink[:, side].sum() + raw_ink[:, 2 * side + (1 if side == 0 else 0)].sum()) >= 2:
            return "cut", {"edge": "left" if side == 0 else "right"}
    ink = _drop_edge_shapes(raw_ink)  # strokes first: a strike-through ties letters to the edges
    lines = text_lines(ink)
    m = {"ink": round(float(ink.mean()), 3), "lines": len(lines)}
    if not lines:  # some ink, but not one line of it: no legible text is drawn here
        return "no ink", m
    text = " ".join(e["text"].split())
    m["dots"] = ends_in_dots(ink, lines[-1])
    if m["dots"] and not text.rstrip().endswith(("\u2026", "...")):
        return "cut", m
    letters = [c for c in text if not c.isspace()]
    if len(text) > 3 and sum(c in LATIN for c in letters) >= 0.9 * len(letters):  # mostly Latin, e.g. with an emoji
        text = "".join(c if c in LATIN else "M" for c in text if not 0xFE00 <= ord(c) <= 0xFE0F)
        l, t, r, b = _measure_font().getbbox(text)
        rendered = (r - l) / max(1, b - t)
        hmed = sorted(e_ - s for s, e_, _, _ in lines)[len(lines) // 2]
        observed = sum(c1 - c0 for _, _, c0, c1 in lines) / max(1, hmed)
        single = max(((c1 - c0) / max(1, e_ - s) for s, e_, c0, c1 in text_lines(ink, keep=0.0) if e_ - s >= 6), default=0.0)
        m["r"] = round(max(observed, single) / max(1e-6, rendered), 3)  # a label under an icon is a short run of its own
        m["pieces"] = round(count_components(ink) / max(1, len(text.replace(" ", ""))), 3)
        if m["pieces"] < 0.25:  # far too few pieces of ink for this text: what is drawn here is something else
            return "other ink", m
        if m["r"] < 0.6 and m["pieces"] < 0.7:
            return "cut", m
    return None, m


def coverage(windows: list[dict], elements: list[dict], png: bytes, width: int, height: int) -> float | None:
    """Share of what the screenshot draws that lies in the text view's boxes (rule 6, for sampling): edge pixels
    (a channel changing by more than 30 between neighbours, at a quarter of the resolution) inside the boxes of the
    elements other than images, widened by 1% of the width, over all edge pixels outside the status and navigation
    bars, the keyboard, the gesture strip and the image elements; None when under 0.2% of the area has edges
    there. A low share means the tree misses much of the screen (self-drawn content, web pages, the screen behind a
    pop-up)."""
    import io
    import numpy as np
    from PIL import Image
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB").resize((width // 4, height // 4), Image.BILINEAR)).astype(np.int16)
    h4, w4 = a.shape[:2]
    edge = np.zeros((h4, w4), bool)
    edge[:, :-1] |= np.abs(a[:, 1:] - a[:, :-1]).max(axis=2) > 30
    edge[:-1, :] |= np.abs(a[1:, :] - a[:-1, :]).max(axis=2) > 30
    region = np.ones((h4, w4), bool)
    region[int(0.97 * h4):] = False
    region[:int(0.05 * h4)] = False  # the status bar, also where no system window is recorded
    for w in windows:
        if w["type"] in LEFT_OUT and w["box"]:
            x1, y1, x2, y2 = (max(0, v // 4) for v in w["box"])
            region[y1:y2, x1:x2] = False
    covered = np.zeros((h4, w4), bool)
    m = 0.01 * w4
    for e in elements:
        x1, y1, x2, y2 = e["bbox"]
        ys, xs = slice(max(0, int(y1 * h4 / 1000 - m)), int(y2 * h4 / 1000 + m) + 1), slice(max(0, int(x1 * w4 / 1000 - m)), int(x2 * w4 / 1000 + m) + 1)
        if e["role"] == "image":  # a picture's own edges say nothing about whether the tree has the screen's text
            region[int(y1 * h4 / 1000):int(y2 * h4 / 1000) + 1, int(x1 * w4 / 1000):int(x2 * w4 / 1000) + 1] = False
        else:
            covered[ys, xs] = True
    total = int((edge & region).sum())
    if total < 0.002 * h4 * w4:  # nothing but pictures, or an empty screen
        return None
    return round(float((edge & region & covered).sum()) / total, 3)


def text_view(windows: list[dict], png: bytes, width: int, height: int) -> tuple[list[dict], list[dict], list[dict]]:
    """extract() and then pixel_check(): the text view, the sources of its elements, and the elements left out by
    the checks (with the reason and the measurements)."""
    import io
    import numpy as np
    from PIL import Image
    els, srcs = extract(windows, width, height)
    im = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(np.int16)
    if im.shape[1] != width or im.shape[0] != height:
        raise ValueError(f"screenshot {im.shape[1]}x{im.shape[0]} is not {width}x{height}")
    px = width / 360
    kept, kept_src, dropped = [], [], []
    for i, (e, s) in enumerate(zip(els, srcs)):
        others = [o["bbox"] for j, o in enumerate(els) if j != i]
        why, m = pixel_check(e, im, width, height, others)
        b = e["bbox"]
        small = (b[2] - b[0]) * width / 1000 <= 64 * px and (b[3] - b[1]) * height / 1000 <= 64 * px
        if why == "other ink" and small:  # a button that draws an icon instead of its name is an icon
            dropped.append({"reason": "other ink, kept as icon", "element": e, "source": s, "measure": m})
            e = {"role": "icon", "bbox": b}
            why = None
        if why:
            dropped.append({"reason": why, "element": e, "source": s, "measure": m})
        else:
            kept.append({k: v for k, v in e.items() if k != "id"})
            kept_src.append({**s, "measure": m})
    ordered = reading_order(kept)
    return ordered, kept_src, dropped


# ---------------------------------------------------------------- self-test

def _enc_varint(v: int) -> bytes:
    v &= (1 << 64) - 1
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _enc(num: int, value) -> bytes:
    if isinstance(value, (bool, int)):
        return _enc_varint(num << 3) + _enc_varint(int(value))
    data = value.encode() if isinstance(value, str) else value
    return _enc_varint(num << 3 | 2) + _enc_varint(len(data)) + data


def test() -> None:
    """Everything a node can say that is not drawn (description, resource name, tooltip, interaction flags) must stay
    out of the text view, while what is drawn stays in."""
    rect = lambda l, t, r, b: _enc(1, l) + _enc(2, t) + _enc(3, r) + _enc(4, b)
    secret = {4: "SECRET_DESCRIPTION", 10: "SECRET_RESOURCE", 31: "SECRET_TOOLTIP"}
    def node(uid, cls, box, text="", hint="", checked=False, children=()):
        m = _enc(1, uid) + _enc(2, rect(*box)) + _enc(3, cls) + _enc(7, text) + _enc(5, hint) + _enc(13, checked) + _enc(23, True)
        m += b"".join(_enc(k, v) for k, v in secret.items()) + b"".join(_enc(k, True) for k in (12, 14, 15, 16, 17, 19, 21, 22))
        if children:
            m += _enc(25, b"".join(_enc_varint(c) for c in children))
        return m
    app = [node(1, "android.widget.FrameLayout", (0, 0, 1080, 2400), children=(2, 3, 4, 5, 6, 7)),
           node(2, "android.widget.TextView", (40, 300, 600, 380), text="Visible label"),
           node(3, "android.widget.ImageButton", (900, 300, 1040, 440), text=""),
           node(4, "android.widget.EditText", (40, 500, 1040, 620), hint="Search here"),
           node(5, "android.widget.CheckBox", (40, 700, 700, 800), text="Remember me", checked=True),
           node(6, "android.webkit.WebView", (0, 900, 1080, 1800), text="Page title not drawn"),
           node(7, "android.widget.TextView", (40, 2300, 600, 2390), text="Under the keyboard")]
    tree = b"".join(_enc(1, n) for n in app)
    win = lambda typ, layer, box, body: _enc(1, rect(*box)) + _enc(4, layer) + _enc(6, typ) + _enc(11, body)
    forest = _enc(1, win(1, 0, (0, 0, 1080, 2400), tree)) + _enc(1, win(2, 2, (0, 2000, 1080, 2400), _enc(1, node(9, "android.widget.TextView", (0, 2000, 300, 2100), text="key"))))
    forest += _enc(1, win(3, 1, (0, 0, 1080, 130), _enc(1, node(10, "android.widget.TextView", (0, 0, 200, 130), text="9:41"))))
    els, _ = extract(L.forest(forest), 1080, 2400)
    dump = json.dumps(els)
    assert "SECRET" not in dump, dump
    texts = [e.get("text") for e in els]
    assert "Visible label" in texts and "Search here" in texts and "Remember me" in texts, texts
    assert not {"Page title not drawn", "Under the keyboard", "key", "9:41"} & set(texts), texts
    roles = collections.Counter(e["role"] for e in els)
    assert roles == {"text": 2, "icon": 1, "field": 1, "mark": 1}, roles
    assert [e for e in els if e["role"] == "field"][0].get("empty") and [e for e in els if e["role"] == "mark"][0]["filled"]
    print("extract test passed:", dump)


if __name__ == "__main__":
    if sys.argv[1:2] != ["test"]:
        raise SystemExit(__doc__.split("Usage: ")[1])
    test()
