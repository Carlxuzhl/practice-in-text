"""Matching a reader's list against the true text view of the same screen, element by element, and the error classes
measured from that matching. The same matching serves the element error rate, the recurrence analysis and the fitting
of rate-matched noise (practice/noise.py)."""
from __future__ import annotations

import collections
import difflib
import re
import unicodedata

SAME_LABEL = 0.75  # difflib ratio at which a reader text counts as the same label, misread
SAME_SPOT = 0.1  # IoU at which an element without text counts as the one at that spot
HEAVY_SPOT = 0.5  # a text box at the same spot whose text is too far off to be "similar" is still that element misread
PLAIN = (None, "", "white", "black")
FAR = 200  # a reader text whose centre is further than 100 units (|dx| + |dy|) away is another element, not this one moved
RELATED = 0.5  # an extra text this similar to a true text of its screen (difflib ratio) comes from it


def norm(text: str | None) -> str:
    """Text up to width, case, spacing and a trailing ellipsis."""
    t = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text or "").lower())
    return re.sub(r"\.{3}$", "", t)


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def gap(a, b) -> int:
    return abs(a[0] + a[2] - b[0] - b[2]) + abs(a[1] + a[3] - b[1] - b[3])


def coloured(e: dict) -> bool:
    return e.get("bg") not in PLAIN


def text_of(e: dict) -> str:
    return e.get("text") or ""


def same_name(a: str | None, b: str | None) -> bool:
    key = lambda n: re.sub(r"[_\s]+", "-", (n or "").lower())
    return key(a) == key(b)


def app_of(template: str) -> str:
    """The app of a template id ("wechat.SetSignature" -> "wechat")."""
    return template.split(".", 1)[0]


def char_class(c: str) -> str:
    if c.isdigit():
        return "digit"
    if c.isascii() and c.isalpha():
        return "latin"
    if "\u4e00" <= c <= "\u9fff":
        return "cjk"
    return "other"


def related_source(x: dict, truth: list[dict]) -> tuple[str, int, dict] | None:
    """The true text of the screen an extra reader element x comes from, or None: the same text ("same"), a part of it
    ("part": where it starts and how long it is, as shares of the true text), the true text with more around it ("more":
    what comes before and after), or a near variant ("near": difflib ratio >= RELATED). The closest kind first, then the
    highest ratio, then the nearest box."""
    nx = norm(text_of(x))
    if not nx:
        return None
    best = None
    for i, u in enumerate(truth):
        nu = norm(text_of(u))
        if not nu:
            continue
        if nx == nu:
            rank, kind, params = (0, 0.0), "same", {}
        elif nx in nu:
            rank, kind, params = (1, 0.0), "part", {"start": nu.find(nx) / len(nu), "length": len(nx) / len(nu)}
        elif nu in nx:
            k = nx.find(nu)
            rank, kind, params = (2, 0.0), "more", {"before": nx[:k], "after": nx[k + len(nu):]}
        else:
            r = difflib.SequenceMatcher(None, nu, nx).ratio()
            if r < RELATED:
                continue
            rank, kind, params = (3, -r), "near", {}
        key = (*rank, gap(u["bbox"], x["bbox"]))
        if best is None or key < best[0]:
            best = (key, kind, i, params)
    return None if best is None else (best[1], best[2], best[3])


def match(truth: list[dict], read: list[dict]) -> dict:
    """Pairs the true elements with the reader's, one to one except that several true texts may sit in one reader line.
    Returns per true element its outcome, and the reader elements no true element accounts for (extras):
      kept   (index of the reader element)  the element is there, maybe misread, renamed, moved or in another state;
      merged (index)                        its text is in a reader element that also holds another true text;
      shape  (index)                        a coloured text element written as a mark or an icon at its spot;
      miss                                  nothing stands for it."""
    out: list[tuple[str, int | None]] = [("miss", None)] * len(truth)
    used: set[int] = set()
    nr = [norm(text_of(r)) for r in read]
    # 1. texts: the same text, or a similar one, nearest first
    cands = []
    for i, t in enumerate(truth):
        nt = norm(text_of(t))
        if not nt:
            continue
        for j, r in enumerate(read):
            if not nr[j] or gap(t["bbox"], r["bbox"]) > FAR:
                continue
            if nr[j] == nt:
                cands.append((0, gap(t["bbox"], r["bbox"]), i, j))
            elif difflib.SequenceMatcher(None, nt, nr[j]).ratio() >= SAME_LABEL:
                cands.append((1, gap(t["bbox"], r["bbox"]), i, j))
    for _, _, i, j in sorted(cands):
        if out[i][0] == "miss" and j not in used:
            out[i] = ("kept", j)
            used.add(j)
    # 2. texts inside a longer reader line that overlaps them: merged when the line holds two or more true texts
    inside = collections.defaultdict(list)
    for i, t in enumerate(truth):
        nt = norm(text_of(t))
        if out[i][0] != "miss" or not nt:
            continue
        hits = [j for j, r in enumerate(read) if nr[j] and nt in nr[j] and len(nr[j]) > len(nt) and overlaps(t["bbox"], r["bbox"])]
        if hits:
            inside[min(hits, key=lambda j: gap(t["bbox"], read[j]["bbox"]))].append(i)
    for j, members in inside.items():
        holder = [i for i, (kind, k) in enumerate(out) if kind == "kept" and k == j]
        if len(members) + len(holder) >= 2:
            for i in members + holder:
                out[i] = ("merged", j)
        elif j not in used:
            out[members[0]] = ("kept", j)
        used.add(j)
    # 3. a text box at the same spot with text too far off: that element, misread
    for i, t in enumerate(truth):
        if out[i][0] != "miss" or not norm(text_of(t)):
            continue
        spot = [(iou(t["bbox"], r["bbox"]), j) for j, r in enumerate(read) if j not in used and nr[j] and r["role"] in ("text", "field")]
        best = max(spot, default=(0, None))
        if best[0] >= HEAVY_SPOT:
            out[i] = ("kept", best[1])
            used.add(best[1])
    # 4. elements without text by their spot, whatever the reader called them, the same role first
    cands = []
    for i, t in enumerate(truth):
        if out[i][0] != "miss" or norm(text_of(t)):
            continue
        for j, r in enumerate(read):
            if j in used:
                continue
            o = iou(t["bbox"], r["bbox"])
            if o >= SAME_SPOT:
                cands.append((r["role"] != t["role"], -o, i, j))
    for _, _, i, j in sorted(cands):
        if out[i][0] == "miss" and j not in used:
            out[i] = ("kept", j)
            used.add(j)
    # 5. a coloured text element whose spot the reader filled with a mark or an icon
    for i, t in enumerate(truth):
        if out[i][0] != "miss" or not coloured(t) or not norm(text_of(t)):
            continue
        spot = [(iou(t["bbox"], r["bbox"]), j) for j, r in enumerate(read) if j not in used and r["role"] in ("mark", "icon") and not nr[j]]
        best = max(spot, default=(0, None))
        if best[0] >= SAME_SPOT:
            out[i] = ("shape", best[1])
            used.add(best[1])
    extras = [j for j in range(len(read)) if j not in used]
    return {"outcomes": out, "extras": extras}


def rates(pairs: list[tuple[list[dict], list[dict]]]) -> dict:
    """The rate of every error class over screens (true list, reader list), measured with match()."""
    c = collections.Counter()
    for truth, read in pairs:
        m = match(truth, read)
        c["screens"] += 1
        c["extras"] += len(m["extras"])
        for j in m["extras"]:
            if norm(text_of(read[j])):
                c["text_extras"] += 1
                c["related_extras"] += related_source(read[j], truth) is not None
        for t, (kind, j) in zip(truth, m["outcomes"]):
            c["elements"] += 1
            c["miss"] += kind == "miss"
            if coloured(t) and norm(text_of(t)):
                c["coloured_texts"] += 1
                c["shape"] += kind == "shape"
            if norm(text_of(t)):
                c["texts"] += 1
                c["merged"] += kind == "merged"
                if kind == "kept":
                    c["texts_kept"] += 1
                    c["misread"] += norm(text_of(read[j])) != norm(text_of(t))
            if t["role"] == "icon" and t.get("name") and kind == "kept":
                c["named_icons_kept"] += 1
                c["renamed"] += not (read[j]["role"] == "icon" and same_name(read[j].get("name"), t["name"]))
            if t["role"] in ("switch", "mark") and kind == "kept" and read[j]["role"] == t["role"]:
                state = "on" if t["role"] == "switch" else "filled"
                c["states_kept"] += 1
                c["state_wrong"] += bool(read[j].get(state)) != bool(t.get(state))
            if kind == "kept":
                c["kept"] += 1
                c["offset_sum"] += sum(abs(a - b) for a, b in zip(t["bbox"], read[j]["bbox"])) / 4
    div = lambda a, b: round(c[a] / c[b], 4) if c[b] else None
    return {"screens": c["screens"], "elements": c["elements"],
            "1_miss": div("miss", "elements"), "2_shape": div("shape", "coloured_texts"), "3_misread": div("misread", "texts_kept"),
            "3_merged": div("merged", "texts"), "4_renamed": div("renamed", "named_icons_kept"), "5_state_wrong": div("state_wrong", "states_kept"),
            "6_mean_offset": div("offset_sum", "kept"), "7_extras_per_screen": div("extras", "screens"),
            "7_text_extras_per_screen": div("text_extras", "screens"), "7_text_extras_related": div("related_extras", "text_extras"),
            "counts": {k: c[k] for k in ("coloured_texts", "texts_kept", "named_icons_kept", "states_kept", "kept", "text_extras")}}
