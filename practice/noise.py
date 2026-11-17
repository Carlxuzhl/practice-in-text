"""Practice environments with corrupted text views: generic noise and rate-matched noise.

Both corrupt the true text view of each screen before the policy sees it (policy/episodes.py --noise).

Generic noise (Generic) is fixed in advance: each episode draws a rate of leaving an element out in [0, 0.2], a
character error rate in [0, 0.05], a box jitter in [0, 15] units, a switch / mark flip rate in [0, 0.1] and a list
truncation probability in [0, 0.1]; every step then applies them independently.

Rate-matched noise (Calibrated) is fitted on the calibration screenshots: every read screen pairs the true text view with
what the reader wrote for it, element by element (reader.match.match). Counted by element features (role, text length,
digits, box size, coloured background, region, elements on the screen; features()), it gives each true element an
outcome as observed for elements like it:
  1. left out;
  2. a coloured button written as a filled mark of its colour, without its text;
  3. text misread by the observed character substitutions, deletions and insertions, or written into one element with
     its right-hand neighbour;
  4. an icon renamed by the observed table (true name -> the name the reader wrote, none included), or read as another
     role;
  5. a switch or mark read in the wrong state, by its role, true state and colour (state_keys());
  6. its box moved by an offset observed for its role and size, so that a tap lands where the moved box is and the
     simulator decides what it hits (the episode runner taps the centre of the box the policy saw);
  7. extra elements. An extra the reader wrote that comes from a true text of its screen (the same text again, a part of
     it, it with more around it, or a near variant; reader.match.related_source) is made from this screen's own texts:
     each true text spawns as many as observed for texts like it, by whether it is written or left out, each by an
     observed operation and box offset. The other extras, as many as observed on screens with as many elements, are
     drawn from those the reader wrote on screens of the same app;
  8. the list cut short as the reader's output was cut for screens with as many elements.
The list then goes in the layout order. Within one episode an unchanged screen gives the same list, as the reader at
temperature 0 does: the draws are seeded from the episode and the screen's content. The draws are independent across
elements and episodes; the same model applied with these independent draws is also the control of the recurrence
analysis (reader/recurrence.py).

fit_check(): the 20% of the episodes held out by hash; the rates of every class under the model fitted on the other 80%
against the reader's own on the same screens, both measured with the same matching. Classes 1, 2 and 5 must be within
5 points.

Usage: python -m practice.noise fit <out.json> <run_dir>... [--episodes calibration_set.json --key calibration]
       python -m practice.noise check <out.json> <run_dir>... [--episodes calibration_set.json --key calibration]
--episodes/--key keep only the episodes that file lists under that key (the calibration set); without them every
episode of the runs counts."""
from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import random
from pathlib import Path

from common.records import episodes_from, screens_of
from reader.match import app_of, char_class, coloured, match, norm, rates, related_source, same_name, text_of
from textview.textview import reading_order

VERSION = "V1.3"  # the model file format; a fitted model carries it
BACKOFF = [(0, 1, 2, 3, 4, 5, 6), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4), (0, 1, 3, 4), (0, 4), (0,)]
MIN_COUNT = 20  # a feature group needs this many observations, or the next coarser one is used
MIN_STATE = 10  # the same for the state of switches and marks
HOLDOUT = 0.2


# ---------------------------------------------------------------- element features

def features(e: dict, n: int) -> tuple:
    """role, text length, digits, box size, coloured background, region, elements on the screen."""
    t = len(norm(text_of(e)))
    length = 0 if t == 0 else 1 if t <= 2 else 2 if t <= 6 else 3 if t <= 15 else 4
    x1, y1, x2, y2 = e["bbox"]
    area = (x2 - x1) * (y2 - y1)
    size = 0 if area < 800 else 1 if area < 2500 else 2 if area < 10000 else 3
    y = (y1 + y2) / 2
    region = "status" if y2 <= 45 else "top" if y < 150 else "bottom" if y >= 880 else "middle"
    crowd = 0 if n < 20 else 1 if n < 40 else 2 if n < 70 else 3
    return (e["role"], length, any(c.isdigit() for c in text_of(e)), size, coloured(e), region, crowd)


def state_keys(e: dict) -> list[tuple]:
    """Groups for a switch or mark's state, finest first: its role, its true state and its colour (the reader reads small
    white filled marks as empty and orange ones right), then without the colour, then the role alone."""
    on = bool(e.get("on" if e["role"] == "switch" else "filled"))
    return [(e["role"], on, e.get("color") or ""), (e["role"], on), (e["role"],)]


def right_neighbour(i: int, elements: list[dict]) -> int | None:
    """The nearest text element to the right of element i on the same line, if any."""
    a = elements[i]["bbox"]
    best = None
    for j, e in enumerate(elements):
        b = e["bbox"]
        if j == i or not text_of(e) or b[0] < a[2] - 2:
            continue
        v = min(a[3], b[3]) - max(a[1], b[1])
        if v < 0.5 * min(a[3] - a[1], b[3] - b[1]) or b[0] - a[2] > 60:
            continue
        if best is None or b[0] < elements[best]["bbox"][0]:
            best = j
    return best


def clip_box(b) -> list[int]:
    x1, y1, x2, y2 = (max(0, min(1000, int(round(v)))) for v in b)
    if x2 <= x1:
        x2 = min(1000, x1 + 2) if x1 < 999 else x1; x1 = min(x1, x2 - 1)
    if y2 <= y1:
        y2 = min(1000, y1 + 2) if y1 < 999 else y1; y1 = min(y1, y2 - 1)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------- rate-matched noise: fitting

def fit(screens: list[dict]) -> dict:
    """The calibration model from screens {"truth": true list, "read": reader list, "cut": output cut by length}."""
    outcome = collections.defaultdict(collections.Counter)  # feature key at each back-off level -> outcome counts
    misread = collections.defaultdict(collections.Counter)  # ... -> misread / ok among kept texts
    merge = collections.defaultdict(collections.Counter)  # ... -> starts a merge / not, among texts with a right neighbour
    role_as = collections.defaultdict(collections.Counter)  # true role -> role written, among kept
    names = collections.defaultdict(collections.Counter)  # true icon name -> name written ("" for none), among icons kept as icons
    icon_text = collections.defaultdict(collections.Counter)  # true icon name -> the text written, among icons written as text
    state = collections.defaultdict(collections.Counter)  # state_keys() -> wrong / right, among switches and marks kept as such
    offsets = collections.defaultdict(list)  # (role, size) and (role,) -> observed box offsets of kept elements
    subs = collections.defaultdict(collections.Counter)  # character -> what it was misread as ("" deleted)
    class_subs = collections.defaultdict(collections.Counter)  # character class -> replacement strings
    inserts = collections.Counter()
    n_ops = collections.Counter()
    op_kind = collections.Counter()
    spawn = {"there": collections.defaultdict(collections.Counter), "miss": collections.defaultdict(collections.Counter)}
    spawn_ops: list[dict] = []  # every extra the reader made from a true text of its screen: how (related_source) and where
    op_index = collections.defaultdict(list)  # (written or left out, role, text length) and coarser -> indices into spawn_ops
    unrel_n = collections.defaultdict(collections.Counter)  # crowd -> extras per screen that come from no true text
    unrel_pool = collections.defaultdict(list)  # app -> those extras as the reader wrote them
    cut = collections.defaultdict(collections.Counter)  # crowd -> cut / not
    cut_at: list[float] = []  # where in the true list (reading order) a cut output stopped
    for s in screens:
        truth, read = reading_order(s["truth"]), s["read"]
        n = len(truth)
        m = match(truth, read)
        crowd = features({"role": "text", "bbox": [0, 0, 1, 1]}, n)[6]
        spawned = collections.Counter()
        unrel = 0
        for j in m["extras"]:
            x = read[j]
            src = related_source(x, truth)
            if src is None:
                unrel += 1
                unrel_pool[app_of(s.get("id", ""))].append({k: v for k, v in x.items() if k != "id"})
                continue
            kind, i, params = src
            spawned[i] += 1
            u = truth[i]
            written = "miss" if m["outcomes"][i][0] == "miss" else "there"
            f = features(u, n)
            spawn_ops.append({"kind": kind, **params, "d": [b - a for a, b in zip(u["bbox"], x["bbox"])],
                        "x": {k: v for k, v in x.items() if k not in ("id", "text", "bbox")}})
            for key in ((written, f[0], f[1]), (written, f[0]), (written,)):
                op_index[key].append(len(spawn_ops) - 1)
        unrel_n[crowd][unrel] += 1
        for i, u in enumerate(truth):
            if norm(text_of(u)):
                f = features(u, n)
                for lv in BACKOFF:
                    spawn["miss" if m["outcomes"][i][0] == "miss" else "there"][tuple(f[k] for k in lv)][str(spawned[i])] += 1
        cut[crowd]["cut" if s.get("cut") else "whole"] += 1
        found = [i for i, (kind, _) in enumerate(m["outcomes"]) if kind != "miss"]
        if s.get("cut") and n:
            cut_at.append((max(found) + 1) / n if found else 0.0)
        starts = set()
        for j in set(k for kind, k in m["outcomes"] if kind == "merged"):
            members = sorted((i for i, (kind, k) in enumerate(m["outcomes"]) if kind == "merged" and k == j), key=lambda i: truth[i]["bbox"][0])
            starts.add(members[0])
        there = lambda i: m["outcomes"][i][0] in ("kept", "merged")
        for i, (t, (kind, j)) in enumerate(zip(truth, m["outcomes"])):
            f = features(t, n)
            for lv in BACKOFF:
                outcome[tuple(f[k] for k in lv)][kind if kind != "merged" else "kept"] += 1
            nb = right_neighbour(i, truth) if norm(text_of(t)) else None
            if nb is not None and there(i) and there(nb):  # as in apply(): both are written, and the left one may take in the right
                for lv in BACKOFF:
                    merge[tuple(f[k] for k in lv)]["start" if i in starts else "no"] += 1
            if kind != "kept":
                continue
            r = read[j]
            role_as[t["role"]][r["role"]] += 1
            offsets[(t["role"], f[3])].append([b - a for a, b in zip(t["bbox"], r["bbox"])])
            offsets[(t["role"],)].append(offsets[(t["role"], f[3])][-1])
            if norm(text_of(t)) and r["role"] in ("text", "field"):
                wrong = norm(text_of(r)) != norm(text_of(t))
                for lv in BACKOFF:
                    misread[tuple(f[k] for k in lv)]["wrong" if wrong else "ok"] += 1
                if wrong:
                    ops = [o for o in difflib.SequenceMatcher(None, text_of(t), text_of(r)).get_opcodes() if o[0] != "equal"]
                    n_ops[min(len(ops), 4)] += 1
                    for tag, a1, a2, b1, b2 in ops:
                        op_kind[tag] += 1
                        a, b = text_of(t)[a1:a2], text_of(r)[b1:b2]
                        if tag == "insert":
                            inserts[b] += 1
                        elif tag == "delete":
                            for ch in a:
                                subs[ch][""] += 1
                        elif len(a) == len(b):
                            for ca, cb in zip(a, b):
                                subs[ca][cb] += 1
                                class_subs[char_class(ca)][cb] += 1
                        else:
                            subs[a[0]][b] += 1
                            class_subs[char_class(a[0])][b] += 1
            if t["role"] == "icon" and r["role"] == "icon":
                names[t.get("name") or ""][r.get("name") or ""] += 1
            if t["role"] == "icon" and r["role"] in ("text", "field"):
                icon_text[t.get("name") or ""][text_of(r)] += 1
            if t["role"] in ("switch", "mark") and r["role"] == t["role"]:
                key = "on" if t["role"] == "switch" else "filled"
                for k in state_keys(t):
                    state[k]["wrong" if bool(r.get(key)) != bool(t.get(key)) else "right"] += 1
    enc = lambda d: {json.dumps(k, ensure_ascii=False): dict(v) for k, v in d.items()}
    return {"version": VERSION, "screens": len(screens), "outcome": enc(outcome), "misread": enc(misread), "merge": enc(merge),
            "role_as": {k: dict(v) for k, v in role_as.items()}, "names": {k: dict(v) for k, v in names.items()},
            "icon_text": {k: dict(v) for k, v in icon_text.items()},
            "state": enc(state), "offsets": {json.dumps(k): v for k, v in offsets.items()},
            "subs": {k: dict(v) for k, v in subs.items()}, "class_subs": {k: dict(v) for k, v in class_subs.items()},
            "inserts": dict(inserts), "n_ops": {str(k): v for k, v in n_ops.items()}, "op_kind": dict(op_kind),
            "spawn": {w: enc(t) for w, t in spawn.items()}, "ops": spawn_ops, "op_index": {json.dumps(k): v for k, v in op_index.items()},
            "unrel_n": {str(k): {str(a): b for a, b in v.items()} for k, v in unrel_n.items()}, "unrel_pool": dict(unrel_pool),
            "cut": {str(k): dict(v) for k, v in cut.items()}, "cut_at": cut_at}


# ---------------------------------------------------------------- rate-matched noise: sampling

def draw(counter: dict, rng: random.Random, default=None):
    """One key of a count table, in proportion to its count."""
    items = sorted(counter.items(), key=lambda kv: str(kv[0]))
    total = sum(v for _, v in items)
    if total <= 0:
        return default
    x = rng.random() * total
    for k, v in items:
        x -= v
        if x < 0:
            return k
    return items[-1][0]


class Calibrated:
    """Rate-matched noise; apply() turns a true list into a list as the reader would have written it."""

    def __init__(self, model: dict):
        if model.get("version") != VERSION:
            raise ValueError(f"calibration model {model.get('version')}, not {VERSION}")
        dec = lambda d: {tuple(json.loads(k)): v for k, v in d.items()}
        self.m = model
        self.outcome, self.misread, self.merge = dec(model["outcome"]), dec(model["misread"]), dec(model["merge"])
        self.state = dec(model["state"])
        self.spawn = {w: dec(t) for w, t in model["spawn"].items()}
        self.op_index = {tuple(json.loads(k)): v for k, v in model["op_index"].items()}
        self.unrel_all = [e for pool in model["unrel_pool"].values() for e in pool]
        self.offsets = {tuple(json.loads(k)): v for k, v in model["offsets"].items()}
        self.icon_text_all = collections.Counter()
        for v in model["icon_text"].values():
            self.icon_text_all.update(v)
        self.all_names = collections.Counter()
        for v in model["names"].values():
            self.all_names.update(v)
        self.digest = hashlib.sha256(json.dumps(model, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

    def table(self, tables: dict, f: tuple) -> dict:
        for lv in BACKOFF:
            c = tables.get(tuple(f[k] for k in lv))
            if c and sum(c.values()) >= MIN_COUNT:
                return c
        return tables.get(tuple(f[k] for k in BACKOFF[-1])) or {}

    def misread_text(self, text: str, rng: random.Random) -> str:
        if not text:
            return text
        for _ in range(4):
            chars = list(text)
            for _ in range(int(draw(self.m["n_ops"], rng, "1"))):
                kind = draw(self.m["op_kind"], rng, "replace")
                pos = rng.randrange(len(chars) + (kind == "insert")) if chars else 0
                if kind == "insert":
                    chars.insert(pos, draw(self.m["inserts"], rng, ""))
                elif not chars:
                    continue
                elif kind == "delete":
                    chars.pop(pos)
                else:
                    c = chars[pos]
                    table = self.m["subs"].get(c)
                    if not table or sum(table.values()) < 2:
                        table = self.m["class_subs"].get(char_class(c)) or {}
                    chars[pos] = draw({k: v for k, v in table.items() if k}, rng, c)
            out = "".join(chars)
            if norm(out) != norm(text):
                return out
        return out

    def rename(self, name: str | None, rng: random.Random) -> str | None:
        table = self.m["names"].get(name or "")
        if not table or sum(table.values()) < 3:
            same = sum(v for n, row in self.m["names"].items() for k, v in row.items() if same_name(k, n) and n)
            none = sum(row.get("", 0) for n, row in self.m["names"].items() if n)
            total = sum(sum(row.values()) for n, row in self.m["names"].items() if n)
            x = rng.random() * max(1, total)
            if x < same:
                return name
            if x < same + none:
                return None
            return draw({k: v for k, v in self.all_names.items() if k}, rng, name)
        out = draw(table, rng, name)
        return out or None

    def apply(self, obs: dict, key: str, app: str | None = None) -> dict:
        """The list for this screen. key fixes the draws: the episode and the screen's content (policy/episodes.py); it
        starts with the template id, which gives the app unless app is given. The seed salt "A2" is kept as the paper's
        runs had it, so that the same key gives the same draws."""
        app = app_of(key.split("|", 1)[0]) if app is None else app
        rng = random.Random(hashlib.sha256(f"A2|{key}".encode()).hexdigest())
        truth = reading_order(obs["elements"])
        n = len(truth)
        out: list[dict] = []
        absorbed: set[int] = set()
        plan = []
        for i, t in enumerate(truth):
            f = features(t, n)
            plan.append((f, draw(self.table(self.outcome, f), rng, "kept")))
        for i, (t, (f, kind)) in enumerate(zip(truth, plan)):
            if i in absorbed or kind == "miss":
                continue
            if kind == "shape":
                out.append({"role": "mark", "filled": True, "color": t.get("bg") or t.get("color"), "bbox": self.move(t, f, rng)})
                continue
            e = {k: v for k, v in t.items() if k != "id"}
            if norm(text_of(t)):
                nb = right_neighbour(i, truth)
                if nb is not None and nb not in absorbed and plan[nb][1] != "miss" and draw(self.table(self.merge, f), rng, "no") == "start":
                    other = truth[nb]
                    absorbed.add(nb)
                    e["text"] = text_of(t) + " " + text_of(other)
                    e["bbox"] = [min(t["bbox"][0], other["bbox"][0]), min(t["bbox"][1], other["bbox"][1]),
                                 max(t["bbox"][2], other["bbox"][2]), max(t["bbox"][3], other["bbox"][3])]
                    out.append({**e, "bbox": self.move(e, f, rng)})
                    continue
            role = draw(self.m["role_as"].get(t["role"]) or {t["role"]: 1}, rng, t["role"])
            e = self.as_role(e, role, rng)
            if role in ("text", "field") and norm(text_of(t)) and draw(self.table(self.misread, f), rng, "ok") == "wrong":
                e["text"] = self.misread_text(text_of(t), rng)
            if role in ("switch", "mark") and role == t["role"]:
                s = next((self.state[k] for k in state_keys(t) if sum(self.state.get(k, {}).values()) >= MIN_STATE), None) \
                    or self.state.get(state_keys(t)[-1]) or {}
                if draw(s, rng, "right") == "wrong":
                    k = "on" if role == "switch" else "filled"
                    e[k] = not bool(t.get(k))
            e["bbox"] = self.move(t, f, rng)
            out.append(e)
        for t, (f, kind) in zip(truth, plan):  # extras made from this screen's own texts
            if not norm(text_of(t)):
                continue
            written = "miss" if kind == "miss" else "there"
            for _ in range(int(draw(self.table(self.spawn[written], f), rng, "0"))):
                op = self.op_for(written, f, rng)
                if op is not None:
                    out.append(self.spawned(t, op, rng))
        crowd = str(features({"role": "text", "bbox": [0, 0, 1, 1]}, n)[6])
        k = int(draw(self.m["unrel_n"].get(crowd) or {"0": 1}, rng, "0"))
        pool = self.m["unrel_pool"].get(app) or []
        if len(pool) < MIN_COUNT:
            pool = self.unrel_all
        for _ in range(k if pool else 0):
            out.append(dict(pool[rng.randrange(len(pool))]))
        out = reading_order(out)
        c = self.m["cut"].get(crowd) or {}
        if self.m["cut_at"] and truth and draw(c, rng, "whole") == "cut":  # nothing below where the output stopped
            at = self.m["cut_at"][rng.randrange(len(self.m["cut_at"]))]
            order = reading_order(truth)
            last = order[max(0, min(n, int(round(at * n))) - 1)]["bbox"][1]
            out = reading_order([e for e in out if e["bbox"][1] <= last]) if at > 0 else []
        return {"elements": out}

    def as_role(self, e: dict, role: str, rng: random.Random) -> dict:
        """The element written as `role` (the same role for most), with the fields that role has."""
        t_role = e["role"]
        e = dict(e, role=role)
        if t_role == "icon" and role == "icon":
            e["name"] = self.rename(e.get("name"), rng)
            if e["name"] is None:
                e.pop("name")
        if role != t_role:
            text, name = e.get("text"), e.get("name")
            for k in ("on", "filled", "empty", "name", "text"):
                e.pop(k, None)
            if role in ("text", "field"):
                if t_role == "icon":
                    table = self.m["icon_text"].get(name or "")
                    if not table or sum(table.values()) < 3:
                        table = self.icon_text_all
                    text = draw(table, rng, "")
                e["text"] = text or ""
            if role == "switch":
                e["on"] = False
            if role == "mark":
                e["filled"] = True
        return e

    def op_for(self, written: str, f: tuple, rng: random.Random) -> dict | None:
        """An observed way the reader wrote an extra from a true text like this one (role and text length, then coarser)."""
        for key in ((written, f[0], f[1]), (written, f[0]), (written,)):
            idx = self.op_index.get(key) or []
            if len(idx) >= MIN_COUNT or (key == (written,) and idx):
                return self.m["ops"][idx[rng.randrange(len(idx))]]
        return None

    def spawned(self, t: dict, op: dict, rng: random.Random) -> dict:
        """The extra element made from true text t by op (related_source()'s kinds), at t's box plus op's offset."""
        w = text_of(t)
        if op["kind"] == "same":
            text = w
        elif op["kind"] == "part":
            n = max(1, min(len(w), round(op["length"] * len(w))))
            start = max(0, min(len(w) - n, round(op["start"] * len(w))))
            text = w[start:start + n]
        elif op["kind"] == "more":
            text = op["before"] + w + op["after"]
        else:
            text = self.misread_text(w, rng)
        return {**op["x"], "text": text, "bbox": clip_box([a + b for a, b in zip(t["bbox"], op["d"])])}

    def move(self, t: dict, f: tuple, rng: random.Random) -> list[int]:
        offs = self.offsets.get((t["role"], f[3])) or []
        if len(offs) < MIN_COUNT:
            offs = self.offsets.get((t["role"],)) or offs
        if not offs:
            return list(t["bbox"])
        d = offs[rng.randrange(len(offs))]
        return clip_box([a + b for a, b in zip(t["bbox"], d)])


# ---------------------------------------------------------------- generic noise

class Generic:
    """Generic noise, fixed in advance and the same for every screen; the rates are drawn once per episode. The seed
    salt "A1" is kept as the paper's runs had it."""
    RANGES = {"miss": (0.0, 0.20), "char": (0.0, 0.05), "jitter": (0.0, 15.0), "flip": (0.0, 0.10), "cut": (0.0, 0.10)}
    digest = hashlib.sha256(json.dumps(RANGES, sort_keys=True).encode()).hexdigest()[:16]

    def episode_rates(self, episode: str) -> dict:
        rng = random.Random(hashlib.sha256(f"A1|{episode}".encode()).hexdigest())
        return {k: rng.uniform(*r) for k, r in self.RANGES.items()}

    def apply(self, obs: dict, episode: str, step: int) -> dict:
        r = self.episode_rates(episode)
        rng = random.Random(hashlib.sha256(f"A1|{episode}|{step}".encode()).hexdigest())
        pool = [c for e in obs["elements"] for c in text_of(e)]
        out = []
        for e in obs["elements"]:
            if rng.random() < r["miss"]:
                continue
            e = dict(e)
            if text_of(e):
                e["text"] = "".join(self.sub(c, rng, pool) if rng.random() < r["char"] else c for c in text_of(e))
            j = r["jitter"]
            e["bbox"] = clip_box([v + rng.uniform(-j, j) for v in e["bbox"]])
            if e["role"] == "switch" and rng.random() < r["flip"]:
                e["on"] = not e["on"]
            if e["role"] == "mark" and rng.random() < r["flip"]:
                e["filled"] = not e["filled"]
            out.append(e)
        if out and rng.random() < r["cut"]:
            out = out[:rng.randrange(1, len(out) + 1)]
        return {"elements": [{**e, "id": i + 1} for i, e in enumerate(out)]}

    @staticmethod
    def sub(c: str, rng: random.Random, pool: list[str]) -> str:
        cls = char_class(c)
        if cls == "digit":
            return rng.choice([d for d in "0123456789" if d != c])
        if cls == "latin":
            return rng.choice([d for d in "abcdefghijklmnopqrstuvwxyz" if d != c.lower()])
        same = [d for d in pool if char_class(d) == cls and d != c]
        return rng.choice(same) if same else c


# ---------------------------------------------------------------- the fit check and the command line

def load_model(path: str | Path) -> Calibrated:
    """A fitted rate-matched noise model (python -m practice.noise fit)."""
    return Calibrated(json.loads(Path(path).read_text(encoding="utf-8")))


def held_out(episode: str) -> bool:
    return int(hashlib.sha256(f"holdout|{episode}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < HOLDOUT


def fit_check(screens: list[dict]) -> dict:
    """Fit on 80% of the episodes, compare on the other 20% (the same screens through the reader and through the model)."""
    train = [s for s in screens if not held_out(s["episode"])]
    test = [s for s in screens if held_out(s["episode"])]
    model = Calibrated(fit(train))
    real = rates([(s["truth"], s["read"]) for s in test])
    sim = rates([(s["truth"], model.apply({"elements": s["truth"]}, f'{s["episode"]}|{s["step"]}', app=app_of(s.get("id", "")))["elements"])
                 for s in test])
    diff = {k: round(100 * (sim[k] - real[k]), 1) for k in real if k[0].isdigit() and isinstance(real[k], float) and isinstance(sim[k], float)
            and not k.startswith(("6", "7"))}
    main = {k: diff.get(k) for k in ("1_miss", "2_shape", "5_state_wrong")}
    passed = all(v is not None and abs(v) <= 5 for v in main.values())
    return {"train_screens": len(train), "test_screens": len(test), "test_episodes": len({s["episode"] for s in test}),
            "reader": real, "model": sim, "difference_points": diff, "main_classes": main, "passed": passed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fit", "check"])
    ap.add_argument("out")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--episodes", help="a JSON file listing episodes, such as calibration_set.json")
    ap.add_argument("--key", default="calibration", help="the list in --episodes to use")
    a = ap.parse_args()
    only = episodes_from(a.episodes, a.key) if a.episodes else None
    screens = screens_of([Path(r) for r in a.runs], only)
    if not screens:
        raise SystemExit("no read screens in these runs")
    result = fit(screens) if a.what == "fit" else fit_check(screens)
    if a.what == "fit":
        result["runs"] = [Path(r).name for r in a.runs]
        result["episodes"] = len({s["episode"] for s in screens})
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=None if a.what == "fit" else 1), encoding="utf-8")
    if a.what == "check":
        print(json.dumps({k: result[k] for k in ("train_screens", "test_screens", "difference_points", "main_classes", "passed")}, ensure_ascii=False))
    else:
        print(json.dumps({"screens": result["screens"], "episodes": result["episodes"], "digest": Calibrated(result).digest}))


if __name__ == "__main__":
    main()
