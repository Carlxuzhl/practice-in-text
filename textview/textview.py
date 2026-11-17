"""The text view: run the visible-element extractor (extract.js) in the page and render its list for a policy.

Each element has a role (text, icon, image, field, switch, mark), its text or icon name, its state, its colour and its
box on a 0-1000 grid. Lines run from top to bottom and, within a row, from left to right (the layout order).

Icon names: lucide icons are named from the library's own geometry (lucide_icon_table.json, generated at setup by
build_icon_table.mjs from MobileGym's lucide-react package); the remaining icons are named by eye in
custom_icons.json, keyed by a hash of their shape. The shape itself never reaches the output."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
JS = (HERE / "extract.js").read_text(encoding="utf-8")
ICON_TABLE = HERE / "lucide_icon_table.json"
COMPASS = ["up", "right", "down", "left"]
_table: dict[str, str] | None = None
_custom: dict[str, str] | None = None


def icon_table() -> dict[str, str]:
    global _table
    if _table is None:
        if not ICON_TABLE.exists():
            raise SystemExit(f"{ICON_TABLE} is missing: run scripts/setup_sim.sh (it builds the table from MobileGym's lucide-react)")
        _table = json.loads(ICON_TABLE.read_text(encoding="utf-8"))
    return _table


def icon_name(shape: str) -> str | None:
    """The name of an icon as a person would read it, or None."""
    global _custom
    if _custom is None:
        _custom = json.loads((HERE / "custom_icons.json").read_text(encoding="utf-8"))
    return icon_table().get(shape) or _custom.get(hashlib.sha1(shape.encode()).hexdigest()[:16])


async def extract(page, audit: bool = False, keep_shapes: bool = False) -> dict:
    """The text view of a Playwright page: {"elements": [...], "viewport", "restored", "diag"}."""
    out = await page.evaluate(JS, {"collectShapes": True, "audit": audit})
    if not out.get("restored"):
        raise RuntimeError("extractor left its stylesheet on the page")
    for e in out["elements"]:
        shape, turn = e.pop("shape", None), e.pop("turn", 0)
        if shape is None:
            continue
        name = icon_name(shape)
        if name:
            e["name"] = turned(name, turn)
        elif keep_shapes:
            e["shape"] = shape
    return out


def turned(name: str, deg: int) -> str:
    """chevron-right turned 180 degrees reads as chevron-left. Names without a single direction stay."""
    parts = name.split("-")
    hits = [i for i, p in enumerate(parts) if p in COMPASS]
    if deg and len(hits) == 1:
        parts[hits[0]] = COMPASS[(COMPASS.index(parts[hits[0]]) + deg // 90) % 4]
    return "-".join(parts)


def render(obs: dict) -> str:
    """One line per element: [id] role name "text" state colour on-background bold dim empty (x1,y1,x2,y2)."""
    lines = []
    for e in obs["elements"]:
        role = e["role"]
        head = f'[{e["id"]}] {role}'
        if role == "icon" and e.get("name"):
            head += f' {e["name"]}'
        if e.get("text") is not None:
            head += f' {json.dumps(e["text"], ensure_ascii=False)}'
        if role == "switch":
            head += " on" if e["on"] else " off"
        if role == "mark":
            head += " filled" if e["filled"] else " empty"
        attrs = [e[k] for k in ("color",) if e.get(k)]
        if e.get("bg"):
            attrs.append(f'on-{e["bg"]}')
        attrs += [k for k in ("bold", "dim", "empty") if e.get(k) and role != "mark"]
        tail = f' ({",".join(map(str, e["bbox"]))})'
        lines.append(head + (" " + " ".join(attrs) if attrs else "") + tail)
    return "\n".join(lines)


def reading_order(elements: list[dict]) -> list[dict]:
    """The extractor's layout order (rows of 6 px of an 800 px screen, then left to right), ids renumbered 1..n."""
    out = sorted(elements, key=lambda e: (round(e["bbox"][1] * 0.8 / 6), e["bbox"][0]))
    return [{**e, "id": i + 1} for i, e in enumerate(out)]
