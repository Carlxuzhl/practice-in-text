"""Small made-up screens for the tests (no MobileGym content)."""
from __future__ import annotations

import json
from pathlib import Path


def el(i: int, role: str, box, **kw) -> dict:
    return {"id": i, "role": role, "bbox": list(box), **kw}


def screen_a() -> list[dict]:
    """A settings-like screen: a status bar, a title, rows with switches and a coloured button."""
    return [
        el(1, "text", (40, 5, 120, 30), text="10:00", color="black"),
        el(2, "icon", (20, 60, 70, 100), name="chevron-left", color="black"),
        el(3, "text", (400, 60, 600, 100), text="Settings", color="black", bold=True),
        el(4, "text", (40, 150, 300, 190), text="Dark mode", color="black"),
        el(5, "switch", (820, 150, 940, 190), on=False, color="lightgray"),
        el(6, "text", (40, 220, 300, 260), text="Notifications", color="black"),
        el(7, "switch", (820, 220, 940, 260), on=True, color="green"),
        el(8, "field", (40, 300, 960, 350), text="Search", empty=True),
        el(9, "mark", (40, 400, 80, 440), filled=True, color="blue", round=True),
        el(10, "text", (100, 400, 400, 440), text="Remember me", color="gray"),
        el(11, "text", (300, 900, 700, 960), text="Submit", color="white", bg="blue"),
        el(12, "icon", (850, 60, 950, 100), color="black", raster=True),
    ]


def write_run(root: Path, name: str, rows: list[dict], steps: dict[str, list[dict]] | None = None, ledger: list[dict] | None = None,
              config: dict | None = None) -> Path:
    run = root / name
    (run / "steps").mkdir(parents=True)
    (run / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    for stem, recs in (steps or {}).items():
        (run / "steps" / f"{stem}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    (run / "vision_ledger.jsonl").write_text("".join(json.dumps(x) + "\n" for x in (ledger or [])), encoding="utf-8")
    (run / "meta.json").write_text(json.dumps({"config": config or {"policy_system": "system", "policy": {"samples": 1}}}), encoding="utf-8")
    return run
