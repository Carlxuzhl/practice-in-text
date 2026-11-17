"""Run directories and their provenance: every run gets WORK_DIR/runs/<time>_<purpose>/ with meta.json recording the
machine, the process, the code version, the environment and the configuration."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

from common.config import REPO, runs_dir


def _cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def code_digest() -> str:
    """A fingerprint of this repository's code and configs."""
    h = hashlib.sha256()
    for p in sorted(REPO.rglob("*")):
        if p.is_file() and p.suffix in {".py", ".sh", ".js", ".mjs", ".json"} and "__pycache__" not in p.parts and ".git" not in p.parts:
            h.update(str(p.relative_to(REPO)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def provenance(config: dict | None = None) -> dict:
    mg = os.environ.get("MOBILEGYM_DIR", "")
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "argv": sys.argv,
        "code_digest": code_digest(),
        "code_commit": _cmd("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "mobilegym_commit": _cmd("git", "-C", mg, "rev-parse", "HEAD") if mg else "",
        "mobilegym_dirty": _cmd("git", "-C", mg, "status", "--short") if mg else "",
        "mobilegym_dataset": "absent" if not mg or not (Path(mg) / "mobilegym-data").exists() else "PRESENT",
        "python": platform.python_version(),
        "node": _cmd("node", "--version"),
        "playwright": _version("playwright"),
        "tz": os.environ.get("TZ", ""),
        "config": config or {},
    }


def new_run(purpose: str, config: dict | None = None) -> Path:
    """Create WORK_DIR/runs/<time>_<purpose>/ with its meta.json."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir() / f"{stamp}_{purpose}"
    run_dir.mkdir(parents=True, exist_ok=False)
    meta = {"run_id": run_dir.name, "purpose": purpose, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **provenance(config)}
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def finish_run(run_dir: Path, summary: dict | None = None) -> None:
    p = run_dir / "meta.json"
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if summary is not None:
        meta["summary"] = summary
    p.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
