"""Paths and settings shared by every part of the code.

Paths come from environment variables (see configs/paths.env.example):
  MOBILEGYM_DIR  the MobileGym checkout; its bench_env package is imported from there
  WORK_DIR       everything this code writes: runs/, data/, adapters/, merged reader models
  MODELS_DIR     the downloaded base models, one directory per model (configs/models.json; used by the scripts)
  SIM_URL        the simulator's address
  READER_URL, POLICY_URL  the two OpenAI-compatible chat endpoints (vLLM)
Fixed settings come from the JSON files in configs/."""
from __future__ import annotations

import functools
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"


def _dir(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set: copy configs/paths.env.example to configs/paths.env, edit it and source it")
    return Path(value).expanduser()


def mobilegym_dir() -> Path:
    return _dir("MOBILEGYM_DIR")


def work_dir() -> Path:
    return _dir("WORK_DIR")


def runs_dir() -> Path:
    return work_dir() / "runs"


def sim_url() -> str:
    return os.environ.get("SIM_URL", "http://127.0.0.1:14173")


def reader_url() -> str:
    return os.environ.get("READER_URL", "http://127.0.0.1:18200/v1/chat/completions")


def policy_url() -> str:
    return os.environ.get("POLICY_URL", "http://127.0.0.1:18100/v1/chat/completions")


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict:
    """configs/<name>.json"""
    return json.loads((CONFIGS / f"{name}.json").read_text(encoding="utf-8"))


def instances(name: str) -> dict:
    """configs/instances/<name>.json: {"templates": [...], "seeds": [...]}"""
    return json.loads((CONFIGS / "instances" / f"{name}.json").read_text(encoding="utf-8"))
