"""MobileGym glue. Import this before anything from bench_env: it puts the MobileGym checkout on the import path and
fixes the time zone and the simulated moment every episode starts from."""
from __future__ import annotations

import datetime
import os
import sys
import time
import zoneinfo

from common.config import mobilegym_dir

# The apps parse date-only records as UTC, so west of UTC they would show them a day earlier than the judge reads them.
# The judge (Python) and the browser (a child process) both follow this one setting.
os.environ["TZ"] = "Asia/Shanghai"
time.tzset()
MG = mobilegym_dir()
if str(MG) not in sys.path:
    sys.path.insert(0, str(MG))

# Every episode starts from this simulated moment, a Tuesday in mid-month, so that runs made at different real times see
# the same screens. The judge reads its time back from the page.
SIM_AT = datetime.datetime(2026, 11, 10, 10, 0, tzinfo=zoneinfo.ZoneInfo("Asia/Shanghai"))


def read_split(name: str) -> list[str]:
    ids = []
    for raw in (MG / "bench_env" / "splits" / f"{name}.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            ids.append(line)
    return ids


def train_ids() -> list[str]:
    """MobileGym's training templates. Its official test templates are never run by this code."""
    train = read_split("train")
    overlap = set(read_split("test")).intersection(train)
    if overlap:
        raise RuntimeError(f"train/test overlap: {sorted(overlap)}")
    return train


def assert_train(task_id: str) -> None:
    from bench_env.splits import base_task_id

    if base_task_id(task_id) not in set(train_ids()):
        raise RuntimeError(f"{task_id} is not in MobileGym's train split; refusing to run it")
