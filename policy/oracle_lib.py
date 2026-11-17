"""Scripted oracle driver. It sees only the text view and acts only through env.step, in the same action space a policy
gets, so a pass shows the task is completable through the real UI. Also the episode start (boot) and the judge, which
every run shares."""
from __future__ import annotations

import asyncio
import dataclasses
import re
from typing import Any, Optional

from common.sim import SIM_AT  # noqa: F401  (also puts MobileGym on the import path)
from textview import textview as tv
from bench_env.env.base import Action, ActionType
from bench_env.runner.base import Controller, Evaluator, ExecutionResult


class OracleError(RuntimeError):
    pass


class Driver:
    def __init__(self, env, task, init_obs, log=None, settle: float = 0.25, corrupt: bool = False):
        self.env, self.task, self.init_obs = env, task, init_obs
        self.corrupt = corrupt  # negative control: same path, but typed text and chosen answers are wrong
        self.corrupted = 0  # how many values that control actually changed
        self.steps = 0
        self.trace: list[dict] = []
        self.log = log
        self.settle = settle
        self.obs_cache: Optional[dict] = None

    async def els(self) -> list[dict]:
        await asyncio.sleep(self.settle)
        self.obs_cache = await tv.extract(self.env.page)
        return self.obs_cache["elements"]

    async def show(self) -> str:
        await self.els()
        return tv.render(self.obs_cache)

    async def _step(self, kind: ActionType, data: dict, note: str = "") -> None:
        self.steps += 1
        self.trace.append({"step": self.steps, "action": kind.value, "data": data, "note": note})
        if self.log:
            self.log(f"  step {self.steps}: {kind.value} {data} {note}")
        await self.env.step(Action(kind, data))

    @staticmethod
    def centre(e: dict) -> list[int]:
        x1, y1, x2, y2 = e["bbox"]
        return [(x1 + x2) // 2, (y1 + y2) // 2]

    def _match(self, els, text=None, role=None, name=None, region=None, regex=None, color=None):
        out = []
        for e in els:
            if role and e["role"] != role:
                continue
            if name and e.get("name") != name:
                continue
            if color and e.get("color") != color:
                continue
            t = e.get("text") or ""
            if text is not None and t != text:
                continue
            if regex is not None and not re.search(regex, t):
                continue
            if region:
                cx, cy = self.centre(e)
                if not (region[0] <= cx <= region[2] and region[1] <= cy <= region[3]):
                    continue
            out.append(e)
        return out

    async def find(self, nth: int = 0, wait: float = 3.0, **kw) -> dict:
        deadline = asyncio.get_event_loop().time() + wait
        while True:
            hits = self._match(await self.els(), **kw)
            if len(hits) > nth:
                return hits[nth]
            if asyncio.get_event_loop().time() > deadline:
                raise OracleError(f"no element for {kw} (nth={nth}); screen:\n{tv.render(self.obs_cache)}")

    async def has(self, **kw) -> bool:
        return bool(self._match(await self.els(), **kw))

    async def reads(self, *els: dict) -> None:
        """Marks screen content an answer depends on. Nothing to do when the driver sees the page itself."""

    async def tap(self, nth: int = 0, **kw) -> dict:
        e = await self.find(nth=nth, **kw)
        await self._step(ActionType.CLICK, {"point": self.centre(e)}, f"on {e.get('text') or e.get('name') or e['role']}")
        return e

    async def tap_xy(self, x: int, y: int, note: str = "") -> None:
        await self._step(ActionType.CLICK, {"point": [x, y]}, note)

    async def long_press(self, nth: int = 0, **kw) -> dict:
        e = await self.find(nth=nth, **kw)
        await self._step(ActionType.LONG_PRESS, {"point": self.centre(e)}, f"on {e.get('text') or e['role']}")
        return e

    async def type(self, value: str, point=None, clear: bool = False) -> None:
        self.corrupted += self.corrupt
        data: dict[str, Any] = {"value": value + "9" if self.corrupt else value, "clear": clear}
        if point is not None:
            data["point"] = point
        await self._step(ActionType.TYPE, data)

    async def swipe(self, p1, p2, duration: int = 400, note: str = "") -> None:
        await self._step(ActionType.SWIPE, {"point1": list(p1), "point2": list(p2), "duration": duration}, note)

    async def drag(self, p1, p2, duration: int = 500) -> None:
        await self._step(ActionType.DRAG, {"point1": list(p1), "point2": list(p2), "duration": duration})

    async def scroll_down(self, frac: int = 450, x: int = 500) -> None:
        await self.swipe((x, 700), (x, 700 - frac))

    async def scroll_up(self, frac: int = 450, x: int = 500) -> None:
        await self.swipe((x, 300), (x, 300 + frac))

    async def back(self) -> None:
        await self._step(ActionType.BACK, {})

    async def home(self) -> None:
        await self._step(ActionType.HOME, {})

    async def enter(self) -> None:
        await self._step(ActionType.ENTER, {})

    async def scroll_to(self, limit: int = 8, frac: int = 450, **kw) -> dict:
        for _ in range(limit):
            hits = self._match(await self.els(), **kw)
            if hits:
                return hits[0]
            await self.scroll_down(frac)
        raise OracleError(f"not found after scrolling: {kw}\n{tv.render(self.obs_cache)}")

    async def finish(self, message: str = "done"):
        """End the episode as a policy would, then judge with the official evaluator."""
        result = await self.env.step(Action(ActionType.COMPLETE, {"return": message}))
        self.steps += 1
        return await judge(self.env, self.task, self.init_obs, result.observation, self.steps, self.trace,
                           ActionType.COMPLETE)


async def judge(env, task, init_obs, last_obs, steps, trace, stop_reason):
    state = await env.get_state(required_apps=list(task.apps) if task.apps else None)
    last = dataclasses.replace(last_obs, state=state)
    ex = ExecutionResult(steps=steps, trace=trace, runtime_s=0.0, finished=True, truncated=False,
                         stop_reason=stop_reason, agent_answer=env.agent_answer, agent_message=env.agent_message)
    return await Evaluator(judge_mode="state", eval_mode="grounded").evaluate(task, init_obs, last, ex)


async def boot(env, task):
    """Start the episode at the fixed simulated moment. The page clock is set first because setup reloads the page and
    seeds data from its time, and the judge later reads the time back from the page. A fresh env would skip that reload
    on its first episode, so its data would follow the real clock; the first episode therefore reloads too."""
    await env.page.clock.set_system_time(SIM_AT)
    env._fresh = False
    return await Controller.setup(env, task, eval_mode="grounded")
