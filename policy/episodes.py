"""Run episodes: a policy (or a program oracle) acts on MobileGym instances through one observation channel.
  text         the true text view of the page (the practice environment)
  screenshots  the reader's parse of a real screenshot (the real pipeline; every reader call goes into vision_ledger.jsonl)
The executor, step budget, judge and action space are shared, and tap targets come from whichever observation the
policy saw, so a reader box that is off misses for real.

Usage: python -m policy.episodes <text|screenshots> --instances FILE [--samples K] [--seed-base N] [--workers 12]
           [--policy-model NAME] [--noise generic|MODEL.json] [--oracle] [--png] [--sort-reader-output]
           [--purpose NAME] [--reader-tag NAME] [--tag SUFFIX] [--limit N] [--skip-done RUN] [--stub]
Policy: notes of up to 100 characters with any action, the last 10 shown at every step; sampling at temperature 0.6,
top-p 0.95, top-k 20, with a seed fixed per episode from the template, the instance seed and --seed-base + k, where k is
the sample index (--samples K runs every instance K times).
Reader (screenshots): the settings of configs/reader.json (half-size screenshots, greedy, repetition penalty 1.05); each
read is booked under --purpose. --png keeps the PNG each read saw and the true page taken right after it (needed for
the calibration set and for reader alignment); otherwise a JPEG of each step is kept.
--sort-reader-output puts the reader's parsed list in the layout order before the policy sees it (ids renumbered); taps
still go to the reader's own boxes, and the raw output is recorded as it came.
--noise (text only): generic noise, or rate-matched noise from a fitted model (practice/noise.py). Within an episode an
unchanged screen gives the same list under rate-matched noise. Taps go to the centre of the box the policy saw, so a
moved box misses for real; each step also keeps the true list.
--oracle (text only) has the program oracle of each template take the policy's place to record demonstrations: it
decides from the true page, each action it means to take is put in the shared action space and carried out by the same
executor, the prompt it is recorded with is the one the policy would get, and what it read for later becomes the note
of its next action. An action the action space cannot express ends the episode as a failure. No model is called.
--skip-done RUN finishes a run that was cut short: instances with a row in RUN are left out (an episode cut off mid-way
has no row, so it runs again from the start). --stub replaces both models with local stand-ins to test the plumbing
without a GPU."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from pathlib import Path

from common import sim  # first: puts MobileGym on the import path and fixes the time zone
from common.config import load, policy_url, reader_url, sim_url
from common.runs import finish_run, new_run
from policy import model as pm
from policy import oracle_lib as ol
from policy.collect_sft import INVALID  # the history entry of an action that could not be carried out
from reader import reader as R
from textview import textview as tv
from bench_env.config import DIFFICULTY_MAX_STEPS
from bench_env.env.base import Action, ActionType
from bench_env.env.mobile_gym import MobileGymEnv
from bench_env.task.registry import TaskRegistry

PCFG, RCFG = load("policy"), load("reader")["call"]
STUCK_AFTER = PCFG["stuck_after"]  # the same action on an unchanged screen this many times in a row ends the episode
KEYS = {ActionType.BACK: "back", ActionType.HOME: "home", ActionType.ENTER: "enter"}
POLICY = {"seed_base": 0, "model": "policy", "samples": 1, "oracle": False}  # set from the command line
READER = {"scale": RCFG["scale"], "rep_penalty": RCFG["repetition_penalty"], "purpose": "eval", "tag": "reader", "png": False, "sort": False}
NOISE: dict = {"kind": "", "model": None, "path": "", "digest": ""}
ORACLES: dict = {}


def noisy(page_obs: dict, tid: str, seed: int, k: int, step: int) -> dict:
    """The corrupted list for this screen (--noise); the episode key is the one the sampling seed comes from."""
    episode = f"{tid}|{seed}|{POLICY['seed_base'] + k}"
    if NOISE["kind"] == "generic":
        return NOISE["model"].apply(page_obs, episode, step)
    return NOISE["model"].apply(page_obs, f"{episode}|{hashlib.sha256(tv.render(page_obs).encode()).hexdigest()[:16]}")


def same_target(action: Action | None, note: str, page_obs: dict) -> str:
    """What an action really did, judged on the page itself so that every channel is judged alike. The reader rewords
    a screen between reads, so its text cannot tell whether anything changed."""
    if action is None:
        return "invalid:" + note
    point = action.data.get("point")
    if not point:
        return f"{action.action_type.value}:{action.data}"
    hits = [e for e in page_obs["elements"] if e["bbox"][0] <= point[0] <= e["bbox"][2] and e["bbox"][1] <= point[1] <= e["bbox"][3]]
    if not hits:
        return f"{action.action_type.value}:empty"
    best = min(hits, key=lambda e: (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]))
    return f'{action.action_type.value}:{best["id"]}'


def step_budget(task) -> int:
    budget = DIFFICULTY_MAX_STEPS.get(task.difficulty, 30)
    return budget + 15 if getattr(task, "answer_fields", None) else budget


def to_action(decision: dict | None, obs: dict) -> tuple[Action | None, str]:
    """Map a decision onto the shared action space. Tap points come from the observation's own boxes."""
    if decision is None:
        return None, "unparsable"
    kind = decision.get("action")
    by_id = {e["id"]: e for e in obs["elements"]}
    target = None
    if kind in ("tap", "long_press", "type"):
        try:
            target = by_id.get(int(decision.get("id")))
        except (TypeError, ValueError):
            target = None
        if target is None:
            return None, "no such element"
    point = None
    if target:
        x1, y1, x2, y2 = target["bbox"]
        point = [(x1 + x2) // 2, (y1 + y2) // 2]
    if kind == "tap":
        return Action(ActionType.CLICK, {"point": point}), ""
    if kind == "long_press":
        return Action(ActionType.LONG_PRESS, {"point": point}), ""
    if kind == "type":
        return Action(ActionType.TYPE, {"value": str(decision.get("text", "")), "point": point,
                                        "clear": bool(decision.get("clear"))}), ""
    if kind == "swipe":
        try:
            a, b = [[max(0, min(1000, int(v))) for v in decision[k]][:2] for k in ("from", "to")]
        except (KeyError, TypeError, ValueError):
            return None, "bad swipe"
        if len(a) != 2 or len(b) != 2:
            return None, "bad swipe"
        return Action(ActionType.SWIPE, {"point1": a, "point2": b, "duration": 400}), ""
    if kind in ("back", "home", "enter"):
        return Action({"back": ActionType.BACK, "home": ActionType.HOME, "enter": ActionType.ENTER}[kind], {}), ""
    if kind == "done":
        return Action(ActionType.COMPLETE, {"return": "done"}), ""
    return None, "unknown action"


def describe(decision: dict | None, obs: dict, note: str) -> str:
    """The history entry of an action, as the policy sees it at later steps."""
    if note:
        return f"{INVALID}（{note}）"
    target = next((e for e in obs["elements"] if e["id"] == decision.get("id")), None) if "id" in decision else None
    label = (target.get("text") or target.get("name") or target["role"]) if target else ""
    extra = f' 输入“{decision.get("text")}”' if decision["action"] == "type" else ""  # 'typed "..."'
    swipe = f' {decision["from"]}→{decision["to"]}' if decision["action"] == "swipe" else ""
    return f'{decision["action"]} {label}{extra}{swipe}'.strip()


def body_of(obs: dict) -> str:
    """The screen without its status bar, whose clock changes on its own."""
    return tv.render({"elements": [e for e in obs["elements"] if e["bbox"][1] > 45]})


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def under(point, obs: dict, prefer: str | None = None) -> dict | None:
    """The smallest element of the observation whose box holds the point (of the preferred role, if there is one)."""
    hits = [e for e in obs["elements"] if e["bbox"][0] <= point[0] <= e["bbox"][2] and e["bbox"][1] <= point[1] <= e["bbox"][3]]
    hits = [e for e in hits if e["role"] == prefer] or hits
    return min(hits, key=lambda e: (e["bbox"][2] - e["bbox"][0]) * (e["bbox"][3] - e["bbox"][1]), default=None)


def same_in(e: dict, obs: dict) -> dict | None:
    """The element of the observation that is e: the same role, text and name, and nearly the same box."""
    same = [x for x in obs["elements"] if x["role"] == e["role"] and x.get("text") == e.get("text") and x.get("name") == e.get("name")]
    best = max(same, key=lambda x: iou(x["bbox"], e["bbox"]), default=None)
    return best if best is not None and iou(best["bbox"], e["bbox"]) >= 0.9 else None


def oracle_decision(kind: ActionType, data: dict, obs: dict, target: dict | None) -> tuple[dict | None, str]:
    """An oracle's action as the policy would write it on this observation; None when the action space cannot say it."""
    if kind in (ActionType.CLICK, ActionType.LONG_PRESS, ActionType.TYPE):
        point = data.get("point")
        e = (same_in(target, obs) if target else None) or (under(point, obs, "field" if kind == ActionType.TYPE else None) if point else None)
        fields = [x for x in obs["elements"] if x["role"] == "field"]
        if e is None and kind == ActionType.TYPE and not point and len(fields) == 1:  # typing into the focused field: the only one there
            e = fields[0]
        if e is None:
            return None, "no element there" if point else "typing with no field"
        if kind == ActionType.TYPE:
            d = {"action": "type", "id": e["id"], "text": str(data.get("value", ""))}
            if data.get("clear"):
                d["clear"] = True
            return d, ""
        return {"action": "tap" if kind == ActionType.CLICK else "long_press", "id": e["id"]}, ""
    if kind == ActionType.SWIPE:
        return {"action": "swipe", "from": [int(v) for v in data["point1"]][:2], "to": [int(v) for v in data["point2"]][:2]}, ""
    if kind in KEYS:
        return {"action": KEYS[kind]}, ""
    if kind == ActionType.COMPLETE:
        return {"action": "done"}, ""
    return None, f"{kind.value} is outside the action space"


class OracleBridge(ol.Driver):
    """A program oracle in the policy's place (--oracle). Each action it means to take goes to the episode loop, which
    carries it out; the oracle waits until then and reads the page again itself."""

    def __init__(self, env, task, init_obs):
        super().__init__(env, task, init_obs)
        self.todo: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.go: asyncio.Queue = asyncio.Queue(maxsize=1)
        self.found: dict | None = None  # the element the last find returned
        self.focus: dict | None = None  # the element of the last tap, for typing with no point
        self.pending: list[str] = []  # what it read for later since its last action

    async def find(self, nth: int = 0, wait: float = 3.0, **kw) -> dict:
        self.found = await super().find(nth=nth, wait=wait, **kw)
        return self.found

    async def reads(self, *els: dict) -> None:
        self.pending += [t for t in (e.get("text") or e.get("name") for e in els) if t]

    async def _step(self, kind: ActionType, data: dict, note: str = "") -> None:
        self.steps += 1
        self.trace.append({"step": self.steps, "action": kind.value, "data": data, "note": note})
        point = data.get("point")
        target = self.found if self.found is not None and point is not None and list(point) == self.centre(self.found) else None
        if kind == ActionType.TYPE and point is None and self.focus is not None and self.focus["role"] == "field":
            target = self.focus
        if kind == ActionType.CLICK:
            self.focus = target
        self.found = None
        await self.todo.put((kind, data, target))
        await self.go.get()

    async def finish(self, message: str = "done"):
        await self._step(ActionType.COMPLETE, {"return": message})


async def oracle_policy(bridge: OracleBridge, script: asyncio.Task, obs: dict, user: str) -> dict:
    """The oracle's next action as a policy answer: an empty thought and one line of JSON. What it read since its last
    action becomes the note, after the prefix "read:" in the apps' language; when its script has returned, the answer
    is done."""
    get = asyncio.ensure_future(bridge.todo.get())
    done, _ = await asyncio.wait({get, script}, timeout=600, return_when=asyncio.FIRST_COMPLETED)
    if get in done:
        kind, data, target = get.result()
        decision, why = oracle_decision(kind, data, obs, target)
    else:
        get.cancel()
        err = script.exception() if script.done() and not script.cancelled() else None
        if script.done() and err is None:
            decision, why = {"action": "done"}, ""
        else:
            decision, why = None, (f"oracle failed: {type(err).__name__}: {err}" if err else "oracle gave no action in 600 s")[:300]
    if decision is not None and bridge.pending:
        decision["note"] = ("读到：" + "；".join(bridge.pending))[:pm.NOTE_CHARS]
    bridge.pending = []
    content = "" if decision is None else "<think>\n\n</think>\n\n" + json.dumps(decision, ensure_ascii=False)
    return {"content": content, "decision": decision, "why": why, "usage": {}, "seconds": 0.0, "finish": "stop", "user": user}


def stub_reader(page_obs: dict) -> dict:
    text = re.sub(r"(?m)^\[\d+\] ", "", tv.render(page_obs))
    obs, bad = R.parse_reader(text)
    return {"content": text, "finish": "stop", "usage": {}, "seconds": 0.0, "obs": obs, "bad_lines": bad}


def stub_policy(step: int, obs: dict) -> dict:
    """Taps the first text it sees, then ends: enough to exercise every code path, notes included."""
    if step >= 3:
        d = {"action": "done"}
    else:
        hits = [e for e in obs["elements"] if e["role"] == "text" and e["bbox"][1] > 100]
        d = {"action": "tap", "id": hits[0]["id"]} if hits else {"action": "back"}
        d["note"] = f"stub step {step}: " + (hits[0].get("text") or "") * 30 if hits else "none"
    return {"content": json.dumps(d), "finish": "stop", "usage": {}, "seconds": 0.0, "decision": d}


def reorder(r: dict) -> dict:
    """--sort-reader-output: the reader's parsed list in the layout order, ids renumbered."""
    r["obs"] = {**r["obs"], "elements": tv.reading_order([dict(e) for e in r["obs"]["elements"]])}
    return r


async def episode(env, reg, tid: str, seed: int, cond: str, run_dir: Path, stub: bool, ledger, k: int = 0) -> dict:
    sim.assert_train(tid)
    task = reg.create_task(tid, _seed=seed)
    init_obs, params = await asyncio.wait_for(ol.boot(env, task), 90)
    budget = step_budget(task)
    row = {"id": tid, "seed": seed, "cond": cond, "instruction": task.description, "params": params, "budget": budget,
           "difficulty": task.difficulty, "suite": tid.split(".")[0], "sample": k}
    history: list[str] = []
    notes: list[str] = []
    episode_seed = int(hashlib.sha256(f"{tid}|{seed}|{POLICY['seed_base'] + k}".encode()).hexdigest()[:8], 16)
    truncated = 0
    trace: list[dict] = []
    last_obs, stop = init_obs, "MAX_STEPS"
    reader_s = policy_s = 0.0
    repeat, prev, last_body = 0, None, None
    name = f"{tid}_{seed}" if POLICY["samples"] == 1 else f"{tid}_{seed}_k{k}"
    steps_log = (run_dir / "steps" / f"{name}.jsonl").open("w", encoding="utf-8")
    bridge = OracleBridge(env, task, init_obs) if POLICY["oracle"] else None
    script = asyncio.ensure_future(ORACLES[tid](bridge)) if bridge else None
    t0 = time.time()
    n_steps = 0
    try:
        for step in range(1, budget + 1):
            n_steps = step
            await asyncio.sleep(0.25)
            page_obs = await tv.extract(env.page)
            body = body_of(page_obs)
            if history and body == last_body:
                history[-1] += pm.UNCHANGED
            last_body = body
            rec: dict = {"step": step}
            if cond == "screenshots":
                png = await env.page.screenshot(type="png")
                if READER["png"]:  # the screenshot as read, and the true page right after it
                    (run_dir / "shots" / f"{name}_{step}.png").write_bytes(png)
                    rec["v0_after_shot"] = (await tv.extract(env.page))["elements"]
                else:
                    await env.page.screenshot(path=str(run_dir / "shots" / f"{name}_{step}.jpg"), type="jpeg", quality=70)
                r = stub_reader(page_obs) if stub else await asyncio.to_thread(
                    R.read_screen, png, RCFG["timeout_s"], R.READER_PROMPT, READER["scale"], READER["rep_penalty"], RCFG["temperature"],
                    RCFG["seed"], RCFG["max_tokens"])
                if READER["sort"]:
                    r = reorder(r)
                reader_s += r["seconds"]
                obs = r["obs"]
                ledger.write(json.dumps({"purpose": READER["purpose"], "reader": READER["tag"], "scale": READER["scale"], "rep_penalty": READER["rep_penalty"],
                                         "t": time.time(), "id": tid, "seed": seed, "sample": k, "step": step, "seconds": r["seconds"],
                                         "usage": r["usage"], "finish": r["finish"], "raw": r["content"]}, ensure_ascii=False) + "\n")
                ledger.flush()
                rec.update(reader_raw=r["content"], bad_lines=r["bad_lines"], reader_finish=r["finish"], reader_usage=r["usage"],
                           reader_seconds=r["seconds"], v0=page_obs["elements"])
            elif NOISE["kind"]:
                obs = noisy(page_obs, tid, seed, k, step)
                rec["v0"] = page_obs["elements"]
            else:
                obs = page_obs
            screen = tv.render(obs)
            if bridge:
                d = await oracle_policy(bridge, script, obs, pm.user_message(task.description, screen, history, step, budget, notes))
            else:
                d = stub_policy(step, obs) if stub else await asyncio.to_thread(pm.decide, task.description, screen, history, step, budget,
                                                                                notes, episode_seed, POLICY["model"])
            policy_s += d["seconds"]
            truncated += d.get("finish") == "length"
            note = (d["decision"] or {}).get("note")
            if isinstance(note, str) and note.strip():
                notes.append(f"{pm.step_label(step)}：{note.strip()[:pm.NOTE_CHARS]}")
            action, note = to_action(d["decision"], obs)
            summary = describe(d["decision"], obs, note)
            history.append(f"{pm.step_label(step)}：{summary}")
            rec.update(screen=screen, policy_raw=d["content"], policy_usage=d["usage"], policy_seconds=d["seconds"], policy_finish=d.get("finish"),
                       policy_user=d.get("user"), note=notes[-1] if notes and notes[-1].startswith(f"{pm.step_label(step)}：") else None,
                       summary=summary, action=None if action is None else {"type": action.action_type.value, "data": action.data})
            steps_log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            steps_log.flush()
            trace.append({"step": step, "action": rec["action"], "note": note})
            if bridge and action is None:  # the oracle failed, or meant an action the action space cannot express
                stop = "ORACLE_FAILED"
                row["oracle_failure"] = d.get("why") or note
                break
            fingerprint = (same_target(action, note, page_obs), body)
            repeat = repeat + 1 if fingerprint == prev else 1
            prev = fingerprint
            if repeat >= STUCK_AFTER:
                stop = "STUCK"
                break
            if action is None:
                continue
            result = await asyncio.wait_for(env.step(action), 60)
            last_obs = result.observation
            if bridge:
                bridge.go.put_nowait(True)
            if result.done:
                stop = ActionType.COMPLETE
                break
    finally:
        steps_log.close()
        if script and not script.done():
            script.cancel()
        if script:
            await asyncio.gather(script, return_exceptions=True)
    verdict = await ol.judge(env, task, init_obs, last_obs, n_steps, trace, stop)
    row.update(stop=str(getattr(stop, "value", stop)), steps=n_steps, state_passed=verdict.passed,
               official_success=bool(verdict.passed and stop == ActionType.COMPLETE), progress=verdict.progress,
               issues=[str(i)[:240] for i in verdict.issues][:6], judge_error=verdict.judge_error,
               runtime_s=round(time.time() - t0, 1), reader_s=round(reader_s, 1), policy_s=round(policy_s, 1), truncated_steps=truncated,
               notes_written=len(notes), episode_seed=episode_seed)
    return row


async def worker(idx: int, queue: asyncio.Queue, out, cond: str, run_dir: Path, stub: bool, ledger) -> None:
    b = PCFG["browser"]
    env = MobileGymEnv(url=sim_url(), headless=True, verbose=False, delay_after_action=b["delay_after_action"],
                       viewport_size=tuple(b["viewport"]), physical_size=tuple(b["physical_size"]), device_scale_factor=b["device_scale_factor"])
    await env.start()
    reg = TaskRegistry()
    while True:
        try:
            tid, seed, k = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        try:
            row = await episode(env, reg, tid, seed, cond, run_dir, stub, ledger, k)
        except Exception as e:  # recorded as a failed episode and never dropped
            row = {"id": tid, "seed": seed, "sample": k, "cond": cond, "suite": tid.split(".")[0], "official_success": False, "state_passed": False,
                   "error": f"{type(e).__name__}: {e}"[:400]}
            try:
                await env.restart()
            except Exception as e2:
                print("restart failed", e2, flush=True)
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        out.flush()
        print(idx, tid, seed, row.get("stop"), row.get("steps"), "PASS" if row["official_success"] else "fail", row.get("error", "")[:120], flush=True)
    await env.close()


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cond", choices=["text", "screenshots"])
    ap.add_argument("--instances", required=True, help='JSON file {"templates": [...], "seeds": [...]}')
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-done", default="")
    ap.add_argument("--seed-base", type=int, default=0)
    ap.add_argument("--policy-model", default=os.environ.get("POLICY_MODEL", "policy"),
                    help="the served model name: a LoRA adapter's name, or policy for the base (default: $POLICY_MODEL)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--purpose", default="eval", help="what the reads are booked as: calibration, train, eval, ...")
    ap.add_argument("--reader-tag", default="reader", help="a name for the reader, recorded with every read")
    ap.add_argument("--png", action="store_true", help="keep the PNG each read saw and the true page right after it")
    ap.add_argument("--sort-reader-output", action="store_true", help="put the reader's list in the layout order")
    ap.add_argument("--oracle", action="store_true", help="record the program oracles' demonstrations (text only)")
    ap.add_argument("--noise", default="", help="generic, or a fitted rate-matched noise model .json (text only)")
    a = ap.parse_args()
    POLICY.update(seed_base=a.seed_base, model=a.policy_model, samples=a.samples, oracle=a.oracle)
    READER.update(purpose=a.purpose, tag=a.reader_tag, png=a.png, sort=a.sort_reader_output)
    if a.noise:
        from practice import noise as N
        if a.cond != "text" or a.oracle:
            raise SystemExit("--noise takes the text condition and a policy")
        if a.noise == "generic":
            NOISE.update(kind="generic", model=N.Generic(), path="", digest=N.Generic.digest)
        else:
            m = N.load_model(a.noise)
            NOISE.update(kind="rate_matched", model=m, path=str(Path(a.noise).resolve()), digest=m.digest)
    spec = json.loads(Path(a.instances).read_text(encoding="utf-8"))
    if a.oracle:
        from policy.oracles import ORACLES as ALL
        ORACLES.update(ALL)
        missing = sorted(set(spec["templates"]) - set(ORACLES))
        if a.cond != "text" or a.stub or a.samples != 1 or missing:
            raise SystemExit(f"--oracle takes the text condition, no --stub, one sample and templates with an oracle; missing {missing}")
    if a.sort_reader_output and a.cond != "screenshots":
        raise SystemExit("--sort-reader-output takes the screenshots condition")
    items = [(t, s, k) for t in spec["templates"] for s in spec["seeds"] for k in range(a.samples)]
    if a.skip_done:
        done = {(r["id"], r["seed"], r.get("sample", 0)) for r in map(json.loads, (Path(a.skip_done) / "rows.jsonl").open(encoding="utf-8"))}
        items = [i for i in items if i not in done]
    if a.limit:
        items = items[:: max(len(items) // a.limit, 1)][:a.limit]
    run_dir = new_run(f"{a.cond}{a.tag}", {"cond": a.cond, "instances": a.instances, "episodes": len(items), "workers": a.workers,
                                           "stub": a.stub, "stuck_after": STUCK_AFTER, "reader_prompt": R.READER_PROMPT,
                                           "policy_system": pm.POLICY_SYSTEM_NOTES, "continues": a.skip_done, "policy": dict(POLICY),
                                           "sampling": pm.SAMPLING, "policy_url": policy_url(), "think": pm.THINK, "reader": dict(READER),
                                           "reader_url": reader_url(), "parser": [R.PARSER_VERSION, R.parser_digest()],
                                           "noise": {k: NOISE[k] for k in ("kind", "path", "digest")}, "sim_at": sim.SIM_AT.isoformat(),
                                           "sim_url": sim_url()})
    (run_dir / "steps").mkdir()
    (run_dir / "shots").mkdir()
    queue: asyncio.Queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)
    with open(run_dir / "rows.jsonl", "w", encoding="utf-8") as out, open(run_dir / "vision_ledger.jsonl", "w", encoding="utf-8") as ledger:
        await asyncio.gather(*[worker(i, queue, out, a.cond, run_dir, a.stub, ledger) for i in range(a.workers)])
    rows = [json.loads(line) for line in (run_dir / "rows.jsonl").open(encoding="utf-8")]
    passed = sum(r["official_success"] for r in rows)
    finish_run(run_dir, {"episodes": len(rows), "passed": passed, "errors": sum("error" in r for r in rows)})
    print("RUN_DIR", run_dir, "passed", passed, "/", len(rows), "errors", sum("error" in r for r in rows), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
