"""Print the text view of a MobileGym task instance at its first screen (the simulator must be served).
With --check, also check on the live page that extraction changes neither the DOM nor hit-testing, and that strings
injected off screen, under cover or in attributes do not come out. The DOM comparison needs a page that does not change
on its own (an analog clock's second hand moves, for example): a control snapshot taken without extraction decides,
and on a changing page that comparison is skipped and said so.
Usage: python -m textview.dump --task clock.DeleteAlarm [--seed 0] [--check]"""
from __future__ import annotations

import argparse
import asyncio
import json
import re

from common import sim  # first: MobileGym on the import path, time zone fixed
from common.config import load, sim_url
from policy import oracle_lib as ol
from textview import textview as tv
from bench_env.env.mobile_gym import MobileGymEnv
from bench_env.task.registry import TaskRegistry

HIT_GRID = """() => { const out = []; for (let y = 5; y < innerHeight; y += 20) for (let x = 5; x < innerWidth; x += 20) {
  const e = document.elementFromPoint(x, y); out.push(e ? e.tagName + ':' + (e.getAttribute('class') || '') : ''); } return out; }"""

INJECT = """() => {
  const host = document.createElement('div');
  host.id = 'CANARY_LIVE_ID';
  host.setAttribute('aria-label', 'CANARY_LIVE_ARIA');
  host.setAttribute('data-trigger', 'CANARY_LIVE_TRIGGER');
  host.title = 'CANARY_LIVE_TITLE';
  host.style.cssText = 'position:fixed;left:20px;top:300px;z-index:999999;background:#fff;color:#000;font-size:16px';
  host.textContent = 'LIVE_VISIBLE';
  const hid = document.createElement('div');
  hid.style.cssText = 'position:fixed;left:-500px;top:320px;z-index:999999';
  hid.textContent = 'CANARY_LIVE_OFFSCREEN';
  const under = document.createElement('div');
  under.style.cssText = 'position:fixed;left:30px;top:310px;z-index:999998';
  under.textContent = 'CANARY_LIVE_COVER';
  document.body.append(host, hid, under);
  return true;
}"""


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    sim.assert_train(a.task)
    b = load("policy")["browser"]
    env = MobileGymEnv(url=sim_url(), headless=True, verbose=False, delay_after_action=b["delay_after_action"],
                       viewport_size=tuple(b["viewport"]), physical_size=tuple(b["physical_size"]), device_scale_factor=b["device_scale_factor"])
    await env.start()
    ok = True
    try:
        task = TaskRegistry().create_task(a.task, _seed=a.seed)
        await ol.boot(env, task)
        await asyncio.sleep(1.0)
        if a.check:
            html_a = await env.page.evaluate("document.documentElement.outerHTML")
            grid_a = await env.page.evaluate(HIT_GRID)
            await asyncio.sleep(0.5)
            html_ctl = await env.page.evaluate("document.documentElement.outerHTML")
        obs = await tv.extract(env.page)
        print(task.description)
        print(tv.render(obs))
        if a.check:
            await tv.extract(env.page)
            static = html_a == html_ctl
            html_b = await env.page.evaluate("document.documentElement.outerHTML")
            checks = {"dom_unchanged": (html_ctl == html_b) if static else "skipped: the page changes on its own",
                      "hit_test_unchanged": grid_a == await env.page.evaluate(HIT_GRID)}
            await env.page.evaluate(INJECT)
            blob = json.dumps(await tv.extract(env.page), ensure_ascii=False)
            checks["no_canary_leak"] = not re.findall(r"CANARY_[A-Z0-9_]+", blob)
            checks["visible_kept"] = "LIVE_VISIBLE" in blob
            ok = all(v is True or isinstance(v, str) for v in checks.values())
            print(json.dumps(checks), "LIVE_CHECK_PASSED" if ok else "LIVE_CHECK_FAILED")
    finally:
        await env.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
