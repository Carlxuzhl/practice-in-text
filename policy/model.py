"""The policy: a language model (Qwen3-8B with a LoRA adapter, thinking on) that reads the task, its recent actions with
the notes it wrote, and the current element list, and answers with one action as one line of JSON.

The prompts are in Chinese, the language of the simulated apps, and must stay byte for byte as they are: every policy
was trained and evaluated with them. In English the system prompt says:
  You operate a phone to complete the user's task. At every step you see the list of elements on the current screen,
  then take exactly one action. Each line of the list is one element: [id] role content colour and other attributes
  (x1,y1,x2,y2), coordinates 0 to 1000. Roles: text, icon, image, field (input box), switch (on/off), mark (checkbox or
  small dot, filled/empty). Output only one line of JSON and nothing else. Actions:
    {"action":"tap","id":N}  tap an element            {"action":"long_press","id":N}  long-press an element
    {"action":"type","id":N,"text":"..."}  tap a field and type; add "clear":true to clear it first
    {"action":"swipe","from":[x,y],"to":[x,y]}  swipe; swiping up scrolls a list down
    {"action":"back"}  back    {"action":"home"}  to the home screen    {"action":"enter"}  press enter
    {"action":"done"}  the task is complete
  When the task asks a question, open the answer-sheet app, answer there and submit, then end with done. Swipe left on
  the home screen to see the second page of apps. Use done only once the task is really complete; steps are limited.
  An earlier action marked "screen unchanged" had no effect: you must take a different action.
  (Notes:) Any action may carry a note: add "note":"..." to the JSON, at most 100 characters, to keep what you will need
  later, such as numbers, names or times you looked up. Your last 10 notes are shown to you at every step.
The user message gives the task, "this is step k of at most n", the previous actions, the notes and the current
screen."""
from __future__ import annotations

import json
import re

from common.config import load, policy_url
from common.llm import chat

CFG = load("policy")
THINK = CFG["thinking"]
NOTE_CHARS, NOTES_KEPT = CFG["notes"]["max_chars"], CFG["notes"]["kept"]
SAMPLING = dict(CFG["sampling"])
HISTORY_KEPT = CFG["history_kept"]

POLICY_SYSTEM = """你在操作一部手机来完成用户的任务。每一步你会看到当前屏幕上的元素列表，然后只做一个动作。

元素列表每行一个元素：[编号] 类型 内容 颜色等属性 (x1,y1,x2,y2)，坐标范围 0 到 1000。
类型：text 文字，icon 图标，image 图片，field 输入框，switch 开关（on/off），mark 勾选框或小圆点（filled/empty）。

只输出一行 JSON，不要输出任何其他内容。可用动作：
{"action":"tap","id":编号}  点击元素
{"action":"long_press","id":编号}  长按元素
{"action":"type","id":编号,"text":"内容"}  点击输入框并输入；加 "clear":true 先清空
{"action":"swipe","from":[x,y],"to":[x,y]}  滑动，向上滑可让列表往下滚
{"action":"back"}  返回
{"action":"home"}  回到桌面
{"action":"enter"}  按回车
{"action":"done"}  任务已经完成

任务要求回答问题时，打开“答题卡”App，在里面作答并提交后再用 done 结束。桌面向左滑可以看到第二页的 App。
确认任务真正完成后才用 done；步数有限。之前的动作后面标着“屏幕没有变化”时，说明那个动作没有效果，必须换一个不同的动作。"""
POLICY_SYSTEM_NOTES = POLICY_SYSTEM + f"""
任何动作都可以带一条笔记：在 JSON 里加 "note":"笔记"，不超过 {NOTE_CHARS} 字，记下以后要用到的内容，例如查到的数字、名字、时间。你最近的 {NOTES_KEPT} 条笔记会在之后每一步给你。"""

# Pieces of the user message and of the action history (shown to the policy, so in the apps' language):
NONE = "（无）"  # "(none)"
UNCHANGED = "（屏幕没有变化）"  # "(screen unchanged)"


def step_label(step: int) -> str:
    return f"第 {step} 步"  # "step <n>"


def user_message(instruction: str, screen: str, history: list[str], step: int, budget: int, notes: list[str]) -> str:
    """The user message of one step: the task, the step count, the last actions, the notes and the current screen."""
    past = "\n".join(history[-HISTORY_KEPT:]) or NONE
    memo = f"你的笔记：\n{chr(10).join(notes[-NOTES_KEPT:]) or NONE}\n\n"  # "your notes:"
    # "task: ... / this is step k, at most n steps / previous actions: ... / current screen: ..."
    return f"任务：{instruction}\n\n这是第 {step} 步，最多 {budget} 步。\n\n之前的动作：\n{past}\n\n{memo}当前屏幕：\n{screen}"


def parse_decision(raw: str) -> dict | None:
    """The action of an answer: the JSON object after the thinking, or None."""
    m = re.search(r"\{.*\}", raw.split("</think>")[-1], re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    return d if isinstance(d, dict) and isinstance(d.get("action"), str) else None


def decide(instruction: str, screen: str, history: list[str], step: int, budget: int, notes: list[str], seed: int,
           model: str = "policy", timeout: float = CFG["timeout_s"]) -> dict:
    """One policy step, sampled with the episode's seed."""
    user = user_message(instruction, screen, history, step, budget, notes)
    max_tokens = CFG["max_tokens"]["thinking" if THINK else "no_thinking"]
    out = chat(policy_url(), {"model": model, **SAMPLING, "seed": seed, "max_tokens": max_tokens,
                              "chat_template_kwargs": {"enable_thinking": THINK}, "messages": [
        {"role": "system", "content": POLICY_SYSTEM_NOTES}, {"role": "user", "content": user}]}, timeout)
    out["decision"] = parse_decision(out["content"])
    out["user"] = user  # kept so that a successful step can be trained on exactly as the policy saw it
    return out
