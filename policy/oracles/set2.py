"""Scripted oracles, part 2 (16 templates), under the same rules as part 1: each reads only the text view and the
task's own parameters (what the instruction says), and acts only through the driver. They were written from the text
view alone, before any reader ran on these templates. UI labels are matched in the apps' language (Chinese)."""
from __future__ import annotations

import asyncio
import re
from typing import Awaitable, Callable

from policy.oracle_lib import Driver, OracleError
from policy.oracles.set1 import open_launcher_app, wheel_to

ORACLES: dict[str, Callable[[Driver], Awaitable[None]]] = {}


def oracle(tid: str):
    def deco(fn):
        ORACLES[tid] = fn
        return fn
    return deco


BOTTOM = (0, 940, 1000, 1000)


def same_row(d: Driver, els: list[dict], anchor: dict, role: str, tol: int = 30) -> dict | None:
    y = d.centre(anchor)[1]
    hits = [e for e in els if e["role"] == role and abs(d.centre(e)[1] - y) < tol]
    return min(hits, key=lambda e: abs(d.centre(e)[1] - y)) if hits else None


async def set_switch(d: Driver, label: str, want: bool) -> None:
    """Make the switch on the row of label read want, and check that it does."""
    row = await d.find(text=label)
    sw = same_row(d, d.obs_cache["elements"], row, "switch")
    if sw is None:
        raise OracleError(f"no switch on the row of {label}")
    await d.reads(sw)
    if sw["on"] == want:
        return
    await d.tap_xy(*d.centre(sw), note=f"{label} -> {want}")
    for _ in range(8):
        await asyncio.sleep(0.3)
        row = await d.find(text=label)
        sw = same_row(d, d.obs_cache["elements"], row, "switch")
        if sw is not None and sw["on"] == want:
            return
    raise OracleError(f"{label} did not switch")


async def answer(d: Driver, value: str) -> None:
    """Open the answer sheet and give one answer: typed into its field, or picked from its options."""
    await open_launcher_app(d, "答题卡", page2=True)
    await d.find(text="提交答案", wait=5)
    if await d.has(role="field"):
        f = await d.find(role="field")
        await d.type(value, point=d.centre(f), clear=True)
        await d.back()  # close the on-screen keyboard
    else:
        await d.tap(text=value)
    await d.tap(text="提交答案")


async def launch(d: Driver, label: str) -> None:
    await d.home()
    if not await d.has(text=label):
        await d.swipe((850, 500), (100, 500))
    await d.tap(text=label)


async def open_wechat(d: Driver) -> None:
    await d.home()
    await d.tap(role="icon", name="wechat")
    await d.find(text="通讯录", region=BOTTOM, wait=5)


async def wechat_send(d: Driver, contact: str, message: str) -> None:
    await d.tap(text="通讯录", region=BOTTOM)
    await d.scroll_to(text=contact, frac=300)
    await d.tap(text=contact)
    await d.tap(text="发消息")
    f = await d.find(role="field")
    await d.type(message, point=d.centre(f))
    await d.tap(text="发送")
    await d.find(text=message, wait=4)


# ---------------------------------------------------------------- Alipay

ALIPAY_ME = (850, 960, 950, 995)


@oracle("alipay.ConfigureLanguageAndFastPay")
async def alipay_language_fastpay(d: Driver) -> None:
    await d.tap(text="我的", region=ALIPAY_ME)
    await d.tap(role="icon", name="settings", region=(850, 40, 1000, 120))
    await d.tap(text="支付设置")
    await d.tap(text="极速付款")
    if await d.has(text="开启"):  # the invitation page comes first while fast pay is off
        await d.tap(text="开启")
    row = await d.find(text="付款彩蛋", wait=5)
    state = [e for e in d.obs_cache["elements"] if e.get("text") in ("已开启", "已关闭") and abs(d.centre(e)[1] - d.centre(row)[1]) < 40]
    if not state:
        raise OracleError("no state on the 付款彩蛋 row")
    await d.reads(state[0])
    if state[0]["text"] == "已开启":
        await d.tap(text="付款彩蛋")
        await d.find(text="已关闭")
    for _ in range(3):  # back to the settings list
        if await d.has(text="通用"):
            break
        await d.back()
    # The language goes last: everything after it reads in English.
    await d.tap(text="通用")
    await d.tap(text="多语言")
    await d.tap(text="English")
    await d.tap(text="保存")
    await asyncio.sleep(0.8)


@oracle("alipay.SendMessageToContact")
async def alipay_send_message(d: Driver) -> None:
    p = d.task.p
    nick, _, rest = p.contact.partition("(")
    real = rest.rstrip(")")
    await d.tap(text="消息", region=(640, 960, 760, 995))
    await d.tap(role="icon", name="user-plus", region=(800, 50, 890, 100))
    await d.scroll_to(text=nick, frac=300)
    rows = [e for e in d.obs_cache["elements"] if e.get("text") == nick]
    if real:  # nicknames repeat; the real name sits on the line below
        named = [r for r in rows if any(e.get("text") == real and 0 < e["bbox"][1] - r["bbox"][1] < 45 for e in d.obs_cache["elements"])]
        rows = named or rows
    await d.tap_xy(*d.centre(rows[0]), note=f"contact {p.contact}")
    await d.tap(text="发消息", wait=4)
    # The input has no border and no placeholder, so it is the empty stretch between the bar's icons.
    mic = await d.find(role="icon", name="mic", region=(0, 930, 150, 1000), wait=4)
    smile = await d.find(role="icon", name="smile", region=(700, 930, 900, 1000))
    await d.type(p.text, point=[(mic["bbox"][2] + smile["bbox"][0]) // 2, d.centre(mic)[1]])
    await d.tap(text="发送")
    await d.find(text=p.text, wait=4)


# ---------------------------------------------------------------- Calendar

GRID = (0, 180, 1000, 530)


def grid_days(els: list[dict]) -> list[dict]:
    return [e for e in els if e["role"] == "text" and re.fullmatch(r"\d{1,2}", e.get("text") or "") and GRID[1] < e["bbox"][1] < GRID[3]]


# ---------------------------------------------------------------- Clock

@oracle("clock.DeleteAlarm")
async def clock_delete_alarm(d: Driver) -> None:
    t = d.task.p.time
    await d.scroll_to(text=t, frac=300)
    await d.long_press(text=t)
    await d.tap(text="删除", region=(400, 940, 600, 980))
    await asyncio.sleep(0.6)


@oracle("clock.LatestTimezoneCity")
async def clock_latest_city(d: Driver) -> None:
    await d.tap(text="世界时钟", region=(300, 950, 460, 975))
    cities: dict[str, tuple] = {}
    for _ in range(6):
        els = await d.els()
        before = len(cities)
        for c in els:
            if c["role"] != "text" or c.get("color") != "black" or not (60 <= c["bbox"][0] <= 120) or c["bbox"][1] < 330:
                continue
            clock = next((e for e in els if re.fullmatch(r"\d\d:\d\d", e.get("text") or "") and e["bbox"][0] > 700 and abs(d.centre(e)[1] - d.centre(c)[1]) < 20), None)
            date = next((e for e in els if re.match(r"\d+月\d+日", e.get("text") or "") and 0 < e["bbox"][1] - c["bbox"][1] < 45), None)
            if clock and date and c["text"] not in cities:
                mo, dy = (int(x) for x in re.match(r"(\d+)月(\d+)日", date["text"]).groups())
                hh, mm = (int(x) for x in clock["text"].split(":"))
                cities[c["text"]] = ((mo, dy, hh, mm), clock, date)
        if len(cities) == before and before:
            break
        await d.scroll_down(frac=250)
    if not cities:
        raise OracleError("no city rows")
    latest = max(cities, key=lambda k: cities[k][0])
    await d.reads(cities[latest][1], cities[latest][2])
    await answer(d, latest)


# ---------------------------------------------------------------- Cross-app

async def alipay_balance(d: Driver) -> str:
    await launch(d, "支付宝")
    await d.tap(text="我的", region=ALIPAY_ME)
    await d.tap(text="余额")
    v = await d.find(regex=r"^[\d,]+\.\d\d$", region=(100, 250, 900, 340), wait=5)
    await d.reads(v)
    return v["text"].replace(",", "")


@oracle("crossapp_commerce.AlipayMonthlySpendToWechat")
async def month_spend_to_wechat(d: Driver) -> None:
    await launch(d, "支付宝")
    await d.tap(text="我的", region=ALIPAY_ME)
    await d.tap(text="账单")
    label = await d.find(text="支出", region=(40, 240, 220, 280), wait=5)
    amount = [e for e in d.obs_cache["elements"] if re.fullmatch(r"¥[\d,]+\.\d\d", e.get("text") or "")
              and abs(e["bbox"][0] - label["bbox"][0]) < 30 and 0 < e["bbox"][1] - label["bbox"][1] < 40]
    if not amount:
        raise OracleError("no spending total")
    await d.reads(amount[0])
    await open_wechat(d)
    await wechat_send(d, d.task.p.contact, f"我这个月支付宝一共花了{amount[0]['text'][1:].replace(',', '')}元")


async def tomorrow_events(d: Driver) -> list[tuple[str, str, dict, dict]]:
    """Open the calendar on tomorrow and list its events as (title, time) from the top."""
    await launch(d, "日历")
    today = await d.find(regex=r"^\d{1,2}$", color="white", region=GRID, wait=5)
    await d.reads(today)
    nxt = int(today["text"]) + 1
    cell = [e for e in grid_days(await d.els()) if e["text"] == str(nxt) and not e.get("dim")]
    if not cell:  # the month ends today: tomorrow is the 1st of the next month
        await d.swipe((850, 350), (150, 350), note="next month")
        await asyncio.sleep(0.5)
        cell = [e for e in grid_days(await d.els()) if e["text"] == "1" and not e.get("dim")]
    await d.tap_xy(*d.centre(cell[0]), note="tomorrow")
    await asyncio.sleep(0.6)
    els = await d.els()
    times = [e for e in els if re.fullmatch(r"\d\d:\d\d(-\d\d:\d\d)?", e.get("text") or "") and e["bbox"][1] > 560]
    out = []
    for t in sorted(times, key=lambda e: e["bbox"][1]):
        title = [e for e in els if e["role"] == "text" and e.get("color") == "black" and abs(e["bbox"][0] - t["bbox"][0]) < 30 and 0 < t["bbox"][1] - e["bbox"][1] < 50]
        if title:
            out.append((title[-1]["text"], t["text"][:5], title[-1], t))
    if not out:
        raise OracleError("no events tomorrow")
    return out


@oracle("crossapp_life.CalendarEventToWechat")
async def calendar_event_to_wechat(d: Driver) -> None:
    title, time, e_title, e_time = (await tomorrow_events(d))[0]
    await d.reads(e_title, e_time)
    await open_wechat(d)
    await wechat_send(d, d.task.p.contact, f"{title} {time}")


@oracle("crossapp_work.CalendarEarliestToAlarm")
async def calendar_earliest_to_alarm(d: Driver) -> None:
    events = await tomorrow_events(d)
    title, time, e_title, e_time = min(events, key=lambda x: x[1])
    await d.reads(e_time)
    hh, mm = (int(x) for x in time.split(":"))
    total = (hh * 60 + mm - 30) % (24 * 60)
    await launch(d, "时钟")
    await d.tap(text="闹钟", region=(60, 950, 180, 975))
    await d.tap(role="icon", name="plus")
    await d.find(text="添加闹钟")
    await wheel_to(d, 352, total // 60, 24)
    await wheel_to(d, 647, total % 60, 60)
    await d.tap(role="icon", region=(840, 90, 950, 150))  # the tick at top right of the sheet
    await d.find(text="世界时钟")


@oracle("notes.ReadNotesCount")
async def notes_count(d: Driver) -> None:
    titles: set[str] = set()
    for _ in range(6):
        els = await d.els()
        cards = [e for e in els if e["role"] == "text" and e.get("bold") and e.get("color") == "black" and 330 < e["bbox"][1] < 900]
        new = {e["text"] for e in cards}
        if new <= titles:
            break
        await d.reads(*cards)
        titles |= new
        await d.scroll_down(frac=300)
    await answer(d, str(len(titles)))


@oracle("notes.TodoBatchWorkflow")
async def notes_todo_batch(d: Driver) -> None:
    p = d.task.p
    await d.tap(text="待办", region=(650, 940, 850, 990))
    await d.tap(role="icon", color="white", region=(760, 380, 890, 860))
    field = await d.find(regex="输入待办内容", role="field")
    await d.type(p.new_todo, point=d.centre(field))
    await d.enter()
    await d.find(text=p.new_todo)
    if await d.has(text="q"):  # the keyboard covers the lower list
        await d.back()
    row = await d.find(text=p.existing_todo)
    mark = same_row(d, d.obs_cache["elements"], row, "mark")
    if mark is None:
        raise OracleError("no mark on the row")
    await d.tap_xy(*d.centre(mark), note=f"complete {p.existing_todo}")
    await asyncio.sleep(0.8)
    els = await d.els()
    done = next((e for e in els if e.get("text") == "已完成"), None)
    limit = done["bbox"][1] if done else 930
    open_marks = [e for e in els if e["role"] == "mark" and not e.get("filled") and e["bbox"][0] < 220 and e["bbox"][1] < limit]
    await d.reads(*open_marks)
    await answer(d, str(len(open_marks)))


# ---------------------------------------------------------------- Tencent Meeting

@oracle("tencent_meeting.CheckPersonalRoomId")
async def meeting_room_id(d: Driver) -> None:
    await d.tap(text="小明", region=(150, 50, 300, 90))
    await d.tap(text="个人会议室")
    num = await d.find(regex=r"^\d{3} \d{3} \d{3,4}$", region=(250, 290, 700, 340), wait=5)
    await d.reads(num)
    await answer(d, num["text"])


# ---------------------------------------------------------------- WeChat

async def wechat_privacy(d: Driver) -> None:
    await d.tap(text="我", region=BOTTOM)
    await d.tap(text="设置")
    await d.tap(text="朋友权限")


@oracle("wechat.OpenBlacklist")
async def wechat_open_blacklist(d: Driver) -> None:
    await wechat_privacy(d)
    await d.tap(text="通讯录黑名单")
    await d.find(text="通讯录黑名单", region=(300, 40, 700, 100))


@oracle("wechat.OpenNewFriends")
async def wechat_open_new_friends(d: Driver) -> None:
    await d.tap(text="通讯录", region=BOTTOM)
    await d.tap(text="新的朋友")
    await d.find(text="新的朋友", region=(300, 40, 700, 100))


@oracle("wechat.SetAddMeSearch")
async def wechat_add_me_search(d: Driver) -> None:
    await wechat_privacy(d)
    await d.tap(text="添加我的方式")
    await set_switch(d, "微信号", True)
    await set_switch(d, "手机号", False)


@oracle("wechat.SetSignature")
async def wechat_signature(d: Driver) -> None:
    text = d.task.p.text
    await d.tap(text="我", region=BOTTOM)
    await d.tap(regex=r"^微信号")
    await d.tap(text="签名")
    f = await d.find(role="field")
    await d.type(text, point=d.centre(f), clear=True)
    await d.tap(text="完成")
    await d.find(text=text, wait=4)


# ---------------------------------------------------------------- WeChat Reading

@oracle("wechat_reading.CheckBookRating")
async def reading_book_rating(d: Driver) -> None:
    title = d.task.p.book_title
    await d.tap(role="icon", name="search", region=(60, 50, 140, 90))
    f = await d.find(role="field")
    await d.type(title, point=d.centre(f))
    await d.enter()
    head = await d.find(text=title, region=(200, 200, 800, 600), wait=5)
    labels = [e for e in d.obs_cache["elements"] if e.get("text") == "推荐值" and e["bbox"][1] > head["bbox"][1]]
    if not labels:
        raise OracleError("no 推荐值 under the book")
    label = min(labels, key=lambda e: e["bbox"][1])
    value = [e for e in d.obs_cache["elements"] if re.fullmatch(r"\d+(\.\d+)?%", e.get("text") or "")
             and abs(d.centre(e)[1] - d.centre(label)[1]) < 15 and e["bbox"][0] > label["bbox"][0]]
    if not value:
        raise OracleError("no value after 推荐值")
    await d.reads(value[0])
    await answer(d, value[0]["text"].rstrip("%"))  # the answer field takes the percentage without its sign


@oracle("wechat_reading.CheckCalendarMonthReading")
async def reading_month_days(d: Driver) -> None:
    p = d.task.p
    # The tab's label sits in the gesture strip at the very bottom, so tap its icon.
    await d.tap(role="icon", name="user", region=(820, 930, 930, 980))
    await d.tap(text="阅读时长", wait=5)
    await d.tap(text="月", region=(280, 100, 360, 145))
    target = (int(p.year), int(p.month))
    for _ in range(24):
        head = await d.find(regex=r"^\d{4}年\d{1,2}月$", region=(130, 160, 420, 210), wait=4)
        shown = tuple(int(x) for x in re.findall(r"\d+", head["text"]))
        if shown == target:
            await d.reads(head)
            break
        region = (420, 160, 500, 205) if target < shown else (505, 160, 580, 205)
        await d.tap(role="icon", region=region)
        await asyncio.sleep(0.5)
    else:
        raise OracleError(f"could not reach {target}")
    label = await d.find(text="阅读", region=(30, 370, 140, 415))
    n = [e for e in d.obs_cache["elements"] if re.fullmatch(r"\d+", e.get("text") or "") and abs(d.centre(e)[1] - d.centre(label)[1]) < 20
         and label["bbox"][2] <= e["bbox"][0] < label["bbox"][2] + 80]
    if not n:
        raise OracleError("no day count after 阅读")
    await d.reads(n[0])
    await answer(d, n[0]["text"])
