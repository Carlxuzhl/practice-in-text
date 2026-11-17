"""Scripted oracles, part 1 (13 templates). Each one reads only the text view plus the task's own parameters (what the
instruction says), and acts only through the driver, in the policy's action space. Answers to query tasks are read
from the screen, never from simulator state. UI labels are matched in the apps' language (Chinese)."""
from __future__ import annotations

import asyncio
import re
from datetime import date
from typing import Awaitable, Callable

from policy.oracle_lib import Driver, OracleError
from textview import textview as tv

ORACLES: dict[str, Callable[[Driver], Awaitable[None]]] = {}


def oracle(tid: str):
    def deco(fn):
        ORACLES[tid] = fn
        return fn
    return deco


WHEEL_Y = 344


async def open_launcher_app(d: Driver, label: str, page2: bool = False) -> None:
    await d.home()
    if page2:
        await d.swipe((850, 500), (100, 500))
    await d.tap(text=label)


async def submit_answers(d: Driver, values: list[str]) -> None:
    await open_launcher_app(d, "答题卡", page2=True)
    for i, v in enumerate(values):
        f = await d.find(role="field", nth=i)
        await d.type(v, point=d.centre(f), clear=True)
    await d.back()  # close the on-screen keyboard
    await d.tap(text="提交答案")


async def submit_choice(d: Driver, option: str) -> None:
    await open_launcher_app(d, "答题卡", page2=True)
    if d.corrupt:
        submit = await d.find(text="提交答案")
        rows = [e for e in await d.els() if e["role"] == "text" and e.get("color") == "black" and e["bbox"][1] > 150
                and e["bbox"][1] < submit["bbox"][1] and e["text"] != option]
        option = rows[0]["text"]
        d.corrupted += 1
    await d.tap(text=option)
    await d.tap(text="提交答案")


# One swipe moves the clock wheel about one row per 20 screen units of travel. Short swipes are noisy
# because the wheel keeps its sub-row offset between gestures, so wheel_to reads the screen after
# each swipe and corrects the rest.
ROW_TRAVEL = 20


async def wheel_to(d: Driver, column_x: int, target: int, modulus: int) -> None:
    """Swipe a number wheel until the row at its centre (the black one) shows target."""
    for _ in range(16):
        els = await d.els()
        # The selected row is the black one. It can sit up to half a row off centre between gestures.
        row = [e for e in els if e["role"] == "text" and e.get("color") == "black" and re.fullmatch(r"\d\d", e["text"] or "")
               and abs(d.centre(e)[0] - column_x) < 80 and abs(d.centre(e)[1] - WHEEL_Y) < 60]
        if not row:
            raise OracleError(f"no wheel centre at x={column_x}")
        await d.reads(row[0])
        cur = int(row[0]["text"])
        if cur == target:
            return
        delta = (target - cur + modulus // 2) % modulus - modulus // 2
        dy = max(30, min(abs(delta) * ROW_TRAVEL, 200)) * (1 if delta > 0 else -1)
        y0 = 440 if dy > 0 else 250  # both starts lie inside the wheel, whose box ends at about 476
        await d.swipe((column_x, y0), (column_x, y0 - dy), note=f"wheel {cur}->{target}")
        await asyncio.sleep(0.5)
    raise OracleError("wheel did not converge")


async def world_clock_row(d: Driver, city: str) -> tuple[str, str]:
    """Open the world clock and read one city's time and offset line from the screen."""
    await d.tap(text="世界时钟")
    row = await d.scroll_to(text=city, frac=250)
    y = d.centre(row)[1]
    els = d.obs_cache["elements"]
    clock = [e for e in els if re.fullmatch(r"\d\d:\d\d", e.get("text") or "") and abs(d.centre(e)[1] - y) < 25 and e["bbox"][0] > 500]
    diff = [e for e in els if "｜" in (e.get("text") or "") and 0 < d.centre(e)[1] - y < 50]
    if not clock or not diff:
        raise OracleError(f"row for {city} incomplete\n{tv.render(d.obs_cache)}")
    await d.reads(clock[0], diff[0])
    return clock[0]["text"], diff[0]["text"].split("｜")[1]


@oracle("clock.AddWorldCity")
async def add_world_city(d: Driver) -> None:
    await d.tap(text="世界时钟")
    await d.tap(role="icon", name="plus")
    field = await d.find(role="field")
    await d.type(d.task.p.city, point=d.centre(field))
    await d.tap(text=d.task.p.city, region=(0, 230, 900, 600))
    await d.find(text=d.task.p.city, wait=4)


@oracle("clock.CityLocalTimeDiff")
async def city_local_time_diff(d: Driver) -> None:
    _, offset = await world_clock_row(d, d.task.p.city)
    await submit_answers(d, [offset])


@oracle("notes.PinNote")
async def notes_pin(d: Driver) -> None:
    title = d.task.p.note_title
    # Narrow cards cut long titles to their first characters.
    hit = rf"^{re.escape(title)}$|^{'|^'.join(re.escape(title[:n]) + '…$' for n in range(2, len(title)))}"
    await d.scroll_to(frac=300, regex=hit)
    await d.long_press(regex=hit)
    await d.tap(text="置顶", region=(0, 920, 1000, 980))
    await d.find(text="全部")


def sms_senders(d: Driver, els: list[dict], bottom: int = 1000) -> list[dict]:
    return [e for e in els if e["role"] == "text" and e.get("color") == "black" and 260 <= e["bbox"][0] <= 290
            and 280 < e["bbox"][1] < bottom]


@oracle("sms.DeleteConversation")
async def sms_delete(d: Driver) -> None:
    await d.scroll_to(text=d.task.p.sender, frac=300)
    await d.long_press(text=d.task.p.sender)
    await d.tap(text="删除")
    await d.find(text="短信")


@oracle("sms.ReadUnreadConversationCount")
async def sms_unread_count(d: Driver) -> None:
    unread: set[str] = set()
    seen: list[str] = []
    for _ in range(12):
        els = await d.els()
        rows = sms_senders(d, els)
        for row in rows:
            if row["text"] not in seen:
                seen.append(row["text"])
            top, bottom = row["bbox"][1] - 10, row["bbox"][1] + 60
            if any(e["role"] == "mark" and e.get("filled") and e["bbox"][0] < 120 and top < e["bbox"][1] < bottom for e in els):
                unread.add(row["text"])
        await d.reads(*rows, *[e for e in els if e["role"] == "mark" and e.get("filled") and e["bbox"][0] < 120])
        before = len(seen)
        await d.scroll_down(frac=220)
        rows = sms_senders(d, await d.els())
        if all(r["text"] in seen for r in rows) and before == len(seen) and rows and rows[-1]["text"] == seen[-1]:
            break
    await submit_answers(d, [str(len(unread))])


async def calendar_open_date(d: Driver, iso: str) -> None:
    """Show a date in the month grid and select it, swiping to its month as a person would."""
    _, month, day = (int(x) for x in iso.split("-"))
    for _ in range(14):
        head = await d.find(regex=r"^(\d+年)?\d+月$", region=(0, 40, 400, 110))
        await d.reads(head)
        shown = int(re.search(r"(\d+)月$", head["text"]).group(1))
        if shown == month:
            break
        forward = (month - shown) % 12 <= 6
        await d.swipe((850, 350) if forward else (150, 350), (150, 350) if forward else (850, 350), note=f"month {shown}->{month}")
        await asyncio.sleep(0.5)
    cells = [e for e in await d.els() if e.get("text") == str(day) and not e.get("dim") and e["role"] == "text" and 180 < e["bbox"][1] < 530]
    if not cells:
        raise OracleError(f"no cell for {iso}\n{tv.render(d.obs_cache)}")
    await d.tap_xy(*d.centre(cells[0]), note=f"date {iso}")


@oracle("calendar.CalculateDateInterval")
async def calendar_interval(d: Driver) -> None:
    a, b = date.fromisoformat(d.task.p.date1), date.fromisoformat(d.task.p.date2)
    await submit_answers(d, [str((b - a).days)])


def below(d: Driver, els: list[dict], anchor: dict, **kw) -> dict | None:
    """The nearest element under anchor that matches."""
    hits = [e for e in d._match(els, **kw) if e["bbox"][1] >= anchor["bbox"][1]]
    return min(hits, key=lambda e: e["bbox"][1]) if hits else None


@oracle("calendar.CompareScheduleDensity")
async def calendar_density(d: Driver) -> None:
    counts = {}
    for iso in (d.task.p.date1, d.task.p.date2):
        await calendar_open_date(d, iso)
        await asyncio.sleep(0.4)
        titles: set[str] = set()
        for _ in range(4):
            els = await d.els()
            if d._match(els, text="暂无日程"):
                break
            # Below the grid each event is a black title with a gray time line.
            new = {e["text"] for e in els if e["role"] == "text" and e.get("color") == "black" and 650 <= e["bbox"][1] <= 920}
            if new <= titles:
                break
            await d.reads(*[e for e in els if e.get("text") in new])
            titles |= new
            await d.swipe((500, 880), (500, 720), note="event list")
        counts[iso] = len(titles)
    a, b = counts[d.task.p.date1], counts[d.task.p.date2]
    await submit_choice(d, "一样多" if a == b else d.task.p.date1 if a > b else d.task.p.date2)


@oracle("alipay.CheckBalance")
async def alipay_balance(d: Driver) -> None:
    await d.tap(text="理财", region=(250, 960, 350, 990))
    head = await d.find(regex="总资产")
    value = below(d, await d.els(), head, regex=r"^[\d,]+\.\d\d$")
    await d.reads(value)
    await submit_answers(d, [value["text"].replace(",", "")])


@oracle("wechat.EnableDarkMode")
async def wechat_dark(d: Driver) -> None:
    await d.tap(text="我", region=(840, 960, 910, 990))
    await d.tap(text="设置")
    await d.tap(text="通用")
    await d.tap(text="深色模式")
    follow = await d.find(role="switch")
    await d.reads(follow)
    if follow["on"]:
        await d.tap(role="switch")
    await d.tap(text="深色模式", region=(0, 280, 500, 350))
    await d.tap(text="完成")


async def meeting_enter_chat(d: Driver, topic: str) -> None:
    row = await d.scroll_to(text=topic, frac=250)
    join = below(d, d.obs_cache["elements"], row, text="入会")
    await d.tap_xy(*d.centre(join), note=f"join {topic}")
    await d.tap(text="说点什么...", wait=5)
    await d.find(text="聊天")


async def meeting_send(d: Driver, message: str) -> None:
    field = await d.find(role="field", regex="消息")
    await d.type(message, point=d.centre(field))
    await d.tap(role="icon", color="blue", region=(880, 500, 1000, 1000))
    await d.find(text=message)


@oracle("tencent_meeting.ChatInMeeting")
async def meeting_chat(d: Driver) -> None:
    await meeting_enter_chat(d, d.task.p.topic)
    await meeting_send(d, d.task.p.message)


@oracle("tencent_meeting.ChatWithSpecificUser")
async def meeting_chat_user(d: Driver) -> None:
    await meeting_enter_chat(d, d.task.p.topic)
    await d.tap(text="会议中所有人")
    await d.tap(text=d.task.p.target_user, region=(140, 220, 1000, 900))
    await meeting_send(d, d.task.p.message)


@oracle("wechat_reading.AddBookToShelf")
async def reading_add(d: Driver) -> None:
    title = d.task.p.book_title
    await d.tap(role="text", region=(130, 50, 320, 90))
    await d.type(title)
    await d.enter()
    await d.tap(text=title, region=(0, 120, 1000, 600))
    await d.tap(regex="加入书架", wait=5)
    await d.find(regex="已在书架|已加入|移出书架")


@oracle("railway12306.CheckDefaultPassengerName")
async def r12306_default_passenger(d: Driver) -> None:
    await d.tap(text="我的", region=(840, 955, 960, 990))
    await d.tap(text="乘车人")
    head = await d.find(text="当前用户")
    name = below(d, await d.els(), head, color="green")
    await d.reads(name)
    await submit_answers(d, [name["text"]])

