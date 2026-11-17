"""The screen reader: a vision-language model turns a screenshot into a list in the text view's line format, and
parse_reader() turns that list back into the text view's element dicts. The aligned reader uses the same prompt,
decoding settings, downscaling and parser; only its weights differ.

The prompt is in Chinese, the language of the simulated apps, and must stay byte for byte as it is: the aligned reader
was trained with it. In English it says:
  This is a phone screenshot. List everything visible on the screen line by line, top to bottom, left to right.
  One element per line, in a fixed format:  text "text on screen" colour (x1,y1,x2,y2) / icon name colour (...) /
  image (...) / field "text shown in the box" empty (...) / switch on|off colour (...) / mark filled|empty colour (...)
  Rules: boxes are element borders normalised to 0-1000 by screen width and height; text exactly as on screen, not
  translated, completed or rewritten, a cut-off text keeps its ellipsis; add on-<background colour> when the
  background is neither white nor black, bold for bold, dim for clearly faded elements; colours only from the list;
  icon = a small icon without text, named in short English, name left out when unclear; image = photo or large
  picture; field = input box, empty when it only shows a hint; switch = toggle; mark = checkbox, radio circle or dot;
  only what is visible, no explanation, no numbering, no Markdown."""
from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import re

from PIL import Image

from common.config import reader_url
from common.llm import chat

ROLES = "text|icon|image|field|switch|mark"
COLOURS = {"black", "gray", "lightgray", "white", "red", "orange", "yellow", "green", "cyan", "blue", "purple", "pink"}

READER_PROMPT = """这是一张手机屏幕截图。把屏幕上所有看得见的内容逐行列出来，从上到下、从左到右。

每个元素一行，格式固定：
text "屏幕上的文字" 颜色 (x1,y1,x2,y2)
icon 图标名 颜色 (x1,y1,x2,y2)
image (x1,y1,x2,y2)
field "框内显示的文字" empty (x1,y1,x2,y2)
switch on 颜色 (x1,y1,x2,y2)
switch off 颜色 (x1,y1,x2,y2)
mark filled 颜色 (x1,y1,x2,y2)
mark empty 颜色 (x1,y1,x2,y2)

规则：
- 坐标是元素边框，按屏幕宽高归一化到 0 到 1000，x 是横向，y 是纵向。
- text 的文字必须与屏幕完全一致，不翻译、不补全、不改写；被截断的文字保留省略号。
- 背景不是白色或黑色时，在颜色后加 on-背景色，例如 white on-blue；粗体加 bold；明显变淡的元素加 dim。
- 颜色只能用：black gray lightgray white red orange yellow green cyan blue purple pink。
- icon 是无文字的小图标，图标名用简短英文，例如 search、plus、chevron-right、settings；看不出来就省略图标名。
- image 是照片或大图。field 是输入框，框里只有提示文字时加 empty。switch 是开关。mark 是勾选框、单选圈或小圆点。
- 只写屏幕上看得见的内容，不要解释，不要编号，不要 Markdown。"""

# The box is the last one on the line: the reader often writes attributes after it, like (1,2,3,4) on-blue.
# It also uses roles the prompt does not offer, like button; such a line still shows what is on screen,
# so it is kept as text when it quotes text and as an icon otherwise.
LINE = re.compile(r"^(?:\[\d+\]\s*)?([A-Za-z][\w-]*)\b(.*)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)(.*)$")
QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


def parse_reader(raw: str) -> tuple[dict, int]:
    """Turn the reader's lines into the text view's observation dict. Returns it and the count of bad lines."""
    elements, bad = [], 0
    for line in raw.splitlines():
        line = line.strip().strip("`")
        if not line:
            continue
        m = LINE.match(line)
        if not m:
            bad += 1
            continue
        role, middle = m.group(1).lower(), m.group(2) + " " + m.group(7)
        if role not in ROLES.split("|"):
            role = "text" if QUOTED.search(middle) else "icon"
        x1, y1, x2, y2 = (max(0, min(1000, int(v))) for v in m.groups()[2:6])
        if x2 <= x1 or y2 <= y1:
            bad += 1
            continue
        e: dict = {"id": len(elements) + 1, "role": role, "bbox": [x1, y1, x2, y2]}
        quoted = QUOTED.search(middle)
        if quoted:
            try:
                e["text"] = json.loads(f'"{quoted.group(1)}"')
            except ValueError:
                e["text"] = quoted.group(1)
            middle = middle[:quoted.start()] + middle[quoted.end():]
        elif role in ("text", "field"):
            e["text"] = ""
        for word in middle.split():
            if word in COLOURS:
                e["color"] = word
            elif word.startswith("on-") and word[3:] in COLOURS:
                e["bg"] = word[3:]
            elif word in ("bold", "dim"):
                e[word] = True
            elif word == "empty" and role == "field":
                e["empty"] = True
            elif role == "switch" and word in ("on", "off"):
                e["on"] = word == "on"
            elif role == "mark" and word in ("filled", "empty"):
                e["filled"] = word == "filled"
            elif role == "icon" and "name" not in e:
                e["name"] = word
        if role == "switch":
            e.setdefault("on", False)
        if role == "mark":
            e.setdefault("filled", False)
        elements.append(e)
    return {"elements": elements}, bad


PARSER_VERSION = "v3"


def parser_digest() -> str:
    """Fingerprint of everything that decides what a reader line becomes."""
    source = inspect.getsource(parse_reader) + LINE.pattern + QUOTED.pattern + ROLES + " ".join(sorted(COLOURS))
    return hashlib.sha256(source.encode()).hexdigest()[:16]


def read_screen(png: bytes, timeout: float = 900, prompt: str = READER_PROMPT, scale: float = 1.0,
                repetition_penalty: float = 1.0, temperature: float = 0.0, seed: int = 0, max_tokens: int = 4096) -> dict:
    """One reader call on a PNG screenshot. scale 0.5 halves each side (LANCZOS) before the call. Returns the raw text,
    the parsed observation ("obs") and the number of lines the parser could not read ("bad_lines")."""
    if scale != 1.0:
        im = Image.open(io.BytesIO(png))
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        png = buf.getvalue()
    url = "data:image/png;base64," + base64.b64encode(png).decode()
    payload = {"model": "reader", "temperature": temperature if temperature else 0, "seed": seed, "max_tokens": max_tokens, "messages": [
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}},
                                     {"type": "text", "text": prompt}]}]}
    if repetition_penalty != 1.0:
        payload["repetition_penalty"] = repetition_penalty
    out = chat(reader_url(), payload, timeout)
    out["obs"], out["bad_lines"] = parse_reader(out["content"])
    return out
