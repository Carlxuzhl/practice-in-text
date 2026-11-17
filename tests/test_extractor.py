"""Leakage checks of the text view's extractor, on made-up pages in a headless Chromium (no MobileGym needed):
1. the extractor's source touches no forbidden API (classes, ids, data-*, aria, titles, routes, app state, ...);
2. hidden or unpainted strings never come out, painted ones do;
3. roles do not depend on tags or interactivity, and the order follows the screen.
Skipped when Playwright or its Chromium is not installed, or the icon table has not been built (scripts/setup_sim.sh)."""
import asyncio
import hashlib
import json
import re

import pytest

from textview import textview as tv

pw_api = pytest.importorskip("playwright.async_api")
if not tv.ICON_TABLE.exists():
    pytest.skip("textview/lucide_icon_table.json not built (scripts/setup_sim.sh)", allow_module_level=True)

FORBIDDEN_SOURCE = [
    r"\.className", r"\.classList", r"\.id\b", r"\.dataset", r"aria", r"\.title\b", r"\.alt\b", r"\.href\b",
    r"\.src\b", r"\.name\b", r"\.role\b", r"location\.", r"__SIM", r"__OS__", r"localStorage", r"sessionStorage",
    r"__getScrollMeta__", r"onclick", r"\.disabled\b", r"\.checked\b", r"\.readOnly\b", r"history\.",
]

PIX = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

FORBIDDEN_SOURCE = [
    r"\.className", r"\.classList", r"\.id\b", r"\.dataset", r"aria", r"\.title\b", r"\.alt\b", r"\.href\b",
    r"\.src\b", r"\.name\b", r"\.role\b", r"location\.", r"__SIM", r"__OS__", r"localStorage", r"sessionStorage",
    r"__getScrollMeta__", r"onclick", r"\.disabled\b", r"\.checked\b", r"\.readOnly\b", r"history\.",
]

CANARY_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>
body{{margin:0;font:16px sans-serif;background:#fff;color:#111}}
.row{{height:40px}}
.srOnly{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}}
.ph::placeholder{{color:transparent}}
</style></head><body>
<div class="row" id="CANARY_ID" title="CANARY_TITLE" aria-label="CANARY_ARIA" data-trigger="CANARY_TRIGGER" data-action="CANARY_ACTION">VISIBLE_PLAIN</div>
<a class="row CANARY_CLASS" href="/CANARY_HREF" style="display:block;color:#111">VISIBLE_LINK</a>
<div style="display:none">CANARY_DISPLAY_NONE</div>
<div style="visibility:hidden">CANARY_VIS_HIDDEN</div>
<div style="opacity:0">CANARY_OPACITY0</div>
<div style="opacity:0"><span>CANARY_OPACITY_PARENT</span></div>
<div style="position:absolute;left:-9999px">CANARY_OFFSCREEN_LEFT</div>
<div style="position:absolute;top:5000px">CANARY_OFFSCREEN_BELOW</div>
<div style="transform:translateX(-2000px)">CANARY_TRANSFORM</div>
<div style="height:0;overflow:hidden">CANARY_ZERO_HEIGHT</div>
<div class="srOnly">CANARY_SR_ONLY</div>
<div style="font-size:0">CANARY_FONT0</div>
<div style="color:transparent">CANARY_TRANSPARENT</div>
<div style="color:#fff;background:#fff">CANARY_SAME_COLOR</div>
<div style="clip-path:inset(100%)">CANARY_CLIP_PATH</div>
<div style="position:relative;height:30px;background:#eee"><div style="position:absolute;z-index:-1">CANARY_NEG_Z</div></div>
<details><summary>VISIBLE_SUMMARY</summary>CANARY_DETAILS</details>
<template>CANARY_TEMPLATE</template><noscript>CANARY_NOSCRIPT</noscript>
<script>var CANARY_SCRIPT = 1;</script><style>.CANARY_STYLE{{color:red}}</style>
<input type="hidden" value="CANARY_INPUT_HIDDEN">
<div class="row"><input value="VISIBLE_VALUE" placeholder="CANARY_PLACEHOLDER_FILLED" name="CANARY_NAME"></div>
<div class="row"><input placeholder="VISIBLE_PLACEHOLDER"></div>
<div class="row"><input type="password" value="CANARY_PASSWORD"></div>
<div class="row"><select><option>VISIBLE_OPTION</option><option>CANARY_OPTION</option></select></div>
<div class="row"><img src="{PIX}" alt="CANARY_ALT" title="CANARY_IMG_TITLE" width="30" height="30"></div>
<div class="row"><svg width="24" height="24" viewBox="0 0 24 24"><title>CANARY_SVG_TITLE</title><desc>CANARY_SVG_DESC</desc><path d="M5 12h14" stroke="#000"/><path d="M12 5v14" stroke="#000"/></svg></div>
<div id="ell" style="width:100px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">VISIBLEHEAD0123456789ABCDEFGHIJKLMNOPQRSTUV_CANARY_ELLIPSIS</div>
<div style="width:120px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden">CLAMPHEAD aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk lll mmm nnn ooo ppp qqq CANARY_CLAMP</div>
<div style="height:30px;overflow-y:auto"><div style="height:30px">VISIBLE_SCROLL_TOP</div><div style="height:30px">CANARY_SCROLL_BELOW</div></div>
<div style="position:relative;height:30px"><div>CANARY_COVERED</div><div style="position:absolute;inset:0;background:#fff"></div></div>
<div style="position:relative;height:30px"><div>VISIBLE_UNDER_SCRIM</div><div style="position:absolute;inset:0;background:rgba(0,0,0,.2)"></div></div>
<div style="position:relative;height:30px"><div>VISIBLE_UNDER_NOPE</div><div style="position:absolute;inset:0;pointer-events:none"></div></div>
<div style="height:48px;width:200px;transform:scale(.8);transform-origin:top left;overflow:hidden"><div style="height:48px">VISIBLE_SCALED_TOP</div><div style="height:48px">CANARY_SCALED_CLIP</div></div>
<div style="position:relative;height:30px;background:#fff"><div style="position:absolute;inset:0;background:rgba(15,23,42,.9);color:#fff">VISIBLE_ON_TRANSLUCENT</div></div>
<div class="row"><input style="color:transparent" value="CANARY_TRANSPARENT_VALUE"></div>
<div class="row"><input class="ph" placeholder="CANARY_PLACEHOLDER_TRANSPARENT"></div>
<div class="row"><input id="longin" style="width:120px" value="CANARY_INPUT_HEAD_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaINPUT_TAIL"></div>
<textarea id="longta" style="display:block;width:320px;height:40px;font:16px/20px sans-serif;padding:0;border:1px solid #888">CANARY_TEXTAREA_LINE1
b
c
d
VISIBLE_TEXTAREA_LINE5
VISIBLE_TEXTAREA_LINE6
CANARY_TEXTAREA_LINE7
h</textarea>
<div id="outer" style="height:30px;overflow-y:auto"><div style="height:30px"></div><textarea style="display:block;width:320px;height:60px;font:16px/20px sans-serif;padding:0;border:0;background:transparent">CANARY_OUTER_SCROLLED
VISIBLE_OUTER_LINE2</textarea><div style="height:60px"></div></div>
<div class="row"><svg width="24" height="24" viewBox="0 0 24 24"><path d="M1 1h20v20z" fill="#e11"/></svg></div>
<div class="row"><svg style="color:transparent" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M4 12l5 5L20 6"/></svg></div>
<div class="row"><span style="display:inline-block;width:20px;height:20px;background:#22c;-webkit-mask-image:url('data:image/svg+xml;CANARY_MASK_FILE,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%2220%22 height=%2220%22%3E%3Crect width=%2220%22 height=%2220%22/%3E%3C/svg%3E')"></span></div>
<div class="row" style="background:#fff"><div style="width:20px;height:20px;border:2px solid #999;border-radius:50%;background:#fff"></div></div>
<div class="row" style="background:#fff"><div style="width:20px;height:20px;border-radius:50%;background:#e11"></div></div>
<div style="width:110px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font:14px sans-serif">\u897f\u6e56 \u7075\u9690\u5bfa \u5e26\u5145\u7535\u5668</div>
<div class="row" style="background:#fff;color:rgba(0,0,0,.4)">VISIBLE_TRANSLUCENT_INK</div>
<script>longin.scrollLeft = 9999; longta.scrollTop = 80; outer.scrollTop = 50;</script>
</body></html>"""

MUST_APPEAR = ["VISIBLE_PLAIN", "VISIBLE_LINK", "VISIBLE_SUMMARY", "VISIBLE_VALUE", "VISIBLE_PLACEHOLDER",
               "VISIBLE_OPTION", "VISIBLEHE", "CLAMPHEAD", "VISIBLE_SCROLL_TOP", "VISIBLE_UNDER_SCRIM",
               "VISIBLE_UNDER_NOPE", "VISIBLE_SCALED_TOP", "VISIBLE_ON_TRANSLUCENT", "INPUT_TAIL",
               "VISIBLE_TEXTAREA_LINE5", "VISIBLE_TEXTAREA_LINE6", "VISIBLE_OUTER_LINE2"]

STRUCT_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;font:16px sans-serif;background:#fff;color:#111}
.a{all:unset;position:absolute;left:20px;font:16px sans-serif;color:#111}
</style></head><body>
<button class="a" style="top:200px" disabled>SAME_C</button>
<div class="a" style="top:100px;cursor:pointer" onclick="1">SAME_B</div>
<span class="a" style="top:10px">SAME_A</span>
<input style="position:absolute;top:300px;left:20px;border:0;outline:0;background:transparent">
</body></html>"""


EXTRA_HTML = """<!doctype html><html><head><meta charset="utf-8"><style>body{margin:0;background:#fff}div{height:40px}</style></head><body>
<div><svg style="transform:rotate(180deg)" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000"><path d="m9 18 6-6-6-6"/></svg></div>
<div><svg style="rotate:90deg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#000"><path d="m9 18 6-6-6-6"/></svg></div>
<div><input type="checkbox" checked></div>
<div><input type="checkbox"></div>
<div><input type="radio" style="opacity:0"></div>
<div style="height:auto"><span style="display:grid;grid-template-columns:repeat(3,4px);gap:2px;width:16px">
<i style="width:4px;height:4px;background:#666"></i><i style="width:4px;height:4px;background:#666"></i><i style="width:4px;height:4px;background:#666"></i>
<i style="width:4px;height:4px;background:#666"></i><i style="width:4px;height:4px;background:#666"></i><i style="width:4px;height:4px;background:#666"></i></span></div>
</body></html>"""


RASTER_HTML = f"""<!doctype html><html><head><meta charset="utf-8"><style>body{{margin:0;background:#fff}}div{{height:40px}}</style></head><body>
<div><img src="{PIX}#CANARY_RASTER_FRAGMENT" alt="CANARY_RASTER_ALT" width="30" height="30"></div>
<div><span style="display:inline-block;width:30px;height:30px;background-image:url('{PIX}')"></span></div>
</body></html>"""


def test_source_lint():
    body = "\n".join(line for line in tv.JS.splitlines() if not line.strip().startswith("//"))
    # `item`, `it`, `prev` are the extractor's own output records, not DOM nodes.
    scan = re.sub(r"\b(item|it|prev)\.(name|role)\b", "", body)
    assert not [p for p in FORBIDDEN_SOURCE if re.search(p, scan)]
    assert set(re.findall(r"getAttribute\(([^)]*)\)", body)) <= {"k"}


async def _pages() -> dict:
    report: dict = {}
    async with pw_api.async_playwright() as pw:
        try:
            browser = await pw.chromium.launch(headless=True)
        except Exception as e:  # Chromium not installed
            pytest.skip(f"Chromium unavailable: {e}")
        ctx = await browser.new_context(viewport={"width": 360, "height": 2000}, device_scale_factor=2)
        page = await ctx.new_page()
        await page.set_content(CANARY_HTML)
        out = await tv.extract(page)
        blob = json.dumps(out, ensure_ascii=False)
        report["canary_leaks"] = sorted(set(re.findall(r"CANARY_[A-Z0-9_]+", blob)))
        report["missing_visible"] = [s for s in MUST_APPEAR if s not in blob]
        ell = [e for e in out["elements"] if (e.get("text") or "").startswith("VISIBLEHE")]
        report["ellipsis_ok"] = bool(ell) and ell[0]["text"].endswith("\u2026")
        report["password_masked"] = bool([e for e in out["elements"] if e["role"] == "field" and "\u2022" in (e.get("text") or "")])
        report["svg_shape_colour"] = [e.get("color") for e in out["elements"] if e["role"] == "icon" and e["bbox"][1] > 400]
        report["marks"] = [(e["filled"], e["color"]) for e in out["elements"] if e["role"] == "mark"]
        report["spaced_ellipsis"] = [e["text"] for e in out["elements"] if (e.get("text") or "").startswith("\u897f\u6e56")]
        report["translucent_ink"] = [e["color"] for e in out["elements"] if e.get("text") == "VISIBLE_TRANSLUCENT_INK"]

        await page.set_content(STRUCT_HTML)
        out = await tv.extract(page)
        same = [e for e in out["elements"] if (e.get("text") or "").startswith("SAME_")]
        report["order"] = [e["text"] for e in same]
        shape = [{k: v for k, v in e.items() if k not in ("id", "text", "bbox")} for e in same]
        report["no_tag_or_interactivity_leak"] = all(s == shape[0] for s in shape)
        report["inkless_field_dropped"] = not [e for e in out["elements"] if e["role"] == "field"]

        await page.set_content(EXTRA_HTML)
        out = await tv.extract(page)
        report["turned_icons"] = [e.get("name") for e in out["elements"] if e["role"] == "icon"]
        report["native_marks"] = [e["filled"] for e in out["elements"] if e["role"] == "mark"]

        # A picture's address keys the name table and goes no further: unknown stays bare, known gets the table's name.
        await page.set_content(RASTER_HTML)
        bare = await tv.extract(page)
        keyed = await tv.extract(page, keep_shapes=True)
        report["raster_bare"] = [e.get("name") for e in bare["elements"] if e["role"] == "icon"]
        shapes = [e.get("shape", "") for e in keyed["elements"] if e["role"] == "icon"]
        report["raster_keys_collected"] = len(shapes) == 2 and all(s.startswith("raster:") for s in shapes)
        tv.icon_name("")
        entry = {hashlib.sha1(s.encode()).hexdigest()[:16]: "raster-test" for s in shapes}
        tv._custom.update(entry)
        named = await tv.extract(page)
        for k in entry:
            del tv._custom[k]
        report["raster_named"] = [e.get("name") for e in named["elements"] if e["role"] == "icon"]
        blob = json.dumps(bare, ensure_ascii=False) + json.dumps(named, ensure_ascii=False)
        report["raster_leaks"] = sorted(set(re.findall(r"CANARY_[A-Z0-9_]+|raster:|R0lGOD", blob)))
        await browser.close()
    return report


def test_canary_pages():
    r = asyncio.run(_pages())
    assert not r["canary_leaks"], r["canary_leaks"]
    assert not r["missing_visible"], r["missing_visible"]
    assert r["ellipsis_ok"] and r["password_masked"]
    assert r["svg_shape_colour"][:1] == ["red"] and r["svg_shape_colour"][1:] == ["blue"]
    assert r["marks"] == [(False, "gray"), (True, "red")]
    assert r["spaced_ellipsis"] and r["spaced_ellipsis"][0].endswith("\u2026")
    assert r["translucent_ink"] == ["gray"]
    assert r["order"] == ["SAME_A", "SAME_B", "SAME_C"] and r["no_tag_or_interactivity_leak"] and r["inkless_field_dropped"]
    assert r["turned_icons"] == ["chevron-left", "chevron-down", "grip"]
    assert r["native_marks"] == [True, False]
    assert r["raster_bare"] == [None, None] and r["raster_keys_collected"]
    assert r["raster_named"] == ["raster-test", "raster-test"] and not r["raster_leaks"]
