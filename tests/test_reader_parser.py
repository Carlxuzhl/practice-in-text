"""The reader's line parser, with no model. The reader often writes attributes after the box; those lines must not be
thrown away. The prompts must stay byte for byte as the study used them."""
import hashlib

from reader import reader as R

CASES = [
    ('text "\u63d0\u4ea4\u7b54\u6848" white (406,930,594,955) on-blue',
     {"role": "text", "text": "\u63d0\u4ea4\u7b54\u6848", "color": "white", "bg": "blue", "bbox": [406, 930, 594, 955]}),
    ('field "Search a city" gray (95,164,547,199) empty', {"role": "field", "text": "Search a city", "empty": True}),
    ('text "Balance" black bold (10,20,30,40)', {"text": "Balance", "bold": True}),
    ('text "Balance" black (10,20,30,40) bold', {"text": "Balance", "bold": True}),
    ('image (70,365,180,415) on-white', {"role": "image", "bg": "white"}),
    ('mark filled black (394,153,474,192) on-white', {"role": "mark", "filled": True}),
    ('switch on green (800,300,900,340)', {"role": "switch", "on": True}),
    ('[12] icon settings black (885,80,944,112)', {"role": "icon", "name": "settings"}),
    ('text "plan (1,2,3,4) notes" black (10,20,30,40)', {"text": "plan (1,2,3,4) notes", "bbox": [10, 20, 30, 40]}),
    ('button "Submit" white on-blue (55,906,945,977)', {"role": "text", "text": "Submit", "bg": "blue", "bbox": [55, 906, 945, 977]}),
    ('button "Submit" filled blue (50,905,940,975)', {"role": "text", "text": "Submit"}),
    ('Toggle on green (800,300,900,340)', {"role": "icon"}),
    ('text "x" black (-5,20,1200,40)', {"bbox": [0, 20, 1000, 40]}),
    ('switch green (800,300,900,340)', {"role": "switch", "on": False}),
]
BAD = ['text "<-" black (905', "icon chevron-right", "just an explanation", 'text "x" black (10,20,5,40)']


def test_good_lines():
    for line, want in CASES:
        obs, bad = R.parse_reader(line)
        assert bad == 0 and len(obs["elements"]) == 1, line
        got = obs["elements"][0]
        for k, v in want.items():
            assert got.get(k) == v, (line, k, got)


def test_bad_lines():
    for line in BAD:
        obs, bad = R.parse_reader(line)
        assert bad == 1 and not obs["elements"], line


def test_ids_and_code_fences():
    obs, bad = R.parse_reader('```\ntext "a" black (1,1,10,10)\n\nicon plus blue (20,20,30,30)\n```')
    assert bad == 0
    assert [e["id"] for e in obs["elements"]] == [1, 2]


def test_prompts_unchanged():
    from policy import model as pm
    assert hashlib.sha256(R.READER_PROMPT.encode()).hexdigest() == "a1967b2b83bd68b21e05fe3f56d8083fc0080d3971342fc0649beb70bffbe075"
    assert hashlib.sha256(pm.POLICY_SYSTEM.encode()).hexdigest() == "8a47ddab0280c99bb1d8c5ed4c692b598ba2a5ba9163a7f74c3df48ccb151ced"
    assert hashlib.sha256(pm.POLICY_SYSTEM_NOTES.encode()).hexdigest() == "59b18010f5a4ce17500bf7b61f1effdf871a88ab2715d1e8f832f02a3912e652"
