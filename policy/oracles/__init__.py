"""Program oracles for the 29 practiced templates: they record the text-view demonstrations the start policy imitates
(policy/episodes.py --oracle)."""
from policy.oracles.set1 import ORACLES as _PART1
from policy.oracles.set2 import ORACLES as _PART2

if set(_PART1) & set(_PART2):
    raise RuntimeError(f"templates with two oracles: {sorted(set(_PART1) & set(_PART2))}")
ORACLES = {**_PART1, **_PART2}
