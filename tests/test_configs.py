"""The configs hold together: instance files match the experiment, revisions are pinned, LoRA settings complete."""
import re

from common.config import load
from policy import manifests


def test_instances_match_experiment():
    for name, spec in manifests.specs().items():
        from common.config import instances
        assert instances(name) == spec, name


def test_experiment_numbers():
    exp = load("experiment")
    assert len(exp["templates"]) == 29 and len(set(exp["templates"])) == 29
    assert len(exp["seeds"]["confirmation"]) == 12
    seeds = [s for k, v in exp["seeds"].items() if isinstance(v, list) for s in v]
    assert len(seeds) == len(set(seeds)), "instance seed sets overlap"
    assert exp["budget"] == {"screenshots": 2000, "first_part": 1000, "direct_round_2": 1000}


def test_models_pinned():
    models = load("models")
    for key in ("policy", "reader", "reader_8b", "reader_other_family", "aligned_reader"):
        assert re.fullmatch(r"[0-9a-f]{40}", models[key]["revision"]), key


def test_lora_settings():
    hp = load("lora")
    assert {k: hp[k] for k in ("lora_r", "lora_alpha", "epochs", "accumulate")} == {"lora_r": 16, "lora_alpha": 32, "epochs": 2, "accumulate": 16}
    assert hp["lr"] == 1e-4 and len(hp["targets"]) == 7


def test_reader_settings():
    call = load("reader")["call"]
    assert call == {"scale": 0.5, "temperature": 0.0, "repetition_penalty": 1.05, "max_tokens": 4096, "seed": 0, "timeout_s": 900}
