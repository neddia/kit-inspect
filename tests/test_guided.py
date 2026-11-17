"""Tests for guided capture's plan logic (no camera needed).

Run: uv run pytest
"""

from pathlib import Path

import pytest
import yaml

from src.capture.capture import scenario_for
from src.capture.guided import build_shots, change_text, position, session_mix, is_swap

SPEC = {"bolt": 3, "nut": 3, "screw": 2}
PARTS = ["bolt", "nut", "screw"]
PLAN = yaml.safe_load((Path(__file__).parents[1] / "configs" / "capture_plan.yaml").read_text())


def test_every_planned_session_builds_and_matches_its_scenarios():
    for s in PLAN["sessions"]:
        mix = session_mix(s, PLAN)
        shots = build_shots(mix, SPEC, PARTS)
        assert len(shots) == sum(mix.values())
        for shot in shots:
            want = "two_faults" if is_swap(shot["scenario"]) else shot["scenario"]
            assert scenario_for(shot["counts"], SPEC, shot["stacked"]) == want


def test_default_mix_is_40_shots_grouped_in_order():
    shots = build_shots(PLAN["default_mix"], SPEC, PARTS)
    assert len(shots) == 40
    order = [s["scenario"] for s in shots]
    n = PLAN["default_mix"]["complete"]
    assert order[:n] == ["complete"] * n
    assert order[-1] == "empty"


def test_extra_part_rotates_through_the_parts():
    shots = build_shots({"extra_part": 3}, SPEC, PARTS)
    extras = [next(p for p in PARTS if s["counts"][p] > SPEC[p]) for s in shots]
    assert extras == ["bolt", "nut", "screw"]


def test_unknown_scenario_is_rejected():
    with pytest.raises(SystemExit):
        build_shots({"missing_washer": 1}, SPEC, PARTS)


def test_change_text_says_what_to_add_or_remove():
    before = {"bolt": 3, "nut": 3, "screw": 2, "other": 0}
    after = {"bolt": 2, "nut": 4, "screw": 2, "other": 0}
    assert change_text(before, after) == "bolt -1, nut +1"
    assert change_text(before, before) == ""


def test_position_resumes_at_first_unfinished_shot():
    order = [{"name": "a"}, {"name": "b"}]
    shots = {"a": [0] * 3, "b": [0] * 2}
    assert position(order, shots, {}) == (0, 0)
    assert position(order, shots, {"a": 2}) == (0, 2)
    assert position(order, shots, {"a": 3, "b": 1}) == (1, 1)
    assert position(order, shots, {"a": 3, "b": 2}) is None


def test_swap_puts_a_smaller_part_in_place_of_a_real_one():
    shots = build_shots({"swap_screw": 1}, SPEC, PARTS)
    assert shots[0]["scenario"] == "swap_screw"
    assert shots[0]["counts"] == {"bolt": 3, "nut": 3, "screw": 1, "other": 1}


def test_lookalike_replaces_a_part_with_something_that_counts_as_other():
    shots = build_shots({"lookalike_nut": 1}, SPEC, PARTS)
    assert shots[0]["scenario"] == "lookalike_nut"
    assert shots[0]["counts"]["nut"] == SPEC["nut"] - 1 and shots[0]["counts"]["other"] == 1
