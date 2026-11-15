"""Tests for the capture rules: scenario names, expected results, manifest files.

Run: uv run pytest
"""

import pytest

from src.capture.capture import (
    all_parts,
    append_row,
    expected_for,
    manifest_columns,
    next_index,
    prepare_manifest,
    remove_row,
    scenario_for,
    session_counts,
)

SPEC = {"bolt": 2, "nut": 2, "washer": 4}


def tray(bolt=2, nut=2, washer=4, other=0):
    return {"bolt": bolt, "nut": nut, "washer": washer, "other": other}


def test_complete_kit_passes():
    assert scenario_for(tray(), SPEC) == "complete"
    assert expected_for(tray(), SPEC, "normal") == "PASS"


def test_one_missing_part_fails():
    assert scenario_for(tray(washer=3), SPEC) == "missing_washer"
    assert scenario_for(tray(nut=1), SPEC) == "missing_nut"
    assert scenario_for(tray(bolt=1), SPEC) == "missing_bolt"
    assert expected_for(tray(washer=3), SPEC, "normal") == "FAIL"


def test_extra_part_fails():
    assert scenario_for(tray(nut=3), SPEC) == "extra_part"
    assert expected_for(tray(nut=3), SPEC, "normal") == "FAIL"


def test_two_faults():
    assert scenario_for(tray(washer=2), SPEC) == "two_faults"  # two washers missing
    assert scenario_for(tray(washer=3, nut=3), SPEC) == "two_faults"


def test_empty_tray_fails():
    assert scenario_for(tray(0, 0, 0), SPEC) == "empty"
    assert expected_for(tray(0, 0, 0), SPEC, "normal") == "FAIL"


def test_foreign_object_must_not_pass():
    assert scenario_for(tray(other=1), SPEC) == "foreign_object"
    assert expected_for(tray(other=1), SPEC, "normal") == "NOT_PASS"


def test_stacked_must_not_pass_even_if_counts_are_right():
    assert scenario_for(tray(), SPEC, stacked=True) == "stacked"
    assert expected_for(tray(), SPEC, "normal", stacked=True) == "NOT_PASS"


def test_invalid_scene_must_not_pass():
    assert expected_for(tray(), SPEC, "invalid") == "NOT_PASS"


def test_numbering_continues_after_existing_images(tmp_path):
    assert next_index(tmp_path) == 1
    (tmp_path / "pilot_0001.jpg").touch()
    (tmp_path / "pilot_0007.jpg").touch()
    assert next_index(tmp_path) == 8


def test_manifest_append_count_and_undo(tmp_path):
    manifest = tmp_path / "manifest.csv"
    cols = manifest_columns(["bolt", "nut", "washer"])
    row = {"image": "pilot/pilot_0001.jpg", "session": "pilot", "split": "train", "tier": "normal",
           "kit_id": "KIT-A", "bolts": 2, "nuts": 2, "washers": 4, "other": 0,
           "expected": "PASS", "scenario": "complete"}
    append_row(manifest, row, cols)
    append_row(manifest, {**row, "image": "pilot/pilot_0002.jpg", "washers": 3,
                          "expected": "FAIL", "scenario": "missing_washer"}, cols)
    assert session_counts(manifest, "pilot") == {"complete": 1, "missing_washer": 1}
    remove_row(manifest, "pilot/pilot_0002.jpg", cols)
    assert session_counts(manifest, "pilot") == {"complete": 1}
    assert manifest.read_text().startswith("image,session,split")  # header kept


# ---------- other kits: parts come from kits.yaml ----------

SCREW_KIT = {"bolt": 3, "nut": 3, "screw": 2}


def screw_tray(bolt=3, nut=3, screw=2, other=0):
    return {"bolt": bolt, "nut": nut, "screw": screw, "other": other}


def test_any_parts_work():
    assert scenario_for(screw_tray(), SCREW_KIT) == "complete"
    assert scenario_for(screw_tray(screw=1), SCREW_KIT) == "missing_screw"
    assert expected_for(screw_tray(screw=1), SCREW_KIT, "normal") == "FAIL"


def test_part_from_another_kit_counts_as_extra():
    tray = {**screw_tray(), "washer": 1}  # a washer in a kit that has none
    assert scenario_for(tray, SCREW_KIT) == "extra_part"
    assert expected_for(tray, SCREW_KIT, "normal") == "FAIL"


def test_columns_follow_the_kits():
    kits = {"KIT-A": SCREW_KIT, "KIT-B": {"bolt": 1, "washer": 2}}
    assert all_parts(kits) == ["bolt", "nut", "screw", "washer"]
    assert manifest_columns(["bolt", "screw"])[5:7] == ["bolts", "screws"]


def test_header_rewritten_when_empty_but_refused_when_rows_exist(tmp_path):
    manifest = tmp_path / "manifest.csv"
    old = manifest_columns(["bolt", "nut", "washer"])
    new = manifest_columns(["bolt", "nut", "screw"])
    prepare_manifest(manifest, old)
    prepare_manifest(manifest, new)  # no rows yet: just switches header
    assert manifest.read_text().splitlines()[0] == ",".join(new)
    append_row(manifest, dict.fromkeys(new, 0) | {"image": "a/a_0001.jpg"}, new)
    with pytest.raises(SystemExit):
        prepare_manifest(manifest, old)  # rows exist: don't silently mix layouts
