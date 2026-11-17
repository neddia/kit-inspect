"""Tests for the review tool's manifest fixes and part picking (no window, no SAM).

Run: uv run pytest
"""

import csv

import src.labeling.review as review

SPEC = {"bolt": 3, "nut": 3, "screw": 2}
COLUMNS = ["image", "session", "split", "tier", "kit_id", "bolts", "nuts", "screws", "other", "expected", "scenario"]


def row(bolts=3, nuts=3, screws=2, scenario="complete", expected="PASS"):
    return {"image": "s/a.jpg", "session": "s", "split": "train", "tier": "normal", "kit_id": "KIT-A",
            "bolts": str(bolts), "nuts": str(nuts), "screws": str(screws), "other": "0",
            "expected": expected, "scenario": scenario}


def test_hidden_part_makes_a_complete_tray_stacked():
    assert review.fix_hidden(row(), SPEC) == {"scenario": "stacked", "expected": "NOT_PASS"}


def test_miscount_sets_counts_to_labelled_and_rescores():
    fix = review.fix_counts(row(), {"bolt": 3, "nut": 2, "screw": 2}, SPEC)
    assert fix["nuts"] == 2 and fix["scenario"] == "missing_nut" and fix["expected"] == "FAIL"


def test_miscount_keeps_a_stacked_tray_stacked():
    fix = review.fix_counts(row(scenario="stacked", expected="NOT_PASS"), {"bolt": 3, "nut": 3, "screw": 2}, SPEC)
    assert fix["scenario"] == "stacked" and fix["expected"] == "NOT_PASS"


def test_update_manifest_changes_one_row_and_logs_it(tmp_path, monkeypatch):
    manifest, changes = tmp_path / "manifest.csv", tmp_path / "changes.csv"
    monkeypatch.setattr(review, "MANIFEST", manifest)
    monkeypatch.setattr(review, "CHANGES", changes)
    other = row() | {"image": "s/b.jpg"}
    with manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows([row(), other])

    review.update_manifest("s/a.jpg", {"scenario": "stacked", "expected": "NOT_PASS"}, "test")

    _, rows = review.read_manifest()
    assert rows[0]["scenario"] == "stacked" and rows[1] == other
    log = list(csv.DictReader(changes.open()))
    assert [(r["field"], r["old"], r["new"]) for r in log] == [("expected", "PASS", "NOT_PASS"), ("scenario", "complete", "stacked")] \
        or [(r["field"], r["old"], r["new"]) for r in log] == [("scenario", "complete", "stacked"), ("expected", "PASS", "NOT_PASS")]


def test_click_picks_the_smallest_outline_under_the_mouse():
    big = {"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}
    small = {"polygon": [[40, 40], [60, 40], [60, 60], [40, 60]]}
    assert review.hit([big, small], 50, 50) == 1
    assert review.hit([big, small], 10, 10) == 0
    assert review.hit([big, small], 200, 200) is None


def test_a_part_in_two_pieces_gets_one_outline_spanning_both():
    import numpy as np
    from src.labeling.auto_label import describe
    mask = np.zeros((100, 300), bool)
    mask[40:50, 10:100] = True    # left piece of a screw
    mask[40:50, 150:290] = True   # right piece, past a bolt lying across it
    obj = describe(mask)
    assert obj["box"][0] <= 10 and obj["box"][2] >= 289   # one box over both pieces


def test_sam_mask_returns_none_instead_of_crashing():
    import torch
    from types import SimpleNamespace as NS

    def fake_sam(masks):
        return NS(predict=lambda *a, **k: [NS(masks=masks)])

    assert review.sam_mask(fake_sam(None), "x.jpg", bboxes=[[0, 0, 1, 1]]) is None
    assert review.sam_mask(fake_sam(NS(data=torch.zeros(0, 10, 10))), "x.jpg") is None   # the crash case
    full = NS(data=torch.ones(1, 10, 10))
    assert review.sam_mask(fake_sam(full), "x.jpg").sum() == 100

    def broken(*a, **k):
        raise RuntimeError("boom")
    assert review.sam_mask(NS(predict=broken), "x.jpg") is None
