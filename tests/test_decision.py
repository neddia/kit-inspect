"""Tests for the PASS / FAIL / REVIEW rules (src/inspection/decision.py). No images, no model.

Run: uv run pytest
"""

from src.inspection.decision import FAIL, PASS, REVIEW, Detection, decide, iou

KITS = {"KIT-A": {"bolt": 3, "nut": 3, "screw": 2}, "KIT-B": {"bolt": 2, "nut": 4, "screw": 2}}


def tray(bolt=3, nut=3, screw=2, conf=0.95):
    """Detections laid out in a row so none overlap."""
    dets, x = [], 0
    for cls, n in (("bolt", bolt), ("nut", nut), ("screw", screw)):
        for _ in range(n):
            dets.append(Detection(cls, conf, [x, 0, x + 50, 50]))
            x += 100
    return dets


def test_complete_kit_passes():
    assert decide("KIT-A", KITS, tray()).result == PASS


def test_missing_part_fails_with_a_reason():
    d = decide("KIT-A", KITS, tray(nut=2))
    assert d.result == FAIL and d.reasons == ["missing 1 nut"]


def test_extra_part_fails_with_a_reason():
    d = decide("KIT-A", KITS, tray(bolt=4))
    assert d.result == FAIL and d.reasons == ["1 extra bolt"]


def test_sure_foreign_object_fails_even_with_right_counts():
    d = decide("KIT-A", KITS, tray() + [Detection("other", 0.9, [2000, 0, 2050, 50])])
    assert d.result == FAIL and d.reasons == ["1 foreign object"]


def test_unsure_foreign_object_goes_to_review():
    d = decide("KIT-A", KITS, tray() + [Detection("other", 0.4, [2000, 0, 2050, 50])])
    assert d.result == REVIEW and d.reasons == ["maybe a foreign object (0.40)"]


def test_part_not_in_the_kit_counts_as_extra():
    kits = {"KIT-C": {"bolt": 3, "nut": 3}}
    d = decide("KIT-C", kits, tray())
    assert d.result == FAIL and d.reasons == ["2 extra screws"]


def test_same_tray_passes_for_one_kit_and_fails_for_another():
    assert decide("KIT-A", KITS, tray()).result == PASS
    d = decide("KIT-B", KITS, tray())
    assert d.result == FAIL and d.reasons == ["1 extra bolt", "missing 1 nut"]


def test_scene_problem_goes_to_review_before_anything_else():
    d = decide("KIT-A", KITS, tray(), scene_problems=["no tray found"])
    assert d.result == REVIEW and d.reasons == ["no tray found"]


def test_unknown_or_missing_kit_id_goes_to_review():
    assert decide("KIT-Z", KITS, tray()).result == REVIEW
    assert decide("", KITS, tray()).result == REVIEW


def test_unsure_detection_that_could_complete_the_kit_goes_to_review():
    dets = tray(nut=2) + [Detection("nut", 0.4, [2000, 0, 2050, 50])]
    d = decide("KIT-A", KITS, dets)
    assert d.result == REVIEW and "unsure about 1 nut" in d.reasons[0]


def test_unsure_detection_that_cant_change_the_answer_still_fails():
    # Missing two bolts: whatever the shaky nut is, this kit isn't complete.
    dets = tray(bolt=1) + [Detection("nut", 0.4, [2000, 0, 2050, 50])]
    d = decide("KIT-A", KITS, dets)
    assert d.result == FAIL and d.reasons == ["missing 2 bolts"]


def test_unsure_extra_on_a_complete_kit_goes_to_review():
    dets = tray() + [Detection("bolt", 0.4, [2000, 0, 2050, 50])]
    assert decide("KIT-A", KITS, dets).result == REVIEW


def test_detections_below_low_are_ignored():
    dets = tray() + [Detection("bolt", 0.1, [2000, 0, 2050, 50])]
    assert decide("KIT-A", KITS, dets).result == PASS


def test_parts_outside_the_bowl_are_ignored():
    dets = tray() + [Detection("bolt", 0.95, [2000, 0, 2050, 50], in_bowl=False)]
    assert decide("KIT-A", KITS, dets).result == PASS


def test_unknown_object_on_a_complete_kit_goes_to_review():
    d = decide("KIT-A", KITS, tray(), unknown_objects=[{"box": [0, 0, 5, 5], "area": 200}])
    assert d.result == REVIEW and d.reasons == ["1 unknown object on the tray"]


def test_unknown_object_on_an_incomplete_kit_is_a_fail_with_a_note():
    d = decide("KIT-A", KITS, tray(nut=2), unknown_objects=[{"box": [0, 0, 5, 5], "area": 200}])
    assert d.result == FAIL and d.reasons == ["missing 1 nut", "also: 1 unknown object on the tray"]


def test_two_boxes_on_one_spot_count_as_unsure():
    # A bolt detected twice on the same spot must not count as 2 bolts for sure.
    dets = tray(bolt=2) + [Detection("bolt", 0.9, [0, 0, 50, 50])]   # duplicate of the first bolt
    d = decide("KIT-A", KITS, dets)
    assert d.result == REVIEW and d.sure["bolt"] == 2 and d.unsure["bolt"] == 1


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(iou([0, 0, 10, 10], [5, 0, 15, 10]) - 1 / 3) < 1e-9


def test_nut_size_uses_longest_side_and_preserves_a_missing_part_failure():
    dets = tray()
    limits = {"KIT-A": 60}
    nuts = [d for d in dets if d.cls == "nut"]
    for nut in nuts:
        nut.box[3] = 60
    assert decide("KIT-A", KITS, dets, min_nut_side_px=limits).result == PASS
    nuts[0].box[3] = 44
    decision = decide("KIT-A", KITS, dets, min_nut_side_px=limits)
    assert decision.result == REVIEW
    assert "50.0 px" in decision.reasons[0]
    assert decide("KIT-A", KITS, dets[:-1], min_nut_side_px=limits).result == FAIL


def test_screw_size_uses_box_diagonal_and_holds_short_detection():
    dets = tray()
    screws = [d for d in dets if d.cls == "screw"]
    for d in screws:
        x = d.box[0]
        d.box = [x, 0, x + 96, 128]  # diagonal exactly 160
    cfg = {"min_screw_diagonal_px": {"KIT-A": 160}}
    assert decide("KIT-A", KITS, dets, **cfg).result == PASS
    screws[0].box[3] = 50
    decision = decide("KIT-A", KITS, dets, **cfg)
    assert decision.result == REVIEW and "check screw size" in decision.reasons[0]


def test_size_upper_bounds_allow_equality_and_report_both_oversize_parts():
    dets = tray()
    screw = next(d for d in dets if d.cls == "screw")
    screw.box[3] = 120  # 50 x 120 gives a diagonal of 130
    limits = dict(max_nut_side_px={"KIT-A": 50}, max_screw_diagonal_px={"KIT-A": 130})
    assert decide("KIT-A", KITS, dets, **limits).result == PASS
    next(d for d in dets if d.cls == "nut").box[3] += 1
    screw.box[3] += 1
    decision = decide("KIT-A", KITS, dets, **limits)
    assert decision.result == REVIEW
    assert decision.reasons == ["check nut size: box 51.0 px (maximum 50 px)",
                                "check screw size: box diagonal 130.9 px (maximum 130 px)"]
