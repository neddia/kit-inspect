"""Kit label check: the rules (no OCR), plus one real OCR read of a drawn, rotated label.

Run: uv run pytest tests/test_kit_label.py
"""

import cv2
import numpy as np
import pytest

from src.inspection.decision import FAIL, PASS, REVIEW, Detection, decide
from src.inspection.kit_label import check_label, normalize

KITS = {"KIT-A": {"bolt": 3, "nut": 3, "screw": 2}, "KIT-B": {"bolt": 2, "nut": 4, "screw": 2}}
RIM = (600, 300, 300)                       # bowl centre (600, 300), rim radius 300


def text(t, x, y, conf=0.99):
    return {"text": t, "conf": conf, "center": [x, y], "box": [[x - 50, y - 20], [x + 50, y - 20], [x + 50, y + 20], [x - 50, y + 20]]}


ON_TRAY = (600 + 420, 300)                  # 1.4 radii from the centre: the tab
ON_BENCH = (600 - 560, 300)                 # 1.87 radii: lying on the bench


def test_normalize_fixes_the_prefix_but_never_guesses_the_kit_letter():
    assert normalize("kit-b") == normalize("KIT B") == normalize("K1T-B") == "KITB"
    assert normalize("KIT-8") == "KIT8"     # 8 is not silently turned into B


def test_matching_label_is_ok():
    assert check_label([text("KIT-A", *ON_TRAY)], RIM, "KIT-A", KITS)["status"] == "ok"


def test_wrong_label_on_the_tray_is_a_mismatch():
    r = check_label([text("KIT-B", *ON_TRAY)], RIM, "KIT-A", KITS)
    assert r["status"] == "mismatch" and r["reason"] == "label says KIT-B, PLC expects KIT-A"


def test_spare_label_on_the_bench_is_ignored():
    """The headline case: KIT-B on the tray, a loose KIT-A on the bench, PLC expects KIT-A."""
    r = check_label([text("KIT-B", *ON_TRAY), text("KIT-A", *ON_BENCH)], RIM, "KIT-A", KITS)
    assert r["status"] == "mismatch" and r["ignored"] == ["KIT-A"]
    r = check_label([text("KIT-A", *ON_BENCH)], RIM, "KIT-A", KITS)     # bench label alone never passes
    assert r["status"] == "review" and "no kit label" in r["reason"]


def test_unreadable_or_unknown_or_two_labels_go_to_review():
    assert check_label([text("KIT-A", *ON_TRAY, conf=0.5)], RIM, "KIT-A", KITS)["status"] == "review"
    assert check_label([text("KIT-8", *ON_TRAY)], RIM, "KIT-A", KITS)["status"] == "review"
    two = [text("KIT-A", *ON_TRAY), text("KIT-B", 600, 300 + 420)]
    assert "conflicting" in check_label(two, RIM, "KIT-A", KITS)["reason"]
    assert check_label([text("hello", *ON_TRAY)], RIM, "KIT-A", KITS)["status"] == "review"   # not a kit label


def test_decision_mismatch_fails_even_with_right_counts_and_doubt_reviews():
    dets = [Detection(c, 0.95, [i * 100, 0, i * 100 + 50, 50]) for i, c in enumerate(["bolt"] * 3 + ["nut"] * 3 + ["screw"] * 2)]
    mismatch = {"status": "mismatch", "reason": "label says KIT-B, PLC expects KIT-A"}
    doubt = {"status": "review", "reason": "no kit label found on the tray"}
    assert decide("KIT-A", KITS, dets, label=mismatch).result == FAIL
    assert decide("KIT-A", KITS, dets, label=doubt).reasons == ["no kit label found on the tray"]
    assert decide("KIT-A", KITS, dets, label={"status": "ok", "reason": "label KIT-A matches"}).result == PASS
    assert decide("KIT-A", KITS, dets[1:], label=doubt).result == FAIL        # counts already say FAIL: stays FAIL
    assert decide("KIT-A", KITS, dets, label=None).result == PASS            # check off


def test_ocr_reads_a_rotated_label_on_the_tray():
    pytest.importorskip("rapidocr")
    from src.inspection.kit_label import make_reader, inspect_label
    img = np.full((720, 1280, 3), 200, np.uint8)
    cv2.circle(img, RIM[:2], RIM[2], (150, 150, 150), 3)
    lab = np.full((80, 240, 3), 245, np.uint8)
    cv2.putText(lab, "KIT-B", (30, 58), cv2.FONT_HERSHEY_DUPLEX, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    m = cv2.getRotationMatrix2D((120, 40), 150, 1.0)                      # upside-down-ish
    m[:, 2] += [ON_TRAY[0] - 120, ON_TRAY[1] - 40]
    mask = cv2.warpAffine(np.full((80, 240), 255, np.uint8), m, (1280, 720))
    img[mask > 0] = cv2.warpAffine(lab, m, (1280, 720))[mask > 0]
    img = cv2.GaussianBlur(img, (3, 3), 0)     # like the camera. (Razor-sharp drawn text upside down read "KT-B" 0.64:
                                               # a dropped letter matches no kit -> REVIEW, the safe direction.)
    _, r = inspect_label(make_reader(), img, RIM, "KIT-A", KITS)
    assert r["status"] == "mismatch", r


def test_detections_on_the_label_are_its_letters_not_parts():
    from src.inspection.kit_label import label_overlap
    quad = [[900, 280], [1100, 280], [1100, 340], [900, 340]]      # a label's text box
    assert label_overlap([950, 285, 975, 335], [quad])                     # the "I" in KIT, read as a bolt
    assert not label_overlap([700, 200, 800, 260], [quad])                 # a real part elsewhere
    assert not label_overlap([1050, 250, 1250, 420], [quad])               # a real part only touching the label


def test_a_box_around_the_whole_label_is_the_label():
    from src.inspection.kit_label import label_overlap
    quad = [[900, 280], [1100, 280], [1100, 340], [900, 340]]      # the text
    assert label_overlap([880, 250, 1120, 380], [quad])                    # "other 0.78" around the label paper
    assert not label_overlap([600, 100, 1300, 700], [quad])                # something far bigger than a label


def test_sideways_label_is_found_by_the_turned_pass():
    """Real sideways labels on the curved wall defeat the first pass (7 of 40 live photos), but a drawn
    label doesn't, so a stand-in OCR plays that part: it only sees text in the TURNED frame."""
    import numpy as np
    from src.inspection.kit_label import inspect_label

    class Result:
        def __init__(self, boxes, txts, scores):
            self.boxes, self.txts, self.scores = boxes, txts, scores

    def ocr(frame):
        h, w = frame.shape[:2]
        if h > w:                                   # the turned frame (720 x 1280 -> 1280 tall)
            # a label at original x 280-320, y 300-420 sits at x' = 719-y, y' = x in the turned frame
            box = np.array([[299, 280], [419, 280], [419, 320], [299, 320]], float)
            return Result([box], ["KIT-A"], [0.98])
        return Result([[[20, 10], [80, 10], [80, 30], [20, 30]]], ["KIT-B"], [1.0])

    _, result = inspect_label(ocr, np.zeros((720, 1280, 3), np.uint8), (600, 360, 300), "KIT-B", KITS)
    assert result["passes"] == 2
    assert result["status"] == "mismatch"
    assert result["quads"] == [result["label"]["box"]]
    assert result["label"]["center"] == [300, 360]          # mapped back to where the label is in the real frame

def test_only_verified_carrier_can_exclude_detections():
    for reads in ([text("KIT-A", *ON_BENCH)], [text("KIT-A", *ON_TRAY, conf=0.4)],
                  [text("KIT-A", *ON_TRAY), text("KIT-B", *ON_TRAY)]):
        assert check_label(reads, RIM, "KIT-A", KITS)["quads"] == []


def test_rim_label_paper_vs_possible_object():
    from src.inspection.kit_label import mark_label_detections

    carrier = text("KIT-A", *ON_TRAY)
    paper = Detection("other", 0.85, [960, 267, 1080, 333])
    part = Detection("bolt", 0.82, [996, 285, 1010, 315])
    localized_junk = Detection("other", 0.85, part.box)
    for detection, floor, doubt in [
        (paper, (600, 300, 280, True), False),
        (part, (600, 300, 280, True), False),
        (localized_junk, (600, 300, 280, True), True),
        (part, (900, 300, 150, True), True),
        (part, (600, 300, 280, False), True),
    ]:
        label = check_label([carrier], RIM, "KIT-A", KITS)
        d = Detection(detection.cls, detection.conf, detection.box)
        mark_label_detections([d], label, floor, 0.7)
        assert d.on_label and not d.in_bowl and d.label_ambiguous == doubt
        decision = decide("KIT-A", {"KIT-A": {}}, [d], label=label)
        assert decision.result == (REVIEW if doubt else PASS)
