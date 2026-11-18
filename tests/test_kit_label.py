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
    assert "two different" in check_label(two, RIM, "KIT-A", KITS)["reason"]
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
    from src.inspection.kit_label import make_reader, read_text
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
    r = check_label(read_text(make_reader(), img), RIM, "KIT-A", KITS)
    assert r["status"] == "mismatch", r
