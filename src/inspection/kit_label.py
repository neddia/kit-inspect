"""Kit label check (OCR): does the label ON THE TRAY match the kit the PLC says it's building?

    photo -> OCR finds and reads every piece of text (PaddleOCR's models via RapidOCR,
             run on OpenVINO: same engine as the detector, no second framework)
          -> keep text that looks like a kit label ("KIT-...")
          -> keep only labels NEAR THE TRAY (a ring around the bowl rim): a spare
             KIT-A label lying on the bench must never pass a KIT-B tray
          -> compare with the PLC's kit

    one label, matches the PLC        -> fine (no effect on the decision)
    one label, a DIFFERENT known kit  -> FAIL  "label says KIT-B, PLC expects KIT-A"  (wrong kit on the line)
    none / two different / unreadable -> REVIEW (can't confirm which kit this is)

Why OCR of plain text (and not a QR code): the check is "does the label an operator
reads match the order". A QR code identifies a carrier; a serial number adds nothing
unless the station knows which tray to expect. Why the full frame: the tray moves
and rotates, so a fixed crop would miss the label; the OCR's own text detector
finds text anywhere and at any angle (an angle classifier handles upside down).

Printed letters can look like fasteners to the part detector. Only the confirmed
carrier label can exclude a detection. Overlap extending onto the parts floor or a confident localized foreign-object
detection needs an operator to distinguish the paper from debris.
"""

import math
import re

import cv2

LABEL_PREFIX = "KIT"
LOOKALIKES = str.maketrans({"1": "I", "L": "I", "|": "I", "!": "I", "0": "O"})   # only fixed in the "KIT" part


def make_reader(threads=2):
    """The OCR, loaded once (models download on first use, ~15 MB)."""
    from rapidocr import EngineType, RapidOCR

    # Limit text search resolution. Recognition crops still use the original frame.

    ov = EngineType.OPENVINO
    return RapidOCR(params={"Det.engine_type": ov, "Det.limit_type": "max", "Det.limit_side_len": 1024, "Cls.engine_type": ov, "Rec.engine_type": ov,
                            "EngineConfig.openvino.inference_num_threads": threads,
                            "Global.log_level": "error"})


def read_orientation(reader, frame, turned=False):
    """Read once and return text polygons in original camera coordinates."""
    image = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) if turned else frame
    result = reader(image)
    if result.boxes is None:
        return []
    texts = []
    for box, text, score in zip(result.boxes, result.txts, result.scores):
        points = [[round(float(x)), round(float(y))] for x, y in box]
        if turned:
            points = [[y, frame.shape[0] - 1 - x] for x, y in points]
        texts.append({"text": text, "conf": round(float(score), 3), "box": points,
                      "center": [round(sum(axis) / 4) for axis in zip(*points)]})
    return texts


def inspect_label(reader, frame, rim, expected, kits, min_conf=0.8, near=(0.9, 1.8)):
    """Retry an unresolved carrier read. A known mismatch is already decisive."""
    texts = read_orientation(reader, frame) if rim is not None else []
    verdict = check_label(texts, rim, expected, kits, min_conf, near)
    passes = int(rim is not None)
    if rim is not None and verdict["status"] == "review" and not verdict["conflict"]:
        texts.extend(read_orientation(reader, frame, turned=True))
        verdict = check_label(texts, rim, expected, kits, min_conf, near)
        passes += 1
    return texts, dict(verdict, passes=passes)


def normalize(text):
    """'kit-b' / 'KIT B' / 'K1T-B' -> 'KITB'. Look-alike characters are fixed in the prefix only:
    the kit letter itself is never guessed (KIT-8 is not KIT-B)."""
    s = re.sub(r"[^A-Z0-9|!]", "", text.upper())
    return s[:3].translate(LOOKALIKES) + s[3:]


def label_overlap(box, quads, margin=15, share=0.6, label_size=6.0):
    """Is this detection box [x1, y1, x2, y2] the kit label, not a part? Two ways the detector gets fooled:
      its LETTERS: at least `share` of the box lies on the label's text (grown by margin px)
                   ("I"/"T" read as a bolt, "B" as a nut);
      the WHOLE LABEL: the box holds most of the label's text and is no more than label_size
                   times the text's area (the white paper around the text: "other 0.78").
    A part lying next to a label is neither: it doesn't sit on the text or contain it."""
    from shapely.geometry import Polygon, box as rect

    b = rect(*box)
    if b.area <= 0:
        return None
    for q in quads:
        text = Polygon(q)
        if text.area and text.intersection(b).area >= 0.8 * text.area and b.area <= label_size * text.area:
            return "paper"
        if text.buffer(margin).intersection(b).area >= share * b.area:
            return "text"
    return None


def mark_label_detections(detections, label, floor, high):
    """Exclude confirmed carrier printing from counts and flag ambiguous overlaps for review."""
    from src.inspection.tray import in_bowl

    for d in detections:
        overlap = label_overlap(d.box, label["quads"]) if d.in_bowl else None
        if not overlap:
            continue
        x1, y1, x2, y2 = d.box
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        on_floor = floor is None or not floor[3] or in_bowl(center, floor, margin=1)
        if on_floor or (overlap == "text" and d.cls == "other" and d.conf >= high):
            label["object_overlap"] = True
            d.label_ambiguous = True
        d.in_bowl, d.on_label = False, True


def check_label(texts, rim, expected, kits, min_conf=0.8, near=(0.9, 1.8)):
    """Decide what the label says about this tray.

    texts: read_orientation() output. rim: (x, y, r) of the bowl rim, or None.
    expected: the PLC's kit name. kits: kits.yaml names.
    near: a label counts as ON the tray if its centre is between near[0] and near[1]
          rim radii from the bowl's centre (the label tab sticks out past the rim).
    Returns dict(status = ok | mismatch | review, reason, label, ignored)."""
    known = {normalize(k): k for k in kits}
    labels = [t for t in texts if normalize(t["text"]).startswith(LABEL_PREFIX)]
    on_tray, ignored = [], []
    for t in labels:
        if rim is not None and near[0] * rim[2] <= math.dist(t["center"], rim[:2]) <= near[1] * rim[2]:
            on_tray.append(t)
        else:
            ignored.append(t["text"])
    out = {"label": None, "ignored": ignored, "quads": [], "conflict": False}
    if rim is None:
        return out | {"status": "review", "reason": "kit label: no tray found to look around"}
    if not on_tray:
        return out | {"status": "review", "reason": "no kit label found on the tray"}
    names = {known.get(normalize(t["text"])) for t in on_tray}
    if len(on_tray) > 1 and len(names) > 1:
        return out | {"status": "review", "conflict": True,
                      "reason": "conflicting kit label reads on the tray: " + ", ".join(t["text"] for t in on_tray)}
    best = max(on_tray, key=lambda t: t["conf"])
    name = known.get(normalize(best["text"]))
    out["label"] = best
    if name is None or best["conf"] < min_conf:
        return out | {"status": "review", "reason": f"kit label unreadable ('{best['text']}', {best['conf']:.2f})"}
    out["quads"] = [best["box"]]
    if name != expected:
        return out | {"status": "mismatch", "reason": f"label says {name}, PLC expects {expected}"}
    return out | {"status": "ok", "reason": f"label {name} matches"}


def main():
    """See what the OCR reads: every text found, the "on the tray" ring, and the verdict against a kit."""
    import argparse
    from pathlib import Path

    import cv2
    import numpy as np

    from src.capture.camera import quiet_jpeg_warnings
    from src.inspection.pipeline import load_settings
    from src.inspection.scene import check_scene

    parser = argparse.ArgumentParser(description="Read the kit label on the tray (camera snapshot or a photo)")
    parser.add_argument("--image", help="a photo instead of the camera")
    parser.add_argument("--kit", default="KIT-A", help="the kit the PLC would expect")
    parser.add_argument("--out", default="reports/kit_label/label_check.jpg")
    args = parser.parse_args()
    quiet_jpeg_warnings()
    cfg, kits = load_settings()
    lab = cfg["label"]
    if args.image:
        frame = cv2.imread(args.image)
    else:
        from src.capture.camera import load_camera_config, open_camera
        cap = open_camera(load_camera_config())
        for _ in range(15):                       # let exposure settle
            ok, frame = cap.read()
        cap.release()
    rim = check_scene(frame, [], **cfg["scene"])["rim"]
    texts, result = inspect_label(make_reader(lab.get("threads", 2)), frame, rim, args.kit, kits,
                                 lab["min_conf"], tuple(lab["near"]))

    img = frame.copy()
    if rim is not None:
        x, y, r = rim
        for k in lab["near"]:
            cv2.circle(img, (int(x), int(y)), int(k * r), (255, 160, 0), 2)
    print(f"rim: {rim}")
    for t in texts:
        d = math.dist(t["center"], rim[:2]) / rim[2] if rim is not None else float("nan")
        print(f"  read '{t['text']}'  conf {t['conf']:.2f}  at {d:.2f} rim radii")
        cv2.polylines(img, [np.array(t["box"], np.int32)], True, (0, 200, 255), 2)
        cv2.putText(img, f"{t['text']} {t['conf']:.2f} @{d:.2f}r", tuple(t["box"][0]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)
    print(f"PLC expects {args.kit} -> {result['status'].upper()}: {result['reason']}"
          + (f"  (ignored, off the tray: {result['ignored']})" if result["ignored"] else ""))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, img)
    if not args.image:                            # the raw photo too, to re-run with --image (never the drawn one)
        raw = str(Path(args.out).with_name(Path(args.out).stem + "_raw.jpg"))
        cv2.imwrite(raw, frame)
        print(f"raw photo: {raw}")
    print(f"picture: {args.out}  (blue rings = where a label counts as ON the tray)")


if __name__ == "__main__":
    main()
