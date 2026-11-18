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

The label's own letters fool the detector (Nov 18, live): "I"/"T" read as bolts
(0.25-0.65), "B" as a nut, the whole label as "other". The label sits on the bowl
wall, inside the counted area, so perfect kits went to REVIEW. Fix: a detection
lying mostly on a kit-label text box is lettering, not a part (on_label). If the
label can't be read at all, its letters still count -> REVIEW, the safe direction.

Measured Nov 18 on fake labels pasted on a real tray photo (0-180 deg): read
every time, confidence 1.0, ~135 ms per 720p frame on 2 CPU cores.
Real printed labels (glare, perspective, focus) still to be tested.
"""

import math
import re

LABEL_PREFIX = "KIT"
LOOKALIKES = str.maketrans({"1": "I", "L": "I", "|": "I", "!": "I", "0": "O"})   # only fixed in the "KIT" part


def make_reader(threads=2):
    """The OCR, loaded once (models download on first use, ~15 MB)."""
    from rapidocr import EngineType, RapidOCR

    ov = EngineType.OPENVINO
    return RapidOCR(params={"Det.engine_type": ov, "Cls.engine_type": ov, "Rec.engine_type": ov,
                            "EngineConfig.openvino.inference_num_threads": threads,
                            "Global.log_level": "error"})


def read_text(reader, frame):
    """Every piece of text in the photo: dicts with text, conf, center (x, y), box (4 corner points)."""
    r = reader(frame)
    if r.boxes is None:
        return []
    return [{"text": t, "conf": round(float(s), 3), "center": [round(float(v)) for v in b.mean(0)],
             "box": [[round(float(x)), round(float(y))] for x, y in b]}
            for b, t, s in zip(r.boxes, r.txts, r.scores)]


def normalize(text):
    """'kit-b' / 'KIT B' / 'K1T-B' -> 'KITB'. Look-alike characters are fixed in the prefix only:
    the kit letter itself is never guessed (KIT-8 is not KIT-B)."""
    s = re.sub(r"[^A-Z0-9|!]", "", text.upper())
    return s[:3].translate(LOOKALIKES) + s[3:]


def on_label(box, quads, margin=15, share=0.6):
    """True if at least `share` of a detection box [x1, y1, x2, y2] lies on a kit label (grown by margin px)."""
    from shapely.geometry import Polygon, box as rect

    b = rect(*box)
    return b.area > 0 and any(Polygon(q).buffer(margin).intersection(b).area >= share * b.area for q in quads)


def check_label(texts, rim, expected, kits, min_conf=0.8, near=(0.9, 1.8)):
    """Decide what the label says about this tray.

    texts: read_text() output. rim: (x, y, r) of the bowl rim, or None.
    expected: the PLC's kit name. kits: kits.yaml names.
    near: a label counts as ON the tray if its centre is between near[0] and near[1]
          rim radii from the bowl's centre (the label tab sticks out past the rim).
    Returns dict(status = ok | mismatch | review, reason, label, ignored)."""
    known = {normalize(k): k for k in kits}
    labels = [t for t in texts if normalize(t["text"]).startswith(LABEL_PREFIX)]
    quads = [t["box"] for t in labels]            # where kit-label lettering is (on_label uses them)
    on_tray, ignored = [], []
    for t in labels:
        if rim is not None and near[0] * rim[2] <= math.dist(t["center"], rim[:2]) <= near[1] * rim[2]:
            on_tray.append(t)
        else:
            ignored.append(t["text"])
    out = {"label": None, "ignored": ignored, "quads": quads}
    if rim is None:
        return out | {"status": "review", "reason": "kit label: no tray found to look around"}
    if not on_tray:
        return out | {"status": "review", "reason": "no kit label found on the tray"}
    names = {known.get(normalize(t["text"])) for t in on_tray}
    if len(on_tray) > 1 and len(names) > 1:
        return out | {"status": "review",
                      "reason": "two different kit labels on the tray: " + ", ".join(t["text"] for t in on_tray)}
    best = max(on_tray, key=lambda t: t["conf"])
    name = known.get(normalize(best["text"]))
    out["label"] = best
    if name is None or best["conf"] < min_conf:
        return out | {"status": "review", "reason": f"kit label unreadable ('{best['text']}', {best['conf']:.2f})"}
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
    parser.add_argument("--out", default="reports/stage7/label_check.jpg")
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
    texts = read_text(make_reader(lab.get("threads", 2)), frame)
    result = check_label(texts, rim, args.kit, kits, lab["min_conf"], tuple(lab["near"]))

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
