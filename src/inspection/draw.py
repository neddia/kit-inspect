"""Draw an inspection result on its photo: the floor circle, boxes, unknown objects, verdict and reasons.

Used by decision_eval (report pictures), the station's --show window and the dashboard.
The verdict and reasons go in a black strip ABOVE the photo, so they never cover the tray.
"""

import cv2
import numpy as np

COLORS = {"PASS": (60, 200, 60), "FAIL": (60, 60, 230), "REVIEW": (0, 200, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def tag(img, text, x, y, color, scale, thick):
    """Text on a filled label just above (x, y); inside the box instead if there's no room above."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    h = th + base + 6
    top = y - h if y - h >= 0 else y
    cv2.rectangle(img, (x, top), (x + tw + 8, top + h), color, -1)
    cv2.putText(img, text, (x + 4, top + th + 3), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw(frame, out, high=0.7):
    """Grey box = outside the bowl, green = sure (>= high), orange = unsure,
    magenta = foreign object (the model's "other" class, or the OpenCV check's "unknown")."""
    img = frame.copy()
    k = img.shape[1] / 1280                       # sizes are tuned for 1280-wide photos
    scale, thick = 0.75 * k, max(1, round(2 * k))
    scene = out["scene"]
    if scene["floor"] is not None:
        x, y, r = scene["floor"][:3]
        cv2.circle(img, (int(x), int(y)), int(r), (255, 160, 0), thick)
    for d in out["detections"]:
        x1, y1, x2, y2 = map(int, d.box)
        color = ((150, 150, 150) if not d.in_bowl else (255, 0, 255) if d.cls == "other"
                 else (60, 200, 60) if d.conf >= high else (0, 200, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        tag(img, f"{d.cls} {d.conf:.2f}" + (" (label)" if d.on_label else ""), x1, y1, color, scale, thick)
    for u in out["unknown_objects"]:
        x1, y1, x2, y2 = u["box"]
        cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 0, 255), thick + 1)
        tag(img, "unknown", x1 - 6, y1 - 6, (255, 0, 255), scale, thick)
    label = out.get("label")
    if label and label["label"]:                  # the kit label the OCR read on the tray
        color = {"ok": (60, 200, 60), "mismatch": (60, 60, 230)}.get(label["status"], (0, 200, 255))
        quad = np.array(label["label"]["box"], np.int32)
        cv2.polylines(img, [quad], True, color, thick + 1)
        x, y = quad.min(0)
        tag(img, f"label {label['label']['text']} {label['label']['conf']:.2f}", int(x), int(y), color, scale, thick)

    dec = out["decision"]
    line = int(30 * k)
    strip = int(44 * k) + line * len(dec.reasons)
    img = cv2.copyMakeBorder(img, strip, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(img, dec.result, (int(10 * k), int(34 * k)), FONT, 1.2 * k, COLORS[dec.result], thick + 1, cv2.LINE_AA)
    for i, reason in enumerate(dec.reasons):
        cv2.putText(img, reason, (int(10 * k), int(44 * k) + line * (i + 1) - int(8 * k)), FONT, 0.8 * k,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return img
