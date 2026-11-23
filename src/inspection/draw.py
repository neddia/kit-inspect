"""Draw an inspection result on its photo: the floor circle, boxes, unknown objects, verdict and reasons.

Used by decision_eval (report pictures), the station's --show window and the dashboard.
The verdict and reasons go in a black strip ABOVE the photo, so they never cover the tray.
"""

import cv2
import numpy as np

COLORS = {"PASS": (60, 200, 60), "FAIL": (60, 60, 230), "REVIEW": (0, 200, 255)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def tag(img, text, x, y, color, scale, thick, occupied=None):
    """Text on a filled label just above (x, y); inside the box instead if there's no room above."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    h = th + base + 6
    top = y - h if y - h >= 0 else y
    if occupied is not None:
        w = tw + 8
        x = max(0, min(x, img.shape[1] - w))
        for offset in range(0, img.shape[0], h + 2):
            slots = [top - offset, top + offset]
            slot = next((t for t in slots if 0 <= t <= img.shape[0] - h and
                         all(x + w < l or x > r or t + h < u or t > b for l, u, r, b in occupied)), None)
            if slot is not None:
                top = slot
                break
        occupied.append((x, top, x + w, top + h))
    cv2.rectangle(img, (x, top), (x + tw + 8, top + h), color, -1)
    cv2.putText(img, text, (x + 4, top + th + 3), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)


def draw(frame, out, high=0.7, operator=False):
    """Grey box = outside the bowl, green = sure (>= high), orange = unsure,
    magenta = foreign object (the model's "other" class, or the OpenCV check's "unknown")."""
    img = frame.copy()
    k = img.shape[1] / 1280                       # sizes are tuned for 1280-wide photos
    scale, thick = 0.75 * k, max(1, round(2 * k))
    scene = out["scene"]
    if scene["floor"] is not None:
        x, y, r = scene["floor"][:3]
        cv2.circle(img, (int(x), int(y)), int(r), (255, 160, 0), thick)
    # Draw geometry before flipping the operator image; add upright text afterward.
    captions = []
    detections = out["detections"]
    if operator:
        detections = sorted(detections, key=lambda d: d.box not in out["decision"].size_check_boxes)
    for d in detections:
        if not d.in_bowl and not d.label_ambiguous:
            continue
        x1, y1, x2, y2 = map(int, d.box)
        color = ((150, 150, 150) if not d.in_bowl else (255, 0, 255) if d.cls == "other"
                 else (60, 200, 60) if d.conf >= high else (0, 200, 255))
        check_size = d.box in out["decision"].size_check_boxes
        if check_size:
            color = COLORS["REVIEW"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick + int(check_size))
        caption = "check label area" if d.on_label else f"{d.cls} {d.conf:.2f}"
        if d.cls == "nut" and not d.on_label:
            caption += f" {d.longest_side_px:.0f}px"
        elif d.cls == "screw" and not d.on_label:
            caption += f" diag {d.diagonal_px:.0f}px"
        if operator:
            name = "label area" if d.on_label else d.cls
            caption = f"{name} {d.conf:.0%}"
        if check_size:
            caption = f"SIZE? {d.cls} {(d.longest_side_px if d.cls == 'nut' else d.diagonal_px):.1f}px"
            if operator:
                caption = f"{d.cls} {d.conf:.0%}  {(d.longest_side_px if d.cls == 'nut' else d.diagonal_px):.1f}px"
        captions.append((caption, x1, y1, x2, y2, color))
    for u in out["unknown_objects"]:
        x1, y1, x2, y2 = u["box"]
        cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 0, 255), thick + 1)
        captions.append(("unknown", x1 - 6, y1 - 6, x2 + 6, y2 + 6, (255, 0, 255)))
    label = out.get("label")
    if label and label["label"]:                  # the kit label the OCR read on the tray
        color = {"ok": (60, 200, 60), "mismatch": (60, 60, 230)}.get(label["status"], (0, 200, 255))
        quad = np.array(label["label"]["box"], np.int32)
        cv2.polylines(img, [quad], True, color, thick + 1)
        x, y = quad.min(0)
        captions.append((f"label {label['label']['text']} {label['label']['conf']:.2f}", int(x), int(y), int(quad[:, 0].max()), int(quad[:, 1].max()), color))

    if operator:
        img = cv2.rotate(img, cv2.ROTATE_180)
    occupied = [] if operator else None
    for caption, left, top, right, bottom, color in captions:
        x = img.shape[1] - 1 - right if operator else left
        y = img.shape[0] - 1 - bottom if operator else top
        tag(img, caption, x, y, color, scale, thick, occupied)

    dec = out["decision"]
    line = int(30 * k)
    strip = int(44 * k) + line * len(dec.reasons)
    img = cv2.copyMakeBorder(img, strip, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    cv2.putText(img, dec.result, (int(10 * k), int(34 * k)), FONT, 1.2 * k, COLORS[dec.result], thick + 1, cv2.LINE_AA)
    for i, reason in enumerate(dec.reasons):
        cv2.putText(img, reason, (int(10 * k), int(44 * k) + line * (i + 1) - int(8 * k)), FONT, 0.8 * k,
                    (255, 255, 255), thick, cv2.LINE_AA)
    return img
