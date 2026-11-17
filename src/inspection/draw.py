"""Draw an inspection result on its photo: the floor circle, boxes, unknown objects, verdict and reasons.

Used by decision_eval (report pictures), the station's --show window and the dashboard.
"""

import cv2

COLORS = {"PASS": (60, 200, 60), "FAIL": (60, 60, 230), "REVIEW": (0, 200, 255)}


def draw(frame, out, high=0.7):
    """Grey box = outside the bowl, green = sure (>= high), orange = unsure, magenta = unknown object."""
    img = frame.copy()
    scene = out["scene"]
    if scene["floor"] is not None:
        x, y, r = scene["floor"][:3]
        cv2.circle(img, (int(x), int(y)), int(r), (255, 160, 0), 1)
    for d in out["detections"]:
        x1, y1, x2, y2 = map(int, d.box)
        color = (150, 150, 150) if not d.in_bowl else (60, 200, 60) if d.conf >= high else (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for u in out["unknown_objects"]:
        x1, y1, x2, y2 = u["box"]
        cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 0, 255), 2)
    dec = out["decision"]
    cv2.rectangle(img, (0, 0), (img.shape[1], 34 + 22 * len(dec.reasons)), (0, 0, 0), -1)
    cv2.putText(img, dec.result, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[dec.result], 2, cv2.LINE_AA)
    for i, reason in enumerate(dec.reasons):
        cv2.putText(img, reason, (8, 52 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img
