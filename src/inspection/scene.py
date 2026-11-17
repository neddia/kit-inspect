"""Scene check: is this a photo the model can be trusted on? Runs BEFORE counting.

Problems found here send the tray to REVIEW, because the model was never trained
on such scenes and its counts can't be trusted:
  - no bowl found (no tray, hand over it, lens covered)
  - floor brightness outside what the training sessions covered
    (measured Nov 17 over all 278 train/val images: 150-248; limits leave a margin)
  - bowl pushed so far that part of the floor is out of the photo

Not checked: blur. An empty tray has almost no edges, so it scores as "blurry"
as a truly blurred photo; a sharpness rule would fire on every empty tray.
"""

import cv2
import numpy as np

from src.inspection.tray import find_floor, find_tray

DEFAULTS = dict(min_brightness=140, max_brightness=252, max_floor_outside=0.10)


def check_scene(frame, boxes=(), min_brightness=140, max_brightness=252, max_floor_outside=0.10):
    """-> dict(rim, floor, brightness, problems). problems is a list of plain-English reasons."""
    problems = []
    rim = find_tray(frame)
    if rim is None:
        return {"rim": None, "floor": None, "brightness": None, "problems": ["no tray found"]}

    floor = find_floor(frame, rim, boxes)
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(floor[0]), int(floor[1])), int(floor[2]), 1, -1)
    floor_pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[mask > 0]
    brightness = float(np.median(floor_pixels)) if floor_pixels.size else None

    if brightness is None or brightness < min_brightness:
        problems.append(f"too dark (floor brightness {brightness or 0:.0f}, normal {min_brightness}+)")
    elif brightness > max_brightness:
        problems.append(f"too bright (floor brightness {brightness:.0f}, normal up to {max_brightness})")

    x, y, r = floor[:3]
    outside = max(0.0, r - x, x + r - w, r - y, y + r - h) / r  # how far the floor circle sticks out of the photo
    if outside > max_floor_outside:
        problems.append(f"tray off position ({outside:.0%} of the floor is outside the photo)")

    return {"rim": rim, "floor": floor, "brightness": brightness, "problems": problems}
