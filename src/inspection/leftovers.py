"""Unknown-object check (OpenCV): anything on the flat floor the detector didn't explain.

The detector only knows bolt, nut and screw. A coin, thread or hair tie on a
complete kit leaves the counts right, so the counts alone would PASS it. This
finds "stuff" on the floor and removes everything the detector boxed; whatever
is left is something unknown -> REVIEW.

Steps:
  1. Floor only: inside the measured flat-floor circle (tray.find_floor).
     Outside it the bowl curves up and its shading looks like objects.
  2. "Something here" if a pixel is
       darker than its own neighbourhood  (black-hat: smooth lighting cancels out), or
       strongly coloured                  (saturation: catches e.g. a red thread)
  3. Blank every detected part's box plus a margin (their shadows reach past the box).
  4. Group what's left into blobs; any blob of at least min_area pixels is an unknown object.

Tuned on all train + val images (Nov 17): catches 23 of 26 foreign objects,
false alarms on 4% of other trays. Known misses: a white disc on the white floor
(no contrast: a tray-colour problem), and a small object inside crowded part
boxes (blanking whole boxes hides it). Objects on the sloped wall aren't checked.
Nov 17, live station: a pale green bead (saturation ~55) was caught in only 3 of 7
photos at saturation 60; at 45 all 7, with no new false alarms (white floor: 22-34).
Big objects need colour or the darker test misses them: black-hat only finds things
smaller than its neighbourhood window.
"""

import cv2
import numpy as np

DEFAULTS = dict(neighbourhood=82, darker_by=35, saturation=45, box_margin=20, min_area=150)


def find_leftovers(frame, floor, boxes, neighbourhood=82, darker_by=35, saturation=45, box_margin=20, min_area=150):
    """List of unknown objects on the floor: dicts with box [x1, y1, x2, y2] and area (pixels)."""
    h, w = frame.shape[:2]
    on_floor = np.zeros((h, w), np.uint8)
    cv2.circle(on_floor, (int(floor[0]), int(floor[1])), int(floor[2]), 1, -1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    half = cv2.resize(gray, (w // 2, h // 2))  # black-hat at half size: faster, same result
    k = max(3, neighbourhood // 2) | 1
    blackhat = cv2.morphologyEx(half, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    darker = cv2.resize(blackhat, (w, h)) > darker_by
    coloured = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[..., 1] > saturation
    stuff = (darker | coloured) & (on_floor > 0)

    for x1, y1, x2, y2 in boxes:
        stuff[max(0, int(y1) - box_margin):int(y2) + box_margin, max(0, int(x1) - box_margin):int(x2) + box_margin] = False

    stuff = cv2.morphologyEx(stuff.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # drop specks
    n, _, stats, _ = cv2.connectedComponentsWithStats(stuff)
    return [{"box": [int(x), int(y), int(x + bw), int(y + bh)], "area": int(area)}
            for x, y, bw, bh, area in stats[1:n] if area >= min_area]
