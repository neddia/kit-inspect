"""Find the bowl: the circle where its flat floor meets the wall.

Plain OpenCV, no model. Used by the auto-labeller to tell parts IN the tray
from spares lying on the desk, and later by the station's scene check.
Tested Nov 16 on all 278 capture images: found in every one.
"""

import cv2

# Searched at half size (faster, and smooths out noise); radii are half-size pixels.
# The floor circle is about 310-350 px radius at full size with zoom 2.
SEARCH = dict(dp=1.5, minDist=500, param1=60, param2=30, minRadius=110, maxRadius=175)


def find_tray(frame):
    """(x, y, r) of the floor circle in full-size pixels, or None if no circle is found."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **SEARCH)
    if circles is None:
        return None
    x, y, r = (float(v) * 2 for v in circles[0][0])
    return x, y, r
