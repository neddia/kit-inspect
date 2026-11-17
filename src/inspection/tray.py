"""Find the bowl in the photo: plain OpenCV, no model.

Two circles, because the bowl is 3D:
  rim    the top of the inner wall: a sharp edge, found every time (HoughCircles).
         Used for "is a part in the bowl?" (with a margin: parts can rest on the wall).
  floor  the edge of the FLAT floor, where it starts curving up into the wall.
         Inside it there are no harsh shadows, so the unknown-object check runs there.
         The rim and floor are NOT concentric in the photo (parallax: the rim is
         higher, so a camera that isn't exactly overhead sees them shifted, up to
         ~60 px), and the floor/rim size ratio varies by session (0.74-0.82). So the
         floor is measured in each photo, not guessed from the rim. (Bob's idea, Nov 17.)

How the floor is found: from the rim's centre, walk outward along 90 spokes.
On each, note the floor's own brightness near the middle, then the FIRST point
where it gets clearly darker and stays darker: the floor edge. Spokes that hit a
part are skipped. Fit a circle through the edge points, drop the ones far off it,
refit. If too few spokes work (a big object over the edge), fall back to a
typical size.
"""

import cv2
import numpy as np

# Rim search runs at half size (faster, smoother); radii are half-size pixels.
# The rim is about 310-350 px radius at full size with zoom 2.
RIM_SEARCH = dict(dp=1.5, minDist=500, param1=60, param2=30, minRadius=110, maxRadius=175)
FLOOR_RATIO_OK = (0.68, 0.86)   # floor radius / rim radius we accept; measured 0.74-0.82
FLOOR_FALLBACK = 0.72           # used when the floor can't be measured (safely inside every session's floor)


def find_tray(frame):
    """(x, y, r) of the inner-rim circle in full-size pixels, or None if no bowl is found."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (w // 2, h // 2))
    gray = cv2.medianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, **RIM_SEARCH)
    if circles is None:
        return None
    x, y, r = (float(v) * 2 for v in circles[0][0])
    return x, y, r


def fit_circle(points):
    """Least-squares circle through (x, y) points -> (cx, cy, r)."""
    x, y = points[:, 0], points[:, 1]
    a, b, c = np.linalg.lstsq(np.column_stack([x, y, np.ones_like(x)]), x ** 2 + y ** 2, rcond=None)[0]
    cx, cy = a / 2, b / 2
    return float(cx), float(cy), float(np.sqrt(c + cx ** 2 + cy ** 2))


def _in_any_box(px, py, boxes, pad):
    return any(x1 - pad <= px <= x2 + pad and y1 - pad <= py <= y2 + pad for x1, y1, x2, y2 in boxes)


def floor_edge_points(frame, rim, boxes=(), spokes=90, drop=10, pad=15):
    """Edge points of the flat floor, one per usable spoke."""
    h, w = frame.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (15, 15), 0).astype(np.float32)
    cx, cy, r = rim
    radii = np.arange(0.35 * r, 1.0 * r)
    points = []
    for angle in np.linspace(0, 2 * np.pi, spokes, endpoint=False):
        xs, ys = cx + radii * np.cos(angle), cy + radii * np.sin(angle)
        if (((xs >= 0) & (xs < w - 1) & (ys >= 0) & (ys < h - 1)).mean()) < 0.9:
            continue  # spoke runs off the photo
        line = cv2.remap(gray, xs.astype(np.float32)[None], ys.astype(np.float32)[None], cv2.INTER_LINEAR)[0]
        free = np.array([not _in_any_box(x, y, boxes, pad) for x, y in zip(xs, ys)])
        inner = (radii < 0.6 * r) & free
        if inner.sum() < 10:
            continue
        floor_level = np.median(line[inner])  # this spoke's flat-floor brightness
        darker = (line < floor_level - drop) & free & (radii > 0.6 * r)
        stays = np.flatnonzero(np.convolve(darker.astype(int), np.ones(10, int), mode="valid") == 10)
        if len(stays) == 0:
            continue
        i = stays[0]  # FIRST point where it's darker and stays darker for 10 px
        if not free[max(0, i - 15):i + 15].all():
            continue  # a part sits right on the edge here
        points.append((xs[i], ys[i]))
    return np.array(points).reshape(-1, 2)


def find_floor(frame, rim, boxes=()):
    """(x, y, r, measured) of the flat floor. measured=False means the fallback size was used."""
    points = floor_edge_points(frame, rim, boxes)
    if len(points) >= 12:
        keep = np.ones(len(points), bool)
        for _ in range(4):  # fit, drop points far off the circle, refit
            fx, fy, fr = fit_circle(points[keep])
            off = np.abs(np.hypot(points[:, 0] - fx, points[:, 1] - fy) - fr)
            keep = off < max(5.0, np.median(off[keep]) * 2.5)
        fx, fy, fr = fit_circle(points[keep])
        if FLOOR_RATIO_OK[0] < fr / rim[2] < FLOOR_RATIO_OK[1]:
            return fx, fy, fr, True
    return rim[0], rim[1], rim[2] * FLOOR_FALLBACK, False


def in_bowl(center, rim, margin=1.15):
    """Is a part's centre in the bowl? margin > 1 because parts can rest on the sloped wall."""
    if rim is None:
        return True
    return bool(np.hypot(center[0] - rim[0], center[1] - rim[1]) < rim[2] * margin)
