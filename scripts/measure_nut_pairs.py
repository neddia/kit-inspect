"""Measure the visible face of the swapped nut in pairs 7-11 of size_pairs01.

Run from kit-inspect with uv run python scripts/measure_nut_pairs.py.
The manually marked hole centers isolate measurement from part detection.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SESSION = "size_pairs01"
# Original 1280 x 720 frame coordinates, regular then small for each pair.
CENTERS = [
    ((599, 265), (597, 252)),
    ((799, 232), (810, 321)),
    ((315, 385), (320, 368)),
    ((603, 579), (645, 587)),
    ((592, 283), (576, 309)),
]


def measure(frame, center):
    """Find the strongest bright-to-dark transition along each ray from the hole."""
    angles = np.arange(360) * np.pi / 180
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    radii = np.arange(6, 55)
    points = (center + directions[:, None, :] * radii[None, :, None]).astype(np.float32)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (5, 5), 1)
    samples = cv2.remap(smooth, points[:, :, 0], points[:, :, 1], cv2.INTER_LINEAR)
    drops = samples[:, :-2].astype(float) - samples[:, 2:]
    edges = points[np.arange(360), np.argmax(drops, axis=1) + 1]
    spans = np.linalg.norm(edges[:180] - edges[180:], axis=1)
    # Median across directions limits the effect of glare or a nearby screw.
    width = np.median(spans)
    axis = np.argmin(np.abs(spans - width))
    return width, np.rint(edges).astype(int), axis


def main():
    out = ROOT / "reports" / "size_pairs01_geometry"
    out.mkdir(parents=True, exist_ok=True)
    panels = []
    with (out / "nut_widths.csv").open("w", newline="") as table:
        writer = csv.writer(table)
        writer.writerow(["pair", "variant", "image", "visible_face_span_px"])
        for pair, centers in enumerate(CENTERS, 7):
            row = []
            for offset, center in enumerate(centers):
                image = f"{SESSION}_{2 * pair - 1 + offset:04}.jpg"
                frame = cv2.imread(str(ROOT / "data/images" / SESSION / image))
                width, edges, axis = measure(frame, center)
                variant = ("regular", "small")[offset]
                writer.writerow([pair, variant, image, round(width, 1)])
                for point in edges:
                    cv2.circle(frame, tuple(point), 0, (60, 240, 60), 1)
                cv2.line(frame, tuple(edges[axis]), tuple(edges[axis + 180]), (255, 190, 0), 1)
                x, y = center
                crop = cv2.resize(frame[y-60:y+60, x-70:x+70], (280, 240))
                cv2.rectangle(crop, (0, 0), (279, 26), (25, 25, 25), -1)
                cv2.putText(crop, f"Pair {pair}  {variant}  {width:.0f} px", (7, 19),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                row.append(crop)
            panels.append(np.hstack(row))
    cv2.imwrite(str(out / "nut_pairs.jpg"), np.vstack(panels))


if __name__ == "__main__":
    main()
