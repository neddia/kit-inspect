"""Replay the detector-based size experiment with python -m scripts.inspect_detected_nuts."""
import csv

import cv2
import numpy as np

from scripts.measure_nut_pairs import CENTERS, ROOT, SESSION, measure
from src.inspection.pipeline import Inspector


def pair9_crop(frame, center, edges, axis, label):
    x, y = np.rint(center).astype(int)
    origin = np.array([x - 64, y - 64])
    raw = frame[y - 64:y + 64, x - 64:x + 64]
    overlay = raw.copy()
    local = edges - origin
    for point in local:
        cv2.circle(overlay, tuple(point), 0, (60, 240, 60), 1)
    cv2.line(overlay, tuple(local[axis]), tuple(local[axis + 180]), (255, 190, 0), 1)
    crop = cv2.resize(np.hstack([raw, overlay]), (384, 192))
    caption = np.zeros((30, crop.shape[1], 3), np.uint8)
    cv2.putText(caption, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX,
                .55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([caption, crop])


def main():
    inspector = Inspector()
    out = ROOT / "reports" / "size_pairs01_geometry"
    out.mkdir(parents=True, exist_ok=True)
    panels = []
    with (out / "detected_nuts.csv").open("w", newline="") as table:
        writer = csv.writer(table)
        writer.writerow(["pair", "image", "x_center", "y_center", "is_target", "variant",
                         "visible_face_span_px"])
        for pair, targets in enumerate(CENTERS, 7):
            for offset, target in enumerate(targets):
                name = f"{SESSION}_{2 * pair - 1 + offset:04}.jpg"
                frame = cv2.imread(str(ROOT / "data/images" / SESSION / name))
                boxes = inspector.model.predict(frame, conf=inspector.cfg["decision"]["low"],
                                                imgsz=inspector.cfg["imgsz"], verbose=False)[0].boxes
                crops = []
                for box, cls in zip(boxes.xyxy.tolist(), boxes.cls.tolist()):
                    if inspector.model.names[int(cls)] != "nut":
                        continue
                    center = np.array(box).reshape(2, 2).mean(axis=0)
                    is_target = np.linalg.norm(center - target) < 12
                    variant = "small" if is_target and offset else "regular"
                    width, edges, axis = measure(frame, center)
                    writer.writerow([pair, name, *center.round(1), is_target, variant, round(width, 1)])
                    if pair == 9:
                        label = f"{name[-8:-4]}  {variant}  {width:.1f} px"
                        crops.append((center[0], pair9_crop(frame, center, edges, axis, label)))
                if crops:
                    panels.append(np.hstack([crop for _, crop in sorted(crops, key=lambda c: c[0])]))
    cv2.imwrite(str(out / "pair9_detected_nuts.jpg"), np.vstack(panels))


if __name__ == "__main__":
    main()
