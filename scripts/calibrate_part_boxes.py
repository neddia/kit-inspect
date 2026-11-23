"""Replay reviewed training frames and the size challenge. Run as a module from kit-inspect."""
import csv
import json
from math import hypot

from ultralytics import YOLO

from src.inspection.decision import iou
from src.inspection.pipeline import REPO, load_settings


def main():
    cfg, _ = load_settings()
    model = YOLO(str(REPO / cfg["model"]), task="detect")
    labels = REPO / "data/labels/reviewed"
    images = REPO / "data/images"
    with (REPO / "data/manifest.csv").open(newline="") as manifest:
        stacked = {r["image"] for r in csv.DictReader(manifest) if r["scenario"] == "stacked"}
    inputs = []
    for path in sorted(labels.glob("train*/*.json")):
        annotation = json.loads(path.read_text())
        inputs.append((annotation["image"], annotation["objects"]))
    for path in sorted((images / "size_pairs01").glob("*.jpg")):
        inputs.append((str(path.relative_to(images)), None))
    out = REPO / "reports/size_pairs01_geometry/part_box_calibration.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image", "part", "sample", "confidence", "cx", "cy", "long_px", "diagonal_px"])
        for name, objects in inputs:
            result = model.predict(str(images / name), conf=cfg["decision"]["low"],
                                   imgsz=cfg["imgsz"], verbose=False)[0]
            for box, c, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                    result.boxes.conf.tolist()):
                part = model.names[int(c)]
                if part not in ("screw", "nut"):
                    continue
                sample = "challenge"
                if objects is not None:
                    matches = [o for o in objects if o["cls"] == part and o["inside"]]
                    match = max(matches, key=lambda o: iou(box, o["box"]), default=None)
                    if match is None or iou(box, match["box"]) < 0.7:
                        continue
                    sample = "isolated" if all(o is match or iou(match["box"], o["box"]) == 0
                                               for o in objects) else "overlap"
                    if name in stacked:
                        sample = "stacked_scene"
                x, y, X, Y = box
                writer.writerow([name, part, sample, round(conf, 3), round((X+x)/2, 1),
                                 round((Y+y)/2, 1), round(max(X-x, Y-y), 2), round(hypot(X-x, Y-y), 2)])


if __name__ == "__main__":
    main()
