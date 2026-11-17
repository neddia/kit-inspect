"""Stage 4 -> 5 bridge: write the reviewed labels in YOLO's training format.

YOLO wants, for every image, a .txt file with one line per part:
    class x_center y_center width height        (all four as fractions of the image size)
class numbers follow kits.yaml order: 0 bolt, 1 nut, 2 screw.

Layout written (rebuilt from scratch every run; it's derived data, not committed):
    data/yolo/images/train/<image>.jpg   symlinks to data/images/<session>/<image>.jpg
    data/yolo/labels/train/<image>.txt   YOLO finds labels by swapping images/ -> labels/
    data/yolo/images/val/...  labels/val/...
    data/yolo/data.yaml                  what you pass to YOLO: folders + class names

Every labelled part is exported, including parts outside the bowl (a bolt on
the desk is still a bolt). Images with no parts (empty tray) get an empty .txt,
which tells YOLO "nothing here": useful background examples.

Run:
    uv run python -m src.labeling.export_yolo
    uv run python -m src.labeling.export_yolo --check 3   # also draw 3 images back from the .txt files
"""

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import yaml

from src.capture.capture import DATA, MANIFEST
from src.labeling.auto_label import PARTS
from src.labeling.review import REVIEWED, label_path

YOLO = DATA / "yolo"


def yolo_line(cls, box, size):
    """'0 0.455078 0.262500 0.091406 0.286111' for a box [x1, y1, x2, y2] in a W x H image."""
    (x1, y1, x2, y2), (w, h) = box, size
    return f"{PARTS.index(cls)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"


def main():
    parser = argparse.ArgumentParser(description="Export reviewed labels to YOLO format")
    parser.add_argument("--check", type=int, default=0, help="draw N random images back from their .txt files")
    args = parser.parse_args()

    with MANIFEST.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val") and r["tier"] == "normal"]

    shutil.rmtree(YOLO, ignore_errors=True)
    counts = {"train": Counter(), "val": Counter()}
    images = {"train": 0, "val": 0}
    missing = []
    for row in rows:
        lab_file = label_path(REVIEWED, row["image"])
        if not lab_file.exists():
            missing.append(row["image"])
            continue
        label = json.loads(lab_file.read_text())
        split, name = row["split"], Path(row["image"]).name
        img_dir, lbl_dir = YOLO / "images" / split, YOLO / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / name).symlink_to((DATA / "images" / row["image"]).resolve())
        lines = [yolo_line(o["cls"], o["box"], label["size"]) for o in label["objects"]]
        (lbl_dir / name).with_suffix(".txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        counts[split].update(o["cls"] for o in label["objects"])
        images[split] += 1
    if missing:
        raise SystemExit(f"{len(missing)} images not reviewed yet, e.g. {missing[:3]}. Review first.")

    (YOLO / "data.yaml").write_text(yaml.safe_dump({
        "path": str(YOLO.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": dict(enumerate(PARTS)),
    }, sort_keys=False))

    for split in ("train", "val"):
        print(f"{split:<5} {images[split]:>3} images  " + "  ".join(f"{p} {counts[split][p]}" for p in PARTS))
    print(f"Wrote {YOLO.relative_to(DATA.parent)}/ and data.yaml")

    if args.check:
        # Round trip: read the .txt back and draw it. If the boxes sit on the parts, the maths is right.
        out_dir = DATA.parent / "reports" / "stage5" / "label_check"
        out_dir.mkdir(parents=True, exist_ok=True)
        colors = [(40, 40, 255), (40, 200, 40), (255, 140, 0)]
        for img in random.Random(0).sample(sorted((YOLO / "images" / "train").glob("*.jpg")), args.check):
            frame = cv2.imread(str(img))
            h, w = frame.shape[:2]
            for line in (YOLO / "labels" / "train" / img.name).with_suffix(".txt").read_text().split("\n"):
                if not line:
                    continue
                c, xc, yc, bw, bh = line.split()
                xc, yc, bw, bh = float(xc) * w, float(yc) * h, float(bw) * w, float(bh) * h
                p1, p2 = (int(xc - bw / 2), int(yc - bh / 2)), (int(xc + bw / 2), int(yc + bh / 2))
                cv2.rectangle(frame, p1, p2, colors[int(c)], 2)
                cv2.putText(frame, PARTS[int(c)], (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[int(c)], 2)
            cv2.imwrite(str(out_dir / img.name), frame)
        print(f"Drew {args.check} images from their .txt files: {out_dir.relative_to(DATA.parent)}/")


if __name__ == "__main__":
    main()
