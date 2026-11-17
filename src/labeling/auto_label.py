"""Stage 4, step 1: SAM 3 draws the outlines, shape decides the class.

What we learned exploring (Nov 16): SAM 3 finds WHERE every part is very well,
but its words are unreliable (bolts called "screw", bolt heads called "nut",
the same part boxed 2-3 times). So per image:

  1. SAM 3 finds outlines with the prompts "bolt", "nut", "screw".
  2. Merge: an outline mostly inside another outline is the same part
     (a bolt head inside its bolt, a double detection) -> keep one.
  3. Classify by shape, measured from the outline's rotated rectangle:
       round (long side / short side < 1.6)        -> nut
       long and thick (area / length > 28 px)      -> bolt
       long and thin                               -> screw
  4. Parts outside the bowl (spares on the desk) are kept, marked outside,
     and not counted.
  5. Compare the counts inside the bowl with the manifest. A mismatch flags
     the image for careful review. (On 42 test images: 36 right, and all 6
     wrong ones were flagged.)

Run from the repo root (train and val sessions; resumes, skips images already done):
    uv run python -m src.labeling.auto_label --limit 5        # quick look first
    uv run python -m src.labeling.auto_label                  # all of them, ~5 min
    uv run python -m src.labeling.auto_label --redo           # start over

Writes:
    data/labels/auto/<session>/<image>.json   outlines, boxes, classes, measurements
    reports/stage4/auto_label.csv             per image: true vs auto counts, flagged or not
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import quiet_jpeg_warnings
from src.capture.capture import DATA, MANIFEST, REPO
from src.labeling.tray import find_tray

PARTS = ["bolt", "nut", "screw"]
AUTO = DATA / "labels" / "auto"
REPORT = REPO / "reports" / "stage4" / "auto_label.csv"

SAM_CONF = 0.25      # SAM's own cutoff; low on purpose: missing a part is worse than an extra outline
MIN_AREA = 200       # px; smaller outlines are specks, not parts
SAME_PART = 0.5      # outline more than half inside another = the same part
NUT_ASPECT = 1.6     # long side / short side below this = round = nut
BOLT_WIDTH = 28      # px; average width (area / length) above this = bolt. Measured: screws ~18, bolts ~40
TRAY_MARGIN = 1.15   # count parts whose centre is within 1.15 x the floor radius (they can sit up the wall)


def load_rows(splits):
    with MANIFEST.open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] in splits and r["tier"] == "normal"]


def merge(masks, conf):
    """Indexes of the outlines to keep: one per part, most confident first."""
    kept = []
    for i in np.argsort(-conf):
        area = masks[i].sum()
        if area < MIN_AREA:
            continue
        if any((masks[i] & masks[k]).sum() / area > SAME_PART for k in kept):
            continue  # mostly inside a part we already have
        # This one swallows a smaller part kept earlier (e.g. the head was kept before its bolt): replace it
        kept = [k for k in kept if (masks[i] & masks[k]).sum() / masks[k].sum() <= SAME_PART]
        kept.append(i)
    return kept


def describe(mask):
    """Outline polygon, box and shape measurements of one part."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest = max(cv2.contourArea(c) for c in contours)
    pieces = [c for c in contours if cv2.contourArea(c) >= 0.05 * biggest]  # drop specks
    if len(pieces) == 1:
        c = cv2.approxPolyDP(pieces[0], 1.0, True)  # fewer points, same shape (within 1 px)
    else:
        # A part split in two by something lying across it (a screw under a bolt):
        # one part, so one outline around all its visible pieces.
        c = cv2.convexHull(np.vstack(pieces))
    (cx, cy), (w, h), _ = cv2.minAreaRect(c)
    length, short = max(w, h), max(min(w, h), 1.0)
    area = float(mask.sum())
    x, y, bw, bh = cv2.boundingRect(c)
    return {
        "polygon": c.reshape(-1, 2).tolist(),
        "box": [x, y, x + bw, y + bh],
        "center": [round(cx, 1), round(cy, 1)],
        "length": round(length, 1),
        "aspect": round(length / short, 2),
        "width": round(area / length, 1),
    }


def in_tray(obj, tray):
    """Is the part's centre in the bowl? No tray found: count everything (the mismatch check will tell)."""
    if not tray:
        return True
    tx, ty, tr = tray
    return bool(np.hypot(obj["center"][0] - tx, obj["center"][1] - ty) < tr * TRAY_MARGIN)


def shape_class(obj):
    if obj["aspect"] < NUT_ASPECT:
        return "nut"
    return "bolt" if obj["width"] > BOLT_WIDTH else "screw"


def label_image(pred, image_path):
    frame = cv2.imread(str(image_path))
    tray = find_tray(frame)
    pred.set_image(str(image_path))
    r = pred(text=PARTS)[0]
    objects = []
    if r.masks is not None:
        masks = r.masks.data.cpu().numpy() > 0.5
        conf = r.boxes.conf.cpu().numpy()
        for k in merge(masks, conf):
            obj = describe(masks[k])
            obj["cls"] = shape_class(obj)
            obj["sam_word"] = r.names[int(r.boxes.cls[k])]
            obj["sam_conf"] = round(float(conf[k]), 3)
            obj["source"] = "sam"
            obj["inside"] = in_tray(obj, tray)
            objects.append(obj)
    return {
        "image": str(image_path.relative_to(DATA / "images")),
        "size": [frame.shape[1], frame.shape[0]],
        "tray": [round(v, 1) for v in tray] if tray else None,
        "objects": objects,
        "reviewed": False,
    }


def counts_inside(label):
    c = Counter(o["cls"] for o in label["objects"] if o["inside"])
    return {p: c.get(p, 0) for p in PARTS}


def main():
    parser = argparse.ArgumentParser(description="Auto-label train/val images with SAM 3 + shape rules")
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit", type=int, help="only the first N images (for a quick look)")
    parser.add_argument("--redo", action="store_true", help="relabel images that already have a label file")
    args = parser.parse_args()

    from ultralytics.models.sam import SAM3SemanticPredictor

    quiet_jpeg_warnings()
    rows = load_rows(args.splits)[: args.limit]
    pred = SAM3SemanticPredictor(overrides=dict(
        conf=SAM_CONF, task="segment", mode="predict", model=str(REPO / "models" / "sam3.pt"),
        half=True, save=False, verbose=False))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report, start = [], time.time()
    for n, row in enumerate(rows, 1):
        out = AUTO / Path(row["image"]).with_suffix(".json")
        if out.exists() and not args.redo:
            label = json.loads(out.read_text())
        else:
            label = label_image(pred, DATA / "images" / row["image"])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(label))
        true = {p: int(row[f"{p}s"]) for p in PARTS}
        auto = counts_inside(label)
        outside = sum(not o["inside"] for o in label["objects"])
        report.append({
            "image": row["image"], "session": row["session"], "scenario": row["scenario"],
            **{f"true_{p}": true[p] for p in PARTS}, **{f"auto_{p}": auto[p] for p in PARTS},
            "outside": outside, "tray_found": label["tray"] is not None, "flagged": auto != true,
        })
        flag = "  FLAGGED" if auto != true else ""
        print(f"[{n}/{len(rows)}] {row['image']}: true {list(true.values())} auto {list(auto.values())}{flag}")

    with REPORT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0]))
        w.writeheader()
        w.writerows(report)

    print(f"\nDone in {time.time() - start:.0f} s. Counts match the manifest:")
    by_session = {}
    for r in report:
        s = by_session.setdefault(r["session"], [0, 0])
        s[0] += not r["flagged"]
        s[1] += 1
    for session, (ok, total) in by_session.items():
        print(f"  {session:<10} {ok:>3}/{total}")
    flagged = sum(r["flagged"] for r in report)
    print(f"  {'all':<10} {len(report) - flagged:>3}/{len(report)}   ({flagged} flagged for careful review)")
    print(f"Saved {AUTO.relative_to(REPO)}/ and {REPORT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
