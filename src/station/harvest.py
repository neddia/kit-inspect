"""Turn the station's history photos into a training session (the "hard cases" loop).

Free-flow capture: run the station with --show, press t ("save for training" ON),
keep a FULL kit on the tray, add 0-2 pieces of junk (coin cells, coins, washers...),
move something, press ENTER in the PLC terminal, repeat. u in the window un-saves
the last photo (a slip: missing part, hand in view). Then:

    uv run python -m src.station.harvest                          # how many are saved
    uv run python -m src.station.harvest --session live01         # take the saved ones
    uv run python -m src.station.harvest --from-id 58 --session live01   # or an id range, saved or not
    uv run python -m src.station.harvest --session liveval --split val   # a check set: never trained on

copies those photos to data/images/live01/ and adds manifest rows: KIT-A, 3/3/2
(the full-kit rule), split train, scenario "live", other = -1 and expected = ""
(junk wasn't recorded; these rows are for training, not scoring). Then the normal
Stage 4-5 pipeline: auto_label -> review -> export_yolo -> train.

Near-duplicates (nothing moved between two ENTERs) are skipped: ten copies of one
tray teach nothing new and make the scores look better than they are.
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.station.history import ROOT, History

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CHANGED = 25      # a thumbnail pixel this much brighter/darker (0-255) = something moved there
MIN_CHANGED = 10  # photos with fewer changed pixels than this are the same tray
# Measured Nov 17: same tray twice -> no pixel differs by more than 6 (camera noise).
# Adding one coin cell -> mean difference only 1.9, but dozens of pixels differ by 59+.
# So count changed pixels; an average would miss a small object moved.


def thumb(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, (160, 90)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Copy station history photos into a training session")
    parser.add_argument("--from-id", type=int, help="take an id range instead of the saved photos")
    parser.add_argument("--to-id", type=int, help="last history id to take (default: the newest)")
    parser.add_argument("--session", help="new session name, e.g. live01")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="val = a check set the model never trains on")
    args = parser.parse_args()

    db = History().db                               # (adds the train/harvested columns to old history files)
    rows = [dict(r) for r in db.execute("SELECT id, time, cycle_id, result, image, train FROM inspections "
                                        "WHERE image IS NOT NULL AND result != 'FAULT' AND harvested IS NULL "
                                        "ORDER BY id")]
    if not args.session:
        saved = [r for r in rows if r["train"]]
        print(f"{len(saved)} photos saved for training, not harvested yet"
              + (f" (ids {saved[0]['id']}-{saved[-1]['id']})" if saved else ""))
        print(f"{len(rows)} photos in the history not harvested yet"
              + (f", newest id {rows[-1]['id']}" if rows else ""))
        print("Take the saved ones with: --session live01   (or an id range: --from-id N [--to-id M] --session live01)")
        return
    if args.from_id is not None:
        rows = [r for r in rows if r["id"] >= args.from_id and (args.to_id is None or r["id"] <= args.to_id)]
    else:
        rows = [r for r in rows if r["train"]]
    if not rows:
        raise SystemExit("nothing to take: no saved photos (press t in the station window) and no --from-id")

    out_dir = DATA / "images" / args.session
    with (DATA / "manifest.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        columns, manifest = reader.fieldnames, list(reader)
    if any(m["session"] == args.session for m in manifest):
        raise SystemExit(f"session {args.session} is already in the manifest; pick a new name (live02...)")

    kept, skipped, last = [], 0, None
    for r in rows:
        t = thumb(ROOT / r["image"])
        if last is not None and (np.abs(t - last) > CHANGED).sum() < MIN_CHANGED:
            skipped += 1
            continue
        last = t
        kept.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    new = []
    for i, r in enumerate(kept, 1):
        name = f"{args.session}_{i:04d}.jpg"
        shutil.copy2(ROOT / r["image"], out_dir / name)
        new.append({"image": f"{args.session}/{name}", "session": args.session, "split": args.split, "tier": "normal",
                    "kit_id": "KIT-A", "bolts": 3, "nuts": 3, "screws": 2, "other": -1, "expected": "",
                    "scenario": "live"})
    with (DATA / "manifest.csv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writerows(new)
    db.executemany("UPDATE inspections SET harvested = ? WHERE id = ?", [(args.session, r["id"]) for r in rows])
    db.commit()                                     # never taken twice (near-duplicates included)
    print(f"{len(new)} photos -> data/images/{args.session}/ and the manifest "
          f"({skipped} near-duplicates skipped). Next:")
    print("  uv run python -m src.labeling.auto_label")
    print(f"  uv run python -m src.labeling.review --session {args.session}")


if __name__ == "__main__":
    main()
