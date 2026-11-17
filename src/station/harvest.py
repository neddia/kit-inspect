"""Turn the station's history photos into a training session (the "hard cases" loop).

Free-flow capture: run the station, keep a FULL kit on the tray, add 0-2 pieces of
junk (coin cells, coins, washers...), move something, press ENTER in the PLC
terminal, repeat. Then:

    uv run python -m src.station.harvest                          # what's in the history
    uv run python -m src.station.harvest --from-id 51 --session live01

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
import sqlite3
from pathlib import Path

import cv2
import numpy as np

from src.station.history import ROOT

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
    parser.add_argument("--from-id", type=int, help="first history id to take (see the listing)")
    parser.add_argument("--to-id", type=int, help="last history id to take (default: the newest)")
    parser.add_argument("--session", default="live01")
    args = parser.parse_args()

    db = sqlite3.connect(ROOT / "history.db")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT id, time, cycle_id, result, image FROM inspections "
                                        "WHERE image IS NOT NULL AND result != 'FAULT' ORDER BY id")]
    if args.from_id is None:
        print(f"{len(rows)} photos in the history, ids {rows[0]['id']}-{rows[-1]['id']}" if rows else "history is empty")
        for r in rows[-5:]:
            print(f"  id {r['id']:>4}  {r['time'][:19]}  {r['result']}")
        print("Take a range with: --from-id N [--to-id M] --session live01")
        return
    rows = [r for r in rows if r["id"] >= args.from_id and (args.to_id is None or r["id"] <= args.to_id)]

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
        new.append({"image": f"{args.session}/{name}", "session": args.session, "split": "train", "tier": "normal",
                    "kit_id": "KIT-A", "bolts": 3, "nuts": 3, "screws": 2, "other": -1, "expected": "",
                    "scenario": "live"})
    with (DATA / "manifest.csv").open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writerows(new)
    print(f"{len(new)} photos -> data/images/{args.session}/ and the manifest "
          f"({skipped} near-duplicates skipped). Next:")
    print("  uv run python -m src.labeling.auto_label")
    print(f"  uv run python -m src.labeling.review --session {args.session}")


if __name__ == "__main__":
    main()
