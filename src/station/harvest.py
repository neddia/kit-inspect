"""Pull real station photos into a training session: the photo that fooled the station.

The station keeps every inspection (data/history/). When one goes wrong on the line
(a battery terminal read as a nut, a label read as a bolt), that exact frame is the
most useful training example there is. Every line the station prints starts with
its history id (#166), so:

    uv run python -m src.station.harvest                                   # recent inspections, with ids
    uv run python -m src.station.harvest --from-id 166 --session live02    # one photo
    uv run python -m src.station.harvest --from-id 160 --to-id 170 --session live02   # a range
    uv run python -m src.station.harvest --from-id 160 --session liveval --split val  # a check set

copies the photos to data/images/<session>/ and adds manifest rows with the kit the
PLC sent and that kit's counts as a starting guess (other = -1, expected = "": the
truth isn't known yet). Then `uv run python -m src.labeling.review --session live02`:
label what's really there; `m` sets the manifest counts to what you labelled.

Near-duplicates in a range (nothing moved between two kits) are skipped: ten
copies of one tray teach nothing new and make scores look better than they are.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

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
    parser.add_argument("--from-id", type=int, help="first history id to take (the #number the station prints)")
    parser.add_argument("--to-id", type=int, help="last history id to take (default: just --from-id)")
    parser.add_argument("--session", help="new session name, e.g. live02")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="val = a check set the model never trains on")
    args = parser.parse_args()

    db = History().db
    rows = [dict(r) for r in db.execute("SELECT id, time, cycle_id, kit_id, result, reasons, image FROM inspections "
                                        "WHERE image IS NOT NULL AND result != 'FAULT' AND harvested IS NULL "
                                        "ORDER BY id")]
    if args.session is None or args.from_id is None:
        print("Recent inspections not harvested yet (take one with --from-id ID --session NAME):")
        for r in rows[-15:]:
            print(f"  #{r['id']:<5} {r['time'][11:19]}  cycle {r['cycle_id'] or '-':>4}  {r['kit_id'] or '-':<6} "
                  f"{r['result']:<7} {', '.join(json.loads(r['reasons'] or '[]'))[:70]}")
        return
    last = args.to_id if args.to_id is not None else args.from_id
    rows = [r for r in rows if args.from_id <= r["id"] <= last]
    if not rows:
        raise SystemExit(f"nothing to take: no photos with ids {args.from_id}-{last} (or already harvested)")

    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]
    with (DATA / "manifest.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        columns, manifest = reader.fieldnames, list(reader)
    if any(m["session"] == args.session for m in manifest):
        raise SystemExit(f"session {args.session} is already in the manifest; pick a new name (live03...)")

    kept, skipped, prev = [], 0, None
    for r in rows:
        t = thumb(ROOT / r["image"])
        if prev is not None and (np.abs(t - prev) > CHANGED).sum() < MIN_CHANGED:
            skipped += 1
            continue
        prev = t
        kept.append(r)

    out_dir = DATA / "images" / args.session
    out_dir.mkdir(parents=True, exist_ok=True)
    new = []
    for i, r in enumerate(kept, 1):
        name = f"{args.session}_{i:04d}.jpg"
        shutil.copy2(ROOT / r["image"], out_dir / name)
        kit = r["kit_id"] if r["kit_id"] in kits else "KIT-A"
        spec = kits[kit]
        new.append({"image": f"{args.session}/{name}", "session": args.session, "split": args.split, "tier": "normal",
                    "kit_id": kit, **{f"{p}s": spec.get(p, 0) for p in ("bolt", "nut", "screw")},
                    "other": -1, "expected": "", "scenario": "live"})
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
