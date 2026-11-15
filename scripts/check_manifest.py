"""Check that data/manifest.csv and data/images/ agree.

Run from the repo root:
    uv run python scripts/check_manifest.py

Catches the problems that come from editing files by hand: a manifest row whose
image was deleted, an image with no manifest row, or the same image listed twice.
Training and evaluation scripts trust the manifest, so run this before them.
"""

import csv
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"


def main():
    with (DATA / "manifest.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    listed = [r["image"] for r in rows]
    on_disk = {str(p.relative_to(DATA / "images")) for p in (DATA / "images").glob("*/*.jpg")}

    problems = []
    for image, n in Counter(listed).items():
        if n > 1:
            problems.append(f"listed {n} times: {image}")
    for image in sorted(set(listed) - on_disk):
        problems.append(f"row but no image file: {image}")
    for image in sorted(on_disk - set(listed)):
        problems.append(f"image but no row: {image}")

    sessions = Counter(r["session"] for r in rows)
    print(f"{len(rows)} rows, {len(on_disk)} images, sessions: {dict(sessions) or 'none yet'}")
    for p in problems:
        print("PROBLEM:", p)
    print("Manifest OK." if not problems else f"{len(problems)} problem(s).")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
