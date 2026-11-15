"""Capture a session of tray images: live view, one saved frame per keypress.

Tune the camera first with the preview (and press s to save the settings).
This script deliberately has NO keys that change camera settings, so a stray
keypress can't change exposure halfway through a session.

Run from the repo root, one session at a time:
    uv run python -m src.capture.capture --session pilot  --split train
    uv run python -m src.capture.capture --session train02 --split train --notes "room lights off"
    uv run python -m src.capture.capture --session stress01 --split test --tier stress --notes "side light"

The loop (from the build plan): rearrange or shake, set the tray down,
hands out, press SPACE.

Keys (click the window first). No Shift needed; Shift doesn't reach OpenCV on WSL:
    SPACE    save a frame              u        undo the last save
    1 2 3..  pick a part (in kits.yaml order)    0   pick "other" (foreign objects)
    -  =     one fewer / one more of the picked part
    h        toggle "parts stacked or hidden"
    c        toggle grid               q / Esc  quit

The counts are what's REALLY on the tray. Set them before pressing SPACE.
The scenario (complete, missing_nut, ...) and the expected result are worked
out from the counts, so you only ever enter facts.

The parts come from configs/kits.yaml, so changing the kit there changes the
counts, the manifest columns and the scenario names here.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import yaml

from src.capture.camera import load_camera_config, open_camera, quiet_jpeg_warnings
from src.capture.preview import draw_overlay

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
MANIFEST = DATA / "manifest.csv"
KITS = REPO / "configs" / "kits.yaml"

# The build plan's mix for each 40-image training/validation session.
# (Test sessions use 20 complete + 40 faulty instead.) A missing_<part> not
# listed here gets MISSING_DEFAULT.
TARGET_MIX = {
    "complete": 18, "missing_washer": 5, "missing_nut": 3, "missing_bolt": 3,
    "extra_part": 4, "two_faults": 2, "foreign_object": 2, "stacked": 2, "empty": 1,
}
MISSING_DEFAULT = 3


def all_parts(kits):
    """Every part name used by any kit, in first-seen order: ['bolt', 'nut', ...]."""
    parts = []
    for spec in kits.values():
        parts += [p for p in spec if p not in parts]
    return parts


def manifest_columns(parts):
    """One count column per part, e.g. bolts, nuts, screws."""
    return ["image", "session", "split", "tier", "kit_id",
            *[f"{p}s" for p in parts], "other", "expected", "scenario"]


# ---------- the rules: what scenario is this tray, and what should the station say? ----------

def scenario_for(counts, spec, stacked=False):
    """Name the situation on the tray, from the real part counts.

    counts: {"bolt": 2, "nut": 2, "washer": 3, "other": 0}
    spec:   {"bolt": 2, "nut": 2, "washer": 4}  (from kits.yaml)
    A part in counts but not in the spec should be 0; any of it is extra.
    """
    parts = [p for p in counts if p != "other"]
    diffs = {p: counts[p] - spec.get(p, 0) for p in parts}
    faults = sum(abs(d) for d in diffs.values()) + (1 if counts["other"] else 0)

    if all(counts[p] == 0 for p in parts) and counts["other"] == 0:
        return "empty"
    if stacked:
        return "stacked"
    if faults == 0:
        return "complete"
    if faults >= 2:
        return "two_faults"
    if counts["other"]:
        return "foreign_object"
    missing = [p for p, d in diffs.items() if d < 0]
    return f"missing_{missing[0]}" if missing else "extra_part"


def expected_for(counts, spec, tier, stacked=False):
    """What the station SHOULD decide. NOT_PASS means FAIL and REVIEW are both correct."""
    if tier == "invalid" or counts["other"] or stacked:
        return "NOT_PASS"
    if all(counts[p] == spec.get(p, 0) for p in counts if p != "other"):
        return "PASS"
    return "FAIL"


# ---------- files: images and the manifest ----------

def next_index(session_dir):
    """Continue numbering after any images already in this session (never overwrite)."""
    numbers = [int(p.stem.rsplit("_", 1)[1]) for p in session_dir.glob("*.jpg")]
    return max(numbers, default=0) + 1


def prepare_manifest(manifest, columns):
    """Make sure the manifest's header matches the current parts.

    With no rows yet, just (re)write the header. With rows, refuse to mix
    layouts: rows written for one kit would be misread under another header.
    """
    rows = []
    if manifest.exists() and manifest.stat().st_size > 0:
        with manifest.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if reader.fieldnames == columns:
                return
    if rows:
        raise SystemExit(
            f"manifest.csv has {len(rows)} rows with columns {reader.fieldnames}, but kits.yaml "
            f"now needs {columns}. Changing parts after capturing means re-shooting or "
            "migrating those rows; sort that out before capturing more."
        )
    with manifest.open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=columns).writeheader()


def append_row(manifest, row, columns):
    new_file = not manifest.exists() or manifest.stat().st_size == 0
    with manifest.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def remove_row(manifest, image, columns):
    with manifest.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["image"] != image]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def session_counts(manifest, session):
    """How many images of each scenario this session already has (for resuming)."""
    if not manifest.exists() or manifest.stat().st_size == 0:
        return {}
    counts = {}
    with manifest.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["session"] == session:
                counts[r["scenario"]] = counts.get(r["scenario"], 0) + 1
    return counts


def write_session_info(session_dir, args, cam_cfg):
    """Record how this session was shot, so it can be reproduced or explained later."""
    info_path = session_dir / "session.yaml"
    if info_path.exists():
        return
    info = {
        "session": args.session, "split": args.split, "tier": args.tier,
        "kit_id": args.kit, "notes": args.notes,
        "started": datetime.now().isoformat(timespec="seconds"),
        "camera": cam_cfg,
    }
    info_path.write_text(yaml.safe_dump(info, sort_keys=False))


# ---------- the capture loop ----------

def is_jpeg(raw):
    """True if this is a flat buffer of JPEG bytes (JPEGs start with FF D8)."""
    return raw is not None and raw.ndim == 2 and raw.shape[0] == 1 and raw.size > 2 \
        and raw[0, 0] == 0xFF and raw[0, 1] == 0xD8


def grab_fresh(cap):
    """Return the newest frame's raw JPEG bytes, or None.

    The driver keeps a few frames queued, so the next read() can be a frame
    from before you pressed the key (maybe with your hand still in it). Throw
    the queued ones away first.
    """
    for _ in range(4):
        cap.grab()
    for _ in range(3):  # retry if a frame comes back garbled
        ok, raw = cap.retrieve()
        if ok and is_jpeg(raw) and cv2.imdecode(raw, cv2.IMREAD_COLOR) is not None:
            return raw
        cap.grab()
    return None


def mix_lines(done, spec, show_targets):
    names = ["complete", *[f"missing_{p}" for p in spec],
             "extra_part", "two_faults", "foreign_object", "stacked", "empty"]
    items = []
    for name in names:
        n = done.get(name, 0)
        target = TARGET_MIX.get(name, MISSING_DEFAULT)
        items.append(f"{name} {n}/{target}" if show_targets else f"{name} {n}")
    half = (len(items) + 1) // 2
    return ["  ".join(items[:half]), "  ".join(items[half:])]


def main():
    parser = argparse.ArgumentParser(description="Capture a session of tray images")
    parser.add_argument("--session", required=True, help="e.g. pilot, train01, val01, test01, stress01")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"],
                        help="pilot joins train; stress and invalid sessions use test")
    parser.add_argument("--tier", default="normal", choices=["normal", "stress", "invalid"])
    parser.add_argument("--kit", default="KIT-A", help="kit ID printed on the tray")
    parser.add_argument("--notes", default="", help="the session's condition, e.g. 'room lights off'")
    args = parser.parse_args()

    kits = yaml.safe_load(KITS.read_text())["kits"]
    spec = kits[args.kit]
    parts = all_parts(kits)          # every part any kit uses (fixed manifest columns)
    columns = manifest_columns(parts)
    prepare_manifest(MANIFEST, columns)
    session_dir = DATA / "images" / args.session
    session_dir.mkdir(parents=True, exist_ok=True)

    quiet_jpeg_warnings()
    cam_cfg = load_camera_config()
    write_session_info(session_dir, args, cam_cfg)
    cap = open_camera(cam_cfg)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # give us the camera's own JPEG bytes

    counts = {p: spec.get(p, 0) for p in parts} | {"other": 0}  # start from a complete kit
    pickable = [*parts, "other"]     # keys 1..N pick parts, 0 picks other
    picked = 0
    stacked = False
    show_grid = True
    done = session_counts(MANIFEST, args.session)
    saved = []  # images saved in this run, for undo
    status, status_until = "", 0.0
    show_targets = args.tier == "normal" and args.split != "test"

    window = f"capture: {args.session}  (SPACE to save, q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, cam_cfg["width"], cam_cfg["height"])
    print(__doc__.split("Keys")[0].strip().splitlines()[0])
    print(f"Saving to {session_dir.relative_to(REPO)}/  ({sum(done.values())} images already in this session)")

    while True:
        ok, raw = cap.read()
        if not ok:
            print("Lost the camera. On WSL: usbipd.exe attach --wsl --busid 8-4")
            break
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR) if is_jpeg(raw) else None
        if frame is None:
            continue  # a frame that's mid-switch or garbled: just wait for the next one

        scenario = scenario_for(counts, spec, stacked)
        expected = expected_for(counts, spec, args.tier, stacked)
        lines = [
            f"{args.session}  split={args.split}  tier={args.tier}  {args.kit}   images: {sum(done.values())}",
            "ON TRAY: " + "  ".join(
                f"{'>' if pickable[picked] == p else ' '}[{(i + 1) % len(pickable)}] {p}s {counts[p]}"
                for i, p in enumerate(pickable)
            ) + f"   stacked {'YES' if stacked else 'no'} [h]   (-/= to change)",
            f"-> {scenario}, expected {expected}",
            *mix_lines(done, spec, show_targets),
            status if time.time() < status_until else "",
        ]
        cv2.imshow(window, draw_overlay(frame, lines, show_grid))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        elif key == ord(" "):
            fresh = grab_fresh(cap)
            if fresh is None:
                status, status_until = "SAVE FAILED: no frame from camera", time.time() + 3
                continue
            index = next_index(session_dir)
            name = f"{args.session}_{index:04d}.jpg"
            # Write the camera's JPEG bytes as-is: no second round of compression.
            (session_dir / name).write_bytes(fresh.tobytes())
            image = f"{args.session}/{name}"
            append_row(MANIFEST, {
                "image": image, "session": args.session, "split": args.split, "tier": args.tier,
                "kit_id": args.kit, **{f"{p}s": counts[p] for p in parts},
                "other": counts["other"], "expected": expected, "scenario": scenario,
            }, columns)
            saved.append((image, scenario))
            done[scenario] = done.get(scenario, 0) + 1
            status, status_until = f"SAVED {name}  ({scenario}, {expected})", time.time() + 2
            print(status)

        elif key == ord("u"):
            if not saved:
                status, status_until = "nothing to undo in this run", time.time() + 2
                continue
            image, scenario = saved.pop()
            (DATA / "images" / image).unlink(missing_ok=True)
            remove_row(MANIFEST, image, columns)
            done[scenario] -= 1
            status, status_until = f"UNDID {image}", time.time() + 2
            print(status)

        elif ord("0") <= key <= ord("9"):
            digit = key - ord("0")
            index = len(pickable) - 1 if digit == 0 else digit - 1  # 0 = other
            if index < len(pickable):
                picked = index
        elif key in (ord("-"), ord("=")):
            part = pickable[picked]
            counts[part] = max(0, counts[part] + (1 if key == ord("=") else -1))
        elif key == ord("h"):
            stacked = not stacked
        elif key == ord("c"):
            show_grid = not show_grid

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Session {args.session}: {sum(done.values())} images. Mix: {done}")


if __name__ == "__main__":
    main()
