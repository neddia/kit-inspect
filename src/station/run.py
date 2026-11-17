"""The inspection station: wait for the PLC's trigger, photograph, inspect, answer, record.

    camera (always running) ──┐
    PLC trigger ─────────────>├─> fresh photo -> inspect -> Result/Done to the PLC -> history
                              │                                   (src/station/history.py)
    --show: live view + last result in a window (for development; part 5 adds a web dashboard)

Run, with the simulated PLC running (uv run python -m src.plc.tags):
    uv run python -m src.station.run                        # the camera
    uv run python -m src.station.run --show                 # + window: live view and last result
    uv run python -m src.station.run --images data/images/val01   # photos from a folder, no camera
Then send kits from the PLC side: uv run python -m src.plc.line
"""

import argparse
import hashlib
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import quiet_jpeg_warnings
from src.inspection.draw import draw
from src.inspection.pipeline import REPO, Inspector
from src.station.handshake import VisionSide
from src.station.history import History

FRAME_TIMEOUT_S = 1.0     # no new camera frame within this after a trigger = camera lost (fault, PLC holds)
WINDOW = "Kit inspection station"


def model_version(path):
    """Model file name + a short fingerprint of its weights, so history says exactly which model decided."""
    path = REPO / path
    files = sorted(path.glob("*")) if path.is_dir() else [path]
    h = hashlib.sha256()
    for f in files:
        h.update(f.read_bytes())
    return f"{path.name}@{h.hexdigest()[:8]}"


class FolderFrames:
    """Photos from a folder, one per trigger, in order (stands in for the camera)."""

    def __init__(self, folder):
        self.paths = sorted(Path(folder).glob("*.jpg"))
        if not self.paths:
            raise SystemExit(f"no .jpg files in {folder}")
        self.i, self.name, self.frame = 0, "-", None

    def get(self, after=None):
        path = self.paths[self.i % len(self.paths)]
        self.i += 1
        self.name, self.frame = path.name, cv2.imread(str(path))
        return self.frame

    def latest(self):
        return self.frame, 0.0


class CameraFrames:
    def __init__(self):
        from src.station.camera_feed import CameraFeed

        self.feed = CameraFeed().start()
        self.name = "camera"
        deadline = time.monotonic() + 15
        while self.feed.frame is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.feed.frame is None:
            print(f"WARNING: no camera frames yet ({self.feed.error}). Triggers will fault until it works.")

    def get(self, after=None):
        return self.feed.fresh_frame(after, FRAME_TIMEOUT_S)

    def latest(self):
        return self.feed.latest()


def side_by_side(live, result, widths=(560, 900)):
    """Live view | last result (bigger, so the labels are readable)."""
    tiles = []
    for img, width in zip((live, result), widths):
        if img is None:
            img = np.zeros((360, width, 3), np.uint8)
        tiles.append(cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1]))))
    h = max(t.shape[0] for t in tiles)
    return np.hstack([cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles])


def main():
    parser = argparse.ArgumentParser(description="Inspection station (the PLC is SIMULATED)")
    parser.add_argument("--images", help="use photos from this folder instead of the camera")
    parser.add_argument("--show", action="store_true", help="window with the live view and the last result")
    parser.add_argument("--no-history", action="store_true", help="don't record inspections")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    quiet_jpeg_warnings()

    inspector = Inspector()
    model = model_version(inspector.cfg["model"])
    high = inspector.cfg["decision"]["high"]
    frames = FolderFrames(args.images) if args.images else CameraFrames()
    warm = frames.get() if args.images else frames.latest()[0]
    if warm is not None:
        inspector.inspect(warm, "KIT-A")         # warm-up: the first inspection is slow
    if args.images:
        frames = FolderFrames(args.images)        # start the folder from its first photo
    history = None if args.no_history else History()
    last = {"img": None}

    def on_result(r):
        if history:
            try:
                history.record(r, model)
            except Exception as e:                # a full disk must not stop the line: warn, keep inspecting
                print(f"WARNING: history not saved: {e}")
        if "error" in r:
            print(f"cycle {r['cycle_id']:>4}  FAULT   {r['error']}")
            return
        d = r["decision"]
        print(f"cycle {r['cycle_id']:>4}  {frames.name:<16} {r['kit_id'] or '-':<7} {d.result:<7} "
              f"photo {r['ms_frame']:>4.0f} ms  total {r['ms']:>4.0f} ms  {'; '.join(d.reasons)}")
        if args.show:
            last["img"] = draw(r["frame"], r["out"], high)

    station = VisionSide(inspector.inspect, frames.get, on_result=on_result)
    source = args.images or "camera"
    print(f"Station ready. Model {model}, photos from {source}"
          f"{', history in data/history/' if history else ''}. Waiting for PLC triggers; Ctrl+C to stop.")
    try:
        if not args.show:
            station.run()
        else:
            worker = threading.Thread(target=station.run, daemon=True)
            worker.start()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            while worker.is_alive():
                cv2.imshow(WINDOW, side_by_side(frames.latest()[0], last["img"]))
                if cv2.waitKey(50) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        station.stop.set()
        if isinstance(frames, CameraFrames):
            frames.feed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
