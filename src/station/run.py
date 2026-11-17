"""The inspection station: wait for the PLC's trigger, photograph, inspect, answer, record.

    camera (always running) ──┐
    PLC trigger ─────────────>├─> fresh photo -> inspect -> Result/Done to the PLC -> history
                              │                                   (src/station/history.py)
    --show: live view + last result in a window (for development; part 5 adds a web dashboard)
            keys: t = save for training on/off, u = undo the last saved photo, q = quit

Run, with the simulated PLC running (uv run python -m src.plc.tags):
    uv run python -m src.station.run                        # the camera
    uv run python -m src.station.run --show                 # + window: live view and last result
    uv run python -m src.station.run --show --collect       # ...with "save for training" already on
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


def fit(img, width):
    if img is None:
        return np.zeros((int(width * 9 / 16), width, 3), np.uint8)
    return cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))


def status_panel(width, collect, marked, note):
    """Under the live view: is "save for training" on, how many are saved, the keys."""
    panel = np.zeros((170, width, 3), np.uint8)
    on = (60, 200, 60) if collect else (110, 110, 110)
    cv2.putText(panel, f"SAVE FOR TRAINING: {'ON' if collect else 'OFF'}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, on, 2, cv2.LINE_AA)
    cv2.putText(panel, f"{marked} photos saved for training", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, note, (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "t = on/off   u = undo last   q = quit", (12, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (170, 170, 170), 1, cv2.LINE_AA)
    return panel


def side_by_side(live, result, panel=None, widths=(560, 900)):
    """Live view (+ status panel under it) | last result (bigger, so the labels are readable)."""
    left = fit(live, widths[0])
    if panel is not None:
        left = np.vstack([left, panel])
    right = fit(result, widths[1])
    h = max(left.shape[0], right.shape[0])
    pad = lambda img: cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.hstack([pad(left), pad(right)])


def main():
    parser = argparse.ArgumentParser(description="Inspection station (the PLC is SIMULATED)")
    parser.add_argument("--images", help="use photos from this folder instead of the camera")
    parser.add_argument("--show", action="store_true", help="window with the live view and the last result")
    parser.add_argument("--no-history", action="store_true", help="don't record inspections")
    parser.add_argument("--collect", action="store_true", help='start with "save for training" on (t toggles)')
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
    state = {"collect": args.collect and history is not None, "note": ""}

    def on_result(r):
        if history:
            try:
                history.record(r, model, train=state["collect"])
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
            marked = history.marked() if history else 0
            while worker.is_alive():
                panel = status_panel(560, state["collect"], marked, state["note"])
                cv2.imshow(WINDOW, side_by_side(frames.latest()[0], last["img"], panel))
                key = cv2.waitKey(50) & 0xFF
                closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1   # the window's X button
                if key == ord("q") or closed:
                    break
                if not history:
                    continue
                if key == ord("t"):
                    state["collect"] = not state["collect"]
                    state["note"] = f"saving {'ON' if state['collect'] else 'OFF'} from the next kit"
                    print(state["note"])
                elif key == ord("u"):
                    cycle = history.unmark_last()
                    state["note"] = f"undone: cycle {cycle} won't be used" if cycle else "nothing to undo"
                    print(state["note"])
                marked = history.marked()
    except KeyboardInterrupt:
        pass
    finally:
        station.stop.set()
        if isinstance(frames, CameraFrames):
            frames.feed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
