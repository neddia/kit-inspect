"""The vision station: wait for the PLC's trigger, photograph, inspect, answer, record.

    camera (always running) ──┐
    PLC trigger ─────────────>├─> fresh photo -> inspect -> Result/Done to the PLC -> history
                              │                                   (src/station/history.py)

One job, no modes: it answers the PLC. Kits are sent from the PLC's operator panel
(src/plc/panel.py), a separate app, like a real cell. The station reconnects to the
PLC by itself and faults the moment the camera is lost (clears when it's back).

--show: a window with the live view, the last result (boxes, label, reasons) and
the station's status. q or the window's X = quit. Every printed line starts with
the history id (#123), so a photo that fooled the station can be pulled into a
training session later: uv run python -m src.station.harvest --from-id 123 ...

Run:
    uv run python -m src.demo                                  # PLC panel + this station, together
    uv run python -m src.station.run --show                    # just the station (camera)
    uv run python -m src.station.run --show --images data/images/val01   # photos from a folder
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
CAMERA_STALE_S = 1.0      # newest frame older than this, even while idle = camera lost (fault until frames return)
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

    def health(self):
        return None


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

    def health(self):
        """None while frames arrive; else why not (checked every PLC poll, so an idle unplug faults at once)."""
        frame, age = self.feed.latest()
        if frame is not None and age <= CAMERA_STALE_S:
            return None
        return f"camera lost: {self.feed.error or f'no new frame for {age:.1f} s'}"


def window_closed():
    """True once the window's X button was clicked. (The Qt build raises instead of returning 0 for a gone window.)"""
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def fit(img, width):
    if img is None:
        return np.zeros((int(width * 9 / 16), width, 3), np.uint8)
    return cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])))


def status_panel(width, lines):
    """Under the live view: the station's status, one (text, colour) per line."""
    panel = np.zeros((40 + 34 * len(lines), width, 3), np.uint8)
    for i, (text, color) in enumerate(lines):
        cv2.putText(panel, text, (12, 34 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1 + (i == 0), cv2.LINE_AA)
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
    parser = argparse.ArgumentParser(description="Vision station (answers the PLC; the PLC is SIMULATED)")
    parser.add_argument("--images", help="use photos from this folder instead of the camera")
    parser.add_argument("--show", action="store_true", help="window with the live view and the last result")
    parser.add_argument("--no-history", action="store_true", help="don't record inspections")
    parser.add_argument("--preview-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    quiet_jpeg_warnings()

    inspector = Inspector()
    model = model_version(inspector.cfg["model"])
    high = inspector.cfg["decision"]["high"]
    frames = FolderFrames(args.images) if args.images else CameraFrames()
    warm = frames.get() if args.images else frames.latest()[0]
    if warm is not None:
        inspector.inspect(warm, "KIT-A")          # warm-up: the first inspection is slow
    if args.images:
        frames = FolderFrames(args.images)        # start the folder from its first photo
    history = None if args.no_history else History()
    last = {"img": None}
    state = {"plc": "PLC: connecting...", "last": "waiting for the first kit", "camera": ""}

    def on_result(r):
        if r.get("recovered"):                    # health fault cleared by itself: frames are back
            state["camera"] = "camera back: fault cleared"
            print(state["camera"])
            return
        row = None
        if history:
            try:
                row = history.record(r, model)
            except Exception as e:                # a full disk must not stop the line: warn, keep inspecting
                print(f"WARNING: history not saved: {e}")
        tag = f"#{row:<5}" if row else "#-    "
        who = f"cycle {r['cycle_id']:>4}" if r["cycle_id"] is not None else "idle      "
        if "error" in r:
            state["camera"] = f"FAULT: {r['error']}"[:70]
            print(f"{tag} {who}  FAULT   {r['error']}")
            return
        d = r["decision"]
        state["camera"] = ""
        print(f"{tag} {who}  {frames.name:<16} {r['kit_id'] or '-':<7} {d.result:<7} "
              f"photo {r['ms_frame']:>4.0f} ms  total {r['ms']:>4.0f} ms  {'; '.join(d.reasons)}")
        state["last"] = f"cycle {r['cycle_id']}: {r['kit_id']} -> {d.result}   ({r['ms']:.0f} ms, history {tag.strip()})"
        if args.show or args.preview_dir:
            annotated = draw(r["frame"], r["out"], high, operator=bool(args.preview_dir))
            if args.preview_dir:
                from src.station.demo_view import publish
                publish(args.preview_dir / "inspection", annotated[-r["frame"].shape[0]:], cycle=r["cycle_id"],
                        kit=r["kit_id"], result=d.result, reasons=d.reasons,
                        expected=d.spec, confirmed=d.sure, uncertain=d.unsure,
                        tag=(r["out"].get("label") or {}).get("label"))
            last["img"] = annotated

    station = VisionSide(inspector.inspect, frames.get, on_result=on_result, health=frames.health)

    def plc_status(text):
        if text != state["plc"]:                  # print changes only, not every retry
            print(text)
        state["plc"] = text

    source = args.images or "camera"
    print(f"Station ready. Model {model}, photos from {source}"
          f"{', history in data/history/' if history else ''}. Answering the PLC; Ctrl+C to stop.")
    try:
        if args.preview_dir:
            from src.station.demo_view import publish
            threading.Thread(target=station.serve, kwargs={"on_status": plc_status}, daemon=True).start()
            while True:
                frame, _ = frames.latest()
                error = frames.health()
                publish(args.preview_dir / "preview", None if error else frame, time=time.monotonic(),
                        error=error, replay=bool(args.images))
                time.sleep(.15)
        elif not args.show:
            station.serve(on_status=plc_status)
        else:
            threading.Thread(target=station.serve, kwargs={"on_status": plc_status}, daemon=True).start()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            while True:
                ok = state["plc"].startswith("PLC: connected")
                lines = [(f"VISION STATION   model {model}", (255, 255, 255)),
                         (state["plc"], (60, 200, 60) if ok else (90, 90, 255)),
                         ("last: " + state["last"], (0, 200, 255)),
                         (state["camera"] or f"camera: {source}", (90, 90, 255) if state["camera"].startswith("FAULT")
                          else (170, 170, 170)),
                         ("kits come from the PLC panel      q = quit", (170, 170, 170))]
                cv2.imshow(WINDOW, side_by_side(frames.latest()[0], last["img"], status_panel(560, lines)))
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q") or window_closed():
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
