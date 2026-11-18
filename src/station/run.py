"""The inspection station: wait for the PLC's trigger, photograph, inspect, answer, record.

    camera (always running) ──┐
    PLC trigger ─────────────>├─> fresh photo -> inspect -> Result/Done to the PLC -> history
                              │                                   (src/station/history.py)
    --show: live view + last result in a window (for development; part 5 adds a web dashboard)
            keys: SPACE = inspect now (bench test, no PLC needed), t = save for training on/off,
                  u = undo the last saved photo, q or the window's X = quit

Easiest (one command, one window): the simulated PLC runs inside, keys drive it:
    uv run python -m src.station.run --show --plc-sim
        SPACE = the PLC sends the next kit (full handshake: trigger, inspect, release/hold)
        1 / 2 = which kit the PLC sends (KIT-A / KIT-B), --every 4 = a kit every 4 s by itself

Or with the PLC in other terminals (uv run python -m src.plc.tags, then src.plc.line):
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


def status_panel(width, collect, marked, note, plc, line="", keys="SPACE = inspect now   t = save on/off   u = undo   q = quit"):
    """Under the live view: is "save for training" on, how many are saved, PLC link and outcome, the keys."""
    panel = np.zeros((236, width, 3), np.uint8)
    on = (60, 200, 60) if collect else (110, 110, 110)
    cv2.putText(panel, f"SAVE FOR TRAINING: {'ON' if collect else 'OFF'}", (12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, on, 2, cv2.LINE_AA)
    cv2.putText(panel, f"{marked} photos saved for training", (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, note, (12, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, line, (12, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (60, 200, 60) if "RELEASE" in line else (90, 90, 255) if "HOLD" in line else (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, plc, (12, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.putText(panel, keys, (12, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
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


def start_plc_sim(kit_name, on_line=print):
    """--plc-sim: start the simulated PLC (or use one already running) and a worker that plays the
    PLC's side (src/plc/line.py) one kit per request. Returns a dict the window uses."""
    import queue

    from src.plc.line import Line
    from src.plc.tags import Tags, load_config, start_sim

    cfg, codes = load_config()
    proc = None
    try:
        with Tags(cfg) as t:
            t.read("trigger")                     # one is already running (src.plc.tags in a terminal): use it
    except Exception:
        proc = start_sim(cfg)
    by_name = {name: code for code, name in codes.items()}
    sim = {"proc": proc, "codes": codes, "kit": by_name.get(kit_name or "KIT-A", min(codes)),
           "last": "press SPACE to send a kit", "queue": queue.Queue()}

    def worker():
        with Tags(cfg) as t:
            line = Line(t, cfg)
            while (code := sim["queue"].get()) is not None:
                sim["last"] = "inspecting..."
                out = line.cycle(code)
                sim["last"] = (f"cycle {out['cycle_id']} {'RELEASE' if out['released'] else 'HOLD'}: {out['why']}")
                on_line(f"PLC   cycle {out['cycle_id']:>4}  {codes.get(code)}  "
                        f"{'RELEASE' if out['released'] else 'HOLD   '}  {out['ms']:>4} ms  {out['why']}")

    threading.Thread(target=worker, daemon=True).start()
    print(f"Simulated PLC {'started' if proc else 'already running, using it'}: "
          f"SPACE = send a kit, 1/2 = which kit ({', '.join(f'{k} {v}' for k, v in sorted(codes.items()))})")
    return sim


def main():
    parser = argparse.ArgumentParser(description="Inspection station (the PLC is SIMULATED)")
    parser.add_argument("--images", help="use photos from this folder instead of the camera")
    parser.add_argument("--show", action="store_true", help="window with the live view and the last result")
    parser.add_argument("--no-history", action="store_true", help="don't record inspections")
    parser.add_argument("--collect", action="store_true", help='start with "save for training" on (t toggles)')
    parser.add_argument("--kit", help="kit for bench inspections (SPACE). Default: the kit the PLC sent last "
                                      "(KIT-A before the first one). With --plc-sim: the kit the PLC starts with")
    parser.add_argument("--plc-sim", action="store_true",
                        help="run the simulated PLC inside: SPACE = PLC sends a kit, 1/2 = which kit (needs --show)")
    parser.add_argument("--every", type=float, help="with --plc-sim: the PLC sends a kit every N seconds by itself")
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
    state = {"collect": args.collect and history is not None, "note": "", "plc": "PLC: connecting..."}
    inspect_lock = threading.Lock()               # PLC triggers and SPACE can't run the model at the same time

    def inspect(frame, kit_id):
        with inspect_lock:
            return inspector.inspect(frame, kit_id)

    def on_result(r):
        if r.get("recovered"):                    # health fault cleared by itself: frames are back
            state["note"] = "camera back: fault cleared, heartbeat running"
            print(state["note"])
            return
        if history:
            try:
                history.record(r, model, train=state["collect"])
            except Exception as e:                # a full disk must not stop the line: warn, keep inspecting
                print(f"WARNING: history not saved: {e}")
        label = (f"cycle {r['cycle_id']:>4}" if r["cycle_id"] is not None
                 else "idle      " if r.get("idle") else "bench     ")
        if "error" in r:
            print(f"{label}  FAULT   {r['error']}")
            state["note"] = f"FAULT: {r['error']}"[:60]
            return
        d = r["decision"]
        if r["cycle_id"] is not None:
            state["plc_kit"] = r["kit_id"]        # SPACE bench tests follow the line's current kit
        print(f"{label}  {frames.name:<16} {r['kit_id'] or '-':<7} {d.result:<7} "
              f"photo {r['ms_frame']:>4.0f} ms  total {r['ms']:>4.0f} ms  {'; '.join(d.reasons)}")
        if args.show:
            last["img"] = draw(r["frame"], r["out"], high)
            state["note"] = (f"last: cycle {r['cycle_id']}, kit {r['kit_id']} from the PLC" if r["cycle_id"] is not None
                             else f"last: SPACE bench test, kit {r['kit_id']}"
                             + (" (--kit)" if args.kit else " (PLC's last kit)" if state.get("plc_kit") else " (default)"))

    def bench_kit():
        return args.kit or state.get("plc_kit") or "KIT-A"

    def bench():
        """SPACE: inspect the tray now, without the PLC (no release; recorded like any other)."""
        t0 = time.monotonic()
        try:
            frame = frames.get(t0)
            ms_frame = round((time.monotonic() - t0) * 1000, 1)
            out = inspect(frame, bench_kit())
        except Exception as e:
            on_result({"cycle_id": None, "kit_id": bench_kit(), "error": f"{type(e).__name__}: {e}"})
            return
        on_result({"cycle_id": None, "kit_id": bench_kit(), "decision": out["decision"], "out": out, "frame": frame,
                   "ms_frame": ms_frame, "ms": round((time.monotonic() - t0) * 1000, 1)})

    station = VisionSide(inspect, frames.get, on_result=on_result, health=frames.health)

    def plc_status(text):
        if text != state["plc"]:                  # print changes only, not every retry
            print(text)
        state["plc"] = text

    def serve_plc():
        """Answer PLC triggers, reconnecting by itself; without a PLC the window still works (SPACE)."""
        station.serve(on_status=plc_status)
    sim = None
    if args.plc_sim:
        sim = start_plc_sim(args.kit, on_line=lambda text: print(text))
    source = args.images or "camera"
    print(f"Station ready. Model {model}, photos from {source}"
          f"{', history in data/history/' if history else ''}. Waiting for PLC triggers; Ctrl+C to stop.")
    try:
        if not args.show:
            station.serve(on_status=plc_status)
        else:
            worker = threading.Thread(target=serve_plc, daemon=True)
            worker.start()
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            marked = history.marked() if history else 0
            names = sorted(sim["codes"].items()) if sim else []         # [(1, KIT-A), (2, KIT-B)]
            keys = ("SPACE = PLC sends a kit   1/2 = which kit   t = save   u = undo   q = quit" if sim
                    else "SPACE = inspect now   t = save on/off   u = undo   q = quit")
            next_auto = time.monotonic() + (args.every or 0)
            while True:
                line = (f"PLC sends {sim['codes'][sim['kit']]}: " + sim["last"]) if sim else ""
                panel = status_panel(560, state["collect"], marked, state["note"], state["plc"], line, keys)
                cv2.imshow(WINDOW, side_by_side(frames.latest()[0], last["img"], panel))
                key = cv2.waitKey(50) & 0xFF
                if key == ord("q") or window_closed():
                    break
                auto = sim and args.every and time.monotonic() >= next_auto
                if sim and (key == ord(" ") or auto):
                    next_auto = time.monotonic() + (args.every or 0)
                    sim["queue"].put(sim["kit"])
                elif key == ord(" "):
                    bench()
                elif sim and ord("1") <= key < ord("1") + len(names):
                    sim["kit"] = names[key - ord("1")][0]
                    sim["last"] = "press SPACE to send it"
                if not history:
                    continue
                if key == ord("t"):
                    state["collect"] = not state["collect"]
                    state["note"] = f"saving {'ON' if state['collect'] else 'OFF'} from the next kit"
                    print(state["note"])
                elif key == ord("u"):
                    row = history.unmark_last()
                    what = "nothing to undo" if row is None else (
                        f"cycle {row['cycle_id']}" if row["cycle_id"] is not None else f"bench photo {row['id']}")
                    state["note"] = what if row is None else f"undone: {what} won't be used"
                    print(state["note"])
                marked = history.marked()
    except KeyboardInterrupt:
        pass
    finally:
        station.stop.set()
        if sim:
            sim["queue"].put(None)
            if sim["proc"]:
                sim["proc"].terminate()
        if isinstance(frames, CameraFrames):
            frames.feed.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
