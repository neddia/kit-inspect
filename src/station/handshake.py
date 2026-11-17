"""The vision station's side of the PLC handshake (the PLC is SIMULATED; see src/plc/).

Two loops:
  heartbeat  toggles Vision_Heartbeat every heartbeat_ms while the station is healthy,
             so the PLC can tell "busy" from "dead". Its own thread and connection.
  main       watches Insp_Trigger. On a rising edge (0 -> 1): Busy = 1, read the kit
             code, take a photo, inspect, write Result + ResultCycleID, Done = 1,
             Busy = 0. When the PLC sets Ack: Done = 0.

If anything fails (camera gone, model error), the station sets Vision_Fault, stops
the heartbeat and never sets Done. The PLC then times out and holds the kit.
Silence is always a hold; only a PASS for the current cycle releases.

Part 2 (no camera yet): photos come from a folder, one per trigger.
    uv run python -m src.station.handshake --images data/images/val01
"""

import argparse
import itertools
import threading
import sys
import time
from pathlib import Path

import cv2

from src.plc.tags import RESULT_CODES, Tags, load_config


class VisionSide:
    def __init__(self, inspect, get_frame, cfg=None, codes=None, port=None, on_result=print):
        """inspect(frame, kit_id) -> dict with "decision" (Inspector.inspect); get_frame() -> BGR image."""
        loaded_cfg, loaded_codes = load_config()
        self.cfg = cfg or loaded_cfg
        self.codes = codes if codes is not None else loaded_codes
        self.port = port
        self.inspect, self.get_frame, self.on_result = inspect, get_frame, on_result
        self.faulted = False
        self.stop = threading.Event()

    def heartbeat(self):
        with Tags(self.cfg, port=self.port) as t:
            beat = False
            while not self.stop.is_set():
                if not self.faulted:
                    beat = not beat
                    t.write("heartbeat", beat)
                self.stop.wait(self.cfg["heartbeat_ms"] / 1000)

    def handle_trigger(self, t):
        t0 = time.monotonic()
        t.write("busy", True)
        cycle_id, code = t.read("cycle_id"), t.read("kit_id")
        kit_id = self.codes.get(code, f"code {code}" if code else "")
        try:
            frame = self.get_frame()
            out = self.inspect(frame, kit_id)
        except Exception as e:                     # camera gone, model error: fault, no answer -> PLC holds
            self.faulted = True
            t.write("fault", True)
            t.write("busy", False)
            self.on_result({"cycle_id": cycle_id, "kit_id": kit_id, "error": f"{type(e).__name__}: {e}"})
            return
        decision = out["decision"]
        t.write("result", RESULT_CODES[decision.result])
        t.write("result_cycle_id", cycle_id)
        if self.faulted:                           # a good cycle clears an earlier fault
            self.faulted = False
            t.write("fault", False)
        t.write("done", True)
        t.write("busy", False)
        self.on_result({"cycle_id": cycle_id, "kit_id": kit_id, "decision": decision, "out": out,
                        "ms": round((time.monotonic() - t0) * 1000)})

    def run(self):
        """Serve triggers until self.stop is set."""
        poll_s = self.cfg["poll_ms"] / 1000
        with Tags(self.cfg, port=self.port) as t:
            for role in ("busy", "done", "fault"):
                t.write(role, False)
            beat = threading.Thread(target=self.heartbeat, daemon=True)
            beat.start()
            last_trigger = t.read("trigger")       # a trigger already up at start-up isn't a new kit
            try:
                while not self.stop.is_set():
                    trigger = t.read("trigger")
                    if trigger and not last_trigger:
                        self.handle_trigger(t)
                    elif t.read("done") and t.read("ack"):
                        t.write("done", False)
                    last_trigger = trigger
                    time.sleep(poll_s)
            finally:
                self.stop.set()
                beat.join()


def folder_frames(folder):
    """Endless photos from a folder, in order (stands in for the camera in part 2)."""
    paths = sorted(Path(folder).glob("*.jpg"))
    if not paths:
        raise SystemExit(f"no .jpg files in {folder}")
    names = itertools.cycle(paths)
    state = {}

    def get_frame():
        state["path"] = next(names)
        return cv2.imread(str(state["path"]))
    return get_frame, state


def main():
    parser = argparse.ArgumentParser(description="Vision station: answer PLC triggers (SIMULATED PLC)")
    parser.add_argument("--images", required=True, help="folder of tray photos used in place of the camera")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # print each line at once, even into a log file

    from src.capture.camera import quiet_jpeg_warnings
    from src.inspection.pipeline import Inspector

    quiet_jpeg_warnings()
    inspector = Inspector()
    get_frame, state = folder_frames(args.images)
    inspector.inspect(get_frame(), "KIT-A")        # warm-up: the first inspection is slow
    get_frame, state = folder_frames(args.images)

    def show(r):
        name = state["path"].name if "path" in state else "-"
        if "error" in r:
            print(f"cycle {r['cycle_id']:>4}  {name:<16} FAULT   {r['error']}")
            return
        d = r["decision"]
        print(f"cycle {r['cycle_id']:>4}  {name:<16} {r['kit_id']:<6} {d.result:<7} {r['ms']:>4} ms  {'; '.join(d.reasons)}")

    station = VisionSide(inspector.inspect, get_frame, on_result=show)
    print(f"Vision station ready ({inspector.cfg['model']}). Waiting for triggers; Ctrl+C to stop.")
    try:
        station.run()
    except KeyboardInterrupt:
        station.stop.set()


if __name__ == "__main__":
    main()
