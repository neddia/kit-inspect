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

Run the station with: uv run python -m src.station.run   (see run.py)
"""

import threading
import time

from src.plc.tags import RESULT_CODES, Tags, load_config


class VisionSide:
    def __init__(self, inspect, get_frame, cfg=None, codes=None, port=None, on_result=print):
        """inspect(frame, kit_id) -> dict with "decision" (Inspector.inspect).
        get_frame(after) -> BGR image taken after time.monotonic() value `after` (the trigger).
        on_result(dict) is called after Done is written, so saving history doesn't delay the PLC."""
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
        frame, ms_frame = None, None
        try:
            frame = self.get_frame(t0)                 # a photo taken AFTER the trigger, never an older one
            ms_frame = round((time.monotonic() - t0) * 1000, 1)
            out = self.inspect(frame, kit_id)
        except Exception as e:                     # camera gone, model error: fault, no answer -> PLC holds
            self.faulted = True
            t.write("fault", True)
            t.write("busy", False)
            self.on_result({"cycle_id": cycle_id, "kit_id": kit_id, "frame": frame, "ms_frame": ms_frame,
                            "error": f"{type(e).__name__}: {e}"})
            return
        decision = out["decision"]
        t.write("result", RESULT_CODES[decision.result])
        t.write("result_cycle_id", cycle_id)
        if self.faulted:                           # a good cycle clears an earlier fault
            self.faulted = False
            t.write("fault", False)
        t.write("done", True)
        t.write("busy", False)
        self.on_result({"cycle_id": cycle_id, "kit_id": kit_id, "decision": decision, "out": out, "frame": frame,
                        "ms_frame": ms_frame, "ms": round((time.monotonic() - t0) * 1000, 1)})

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

