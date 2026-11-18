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

Health check (Nov 18): the main loop also asks health() every poll, so a camera
unplugged while the line is IDLE faults at once (not only when the next kit
arrives), and the fault clears by itself when frames come back: the heartbeat
resumes and the next kit is inspected normally. A fault from an inspection error
(model crash) still clears only on the next good cycle.

Run the station with: uv run python -m src.station.run   (see run.py)
"""

import threading
import time

from src.plc.tags import RESULT_CODES, Tags, load_config


class VisionSide:
    def __init__(self, inspect, get_frame, cfg=None, codes=None, port=None, on_result=print, health=None):
        """inspect(frame, kit_id) -> dict with "decision" (Inspector.inspect).
        get_frame(after) -> BGR image taken after time.monotonic() value `after` (the trigger).
        on_result(dict) is called after Done is written, so saving history doesn't delay the PLC.
        health() -> None when fine, else what's wrong (e.g. "camera lost: no frames for 1.2 s")."""
        loaded_cfg, loaded_codes = load_config()
        self.cfg = cfg or loaded_cfg
        self.codes = codes if codes is not None else loaded_codes
        self.port = port
        self.inspect, self.get_frame, self.on_result = inspect, get_frame, on_result
        self.health = health or (lambda: None)
        self.faulted = False
        self.fault_from_health = False            # health faults clear themselves; inspection faults need a good cycle
        self.stop = threading.Event()

    def heartbeat(self, over):
        """Toggle Vision_Heartbeat until the station stops or this PLC session is over."""
        with Tags(self.cfg, port=self.port) as t:
            beat = False
            while not self.stop.is_set() and not over.is_set():
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
            # a camera error is a health fault (clears when frames return); anything else needs a good cycle
            self.fault_from_health = self.fault_from_health or frame is None
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
            self.faulted, self.fault_from_health = False, False
            t.write("fault", False)
        t.write("done", True)
        t.write("busy", False)
        self.on_result({"cycle_id": cycle_id, "kit_id": kit_id, "decision": decision, "out": out, "frame": frame,
                        "ms_frame": ms_frame, "ms": round((time.monotonic() - t0) * 1000, 1)})

    def check_health(self, t):
        """Fault the moment health() reports a problem, clear it when the problem goes away."""
        problem = self.health()
        if problem and not self.faulted:
            self.faulted, self.fault_from_health = True, True
            t.write("fault", True)
            self.on_result({"cycle_id": None, "kit_id": None, "idle": True, "error": problem})
        elif not problem and self.faulted and self.fault_from_health:
            self.faulted, self.fault_from_health = False, False
            t.write("fault", False)
            self.on_result({"cycle_id": None, "kit_id": None, "idle": True, "recovered": True})

    def serve(self, on_status=print, retry_s=2.0):
        """run() forever, reconnecting every retry_s when the PLC is unreachable or drops out
        (PLC rebooted, cable pulled, simulator started after the station). Until self.stop."""
        while not self.stop.is_set():
            try:
                self.run(on_connected=lambda: on_status("PLC: connected, answering triggers"))
            except Exception as e:
                on_status(f"PLC: not connected ({type(e).__name__}: {str(e)[:60]}), retrying every {retry_s:.0f} s")
                self.stop.wait(retry_s)

    def run(self, on_connected=None):
        """One PLC session: serve triggers until self.stop is set; raises if the connection fails."""
        poll_s = self.cfg["poll_ms"] / 1000
        with Tags(self.cfg, port=self.port) as t:
            for role in ("busy", "done", "fault"):
                t.write(role, False)
            if self.faulted:                       # still faulted from before a reconnect: tell the PLC again
                t.write("fault", True)
            if on_connected:
                on_connected()
            over = threading.Event()
            beat = threading.Thread(target=self.heartbeat, args=(over,), daemon=True)
            beat.start()
            last_trigger = t.read("trigger")       # a trigger already up at start-up isn't a new kit
            try:
                while not self.stop.is_set():
                    self.check_health(t)
                    trigger = t.read("trigger")
                    if trigger and not last_trigger:
                        self.handle_trigger(t)
                    elif t.read("done") and t.read("ack"):
                        t.write("done", False)
                    last_trigger = trigger
                    time.sleep(poll_s)
            finally:
                over.set()
                beat.join()

