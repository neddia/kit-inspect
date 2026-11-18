"""The PLC's side of the handshake, played in Python. The PLC is SIMULATED.

In a real cell this is ladder logic in the PLC. It does the same steps:

    PLC (this file)                                Vision (src/station/handshake.py)
    1. Release_Permit = 0 (default: hold)
       Insp_KitID = kit code, Insp_CycleID += 1,
       Insp_Trigger = 1                     ───>   sees the trigger rise, Busy = 1,
                                                   reads the kit, photographs, inspects
                                            <───   Insp_Result, Insp_ResultCycleID,
                                                   Done = 1, Busy = 0
    2. Result for THIS cycle, PASS, heartbeat
       alive, no fault?  -> Release_Permit = 1
       anything else     -> stays 0 (hold the kit)
       Insp_Ack = 1, Insp_Trigger = 0      ───>   sees Ack, Done = 0
    3. Done back to 0 -> Insp_Ack = 0. Ready for the next kit.

Fail-safe: no Done within timeout_s = hold. No heartbeat toggle within
heartbeat_timeout_s = hold. A result for another cycle = hold. Only a matched
PASS releases.

Between kits the PLC keeps scanning (watchdog(), every poll): if the vision
station's heartbeat stops or it raises a fault, Release_Permit drops at once,
even with nobody pressing anything. (Before Nov 18 a PASS could leave the
release on while the PLC sat waiting for the next kit.)

Run (with the simulated PLC and the vision station running):
    uv run python -m src.plc.line                    # ENTER sends the next kit (KIT-A)
    uv run python -m src.plc.line --kit KIT-B        # same trays, different recipe
    uv run python -m src.plc.line --every 2 --cycles 20
"""

import argparse
import queue
import sys
import threading
import time

from src.plc.tags import RESULT_CODES, Tags, load_config

RESULT_NAMES = {v: k for k, v in RESULT_CODES.items()}


class Line:
    def __init__(self, tags, cfg):
        self.tags = tags
        self.timeout_s = cfg["timeout_s"]
        self.heartbeat_timeout_s = cfg["heartbeat_timeout_s"]
        self.poll_s = cfg["poll_ms"] / 1000
        self.cycle_id = tags.read("cycle_id")
        self.last_beat = tags.read("heartbeat")
        self.last_beat_t = time.monotonic()

    def heartbeat_ok(self):
        beat = self.tags.read("heartbeat")
        if beat != self.last_beat:
            self.last_beat, self.last_beat_t = beat, time.monotonic()
        return time.monotonic() - self.last_beat_t < self.heartbeat_timeout_s

    def watchdog(self):
        """Between kits: vision heartbeat stopped or faulted -> Release_Permit off now.
        Returns dict(heartbeat_ok, fault, released, dropped) for a panel to show."""
        beat_ok, fault = self.heartbeat_ok(), self.tags.read("fault")
        released = self.tags.read("release_permit")
        dropped = None
        if released and (not beat_ok or fault):
            self.tags.write("release_permit", False)
            released = False
            dropped = "vision heartbeat lost -> HOLD" if not beat_ok else "vision fault -> HOLD"
        return {"heartbeat_ok": beat_ok, "fault": fault, "released": released, "dropped": dropped}

    def wait_for(self, role, value, timeout_s):
        """Poll a tag until it equals value (True) or the time runs out (False). Watches the heartbeat meanwhile."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self.heartbeat_ok()
            if self.tags.read(role) == value:
                return True
            time.sleep(self.poll_s)
        return False

    def clear_stale_done(self):
        """A late answer from a timed-out cycle may still be up: acknowledge it away first."""
        if self.tags.read("done"):
            self.tags.write("ack", True)
            self.wait_for("done", False, self.timeout_s)
            self.tags.write("ack", False)

    def cycle(self, kit_code):
        """One kit: trigger, wait, decide release. Returns what happened."""
        t = self.tags
        t.write("release_permit", False)          # default: hold
        self.clear_stale_done()
        self.cycle_id += 1
        t.write("kit_id", kit_code)
        t.write("cycle_id", self.cycle_id)
        t0 = time.monotonic()
        t.write("trigger", True)

        answered = self.wait_for("done", True, self.timeout_s)
        ms = round((time.monotonic() - t0) * 1000)
        out = {"cycle_id": self.cycle_id, "kit_code": kit_code, "result": None, "released": False, "ms": ms}
        if not answered:
            t.write("trigger", False)
            out["why"] = f"timeout: no result within {self.timeout_s} s -> hold"
            return out

        code, result_cycle = t.read("result"), t.read("result_cycle_id")
        fault, beat_ok = t.read("fault"), self.heartbeat_ok()
        out["result"] = RESULT_NAMES.get(code, f"unknown code {code}")
        if result_cycle != self.cycle_id:
            out["why"] = f"result is for cycle {result_cycle}, not {self.cycle_id} -> hold"
        elif not beat_ok:
            out["why"] = "vision heartbeat lost -> hold"
        elif fault:
            out["why"] = "vision fault -> hold"
        elif code == RESULT_CODES["PASS"]:
            out["released"] = True
            out["why"] = "matched PASS -> release"
        else:
            out["why"] = f"{out['result']} -> hold"
        t.write("release_permit", out["released"])
        t.write("ack", True)
        t.write("trigger", False)
        self.wait_for("done", False, self.timeout_s)
        t.write("ack", False)
        return out


def main():
    parser = argparse.ArgumentParser(description="Play the PLC's side of the handshake (SIMULATED PLC)")
    parser.add_argument("--kit", default="KIT-A", help="kit name from kits.yaml, or a raw code number")
    parser.add_argument("--every", type=float, help="seconds between kits (default: press ENTER for each)")
    parser.add_argument("--cycles", type=int, default=0, help="stop after this many (0 = run until Ctrl+C)")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # print each line at once, even into a log file

    cfg, codes = load_config()
    by_name = {name: code for code, name in codes.items()}
    kit_code = int(args.kit) if args.kit.isdigit() else by_name[args.kit]
    enter = queue.Queue()
    if not args.every:
        def read_enter():
            while True:
                input()
                enter.put(1)
        threading.Thread(target=read_enter, daemon=True).start()
    with Tags(cfg) as tags:
        line = Line(tags, cfg)
        print(f"PLC (simulated) sending kit code {kit_code} ({codes.get(kit_code, 'not in kits.yaml')}). "
              + (f"A kit every {args.every} s." if args.every else "ENTER = next kit."))
        n, next_t = 0, time.monotonic() + (args.every or 0)
        try:
            while not args.cycles or n < args.cycles:
                due = time.monotonic() >= next_t if args.every else not enter.empty()
                if not due:
                    w = line.watchdog()               # between kits: keep watching the vision station
                    if w["dropped"]:
                        print(f"            HOLD     release dropped: {w['dropped']}")
                    time.sleep(line.poll_s)
                    continue
                if not args.every:
                    enter.get()
                next_t = time.monotonic() + (args.every or 0)
                out = line.cycle(kit_code)
                n += 1
                status = "RELEASE" if out["released"] else "HOLD   "
                print(f"cycle {out['cycle_id']:>4}  {status}  result {out['result'] or '-':<7} "
                      f"{out['ms']:>5} ms  {out['why']}")
        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    main()
