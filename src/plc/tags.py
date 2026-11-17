"""The handshake tags, and the simulated PLC that holds them. The PLC is SIMULATED.

A real Logix PLC keeps tags (named variables) in its memory and runs ladder
logic on them. Other devices read and write those tags over EtherNet/IP.
Here, cpppo plays the PLC's memory (it only holds tags, no logic), the vision
station talks to it with pylogix exactly as it would to a real PLC, and
src/plc/line.py plays the PLC's logic.

Run the simulated PLC (leave it running in its own terminal):
    uv run python -m src.plc.tags

Code uses role names ("trigger", "result"); station.yaml maps them to tag names
("Insp_Trigger", "Insp_Result"), so renaming a tag for a real PLC is config only.
"""

import subprocess
import sys
import time
from pathlib import Path

import yaml
from pylogix import PLC

REPO = Path(__file__).resolve().parents[2]

TYPES = {                     # role -> PLC data type (written by: see the plan's tag table)
    "trigger": "BOOL",         # PLC
    "cycle_id": "DINT",        # PLC
    "kit_id": "DINT",          # PLC (code from kits.yaml `codes`)
    "busy": "BOOL",            # vision
    "result": "DINT",          # vision: 1 PASS, 2 FAIL, 3 REVIEW
    "result_cycle_id": "DINT", # vision
    "done": "BOOL",            # vision
    "ack": "BOOL",             # PLC
    "heartbeat": "BOOL",       # vision
    "fault": "BOOL",           # vision
    "release_permit": "BOOL",  # PLC
}
RESULT_CODES = {"PASS": 1, "FAIL": 2, "REVIEW": 3}


def load_config():
    station = yaml.safe_load((REPO / "configs" / "station.yaml").read_text())
    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())
    return station["plc"], kits.get("codes", {})


class PLCError(RuntimeError):
    pass


class Tags:
    """Read and write handshake tags by role. One connection; use one Tags per thread."""

    def __init__(self, cfg=None, host=None, port=None):
        self.cfg = cfg or load_config()[0]
        self.names = self.cfg["tags"]
        self.comm = PLC(host or self.cfg["host"], port=port or self.cfg["port"], timeout=2.0)

    def read(self, role):
        r = self.comm.Read(self.names[role])
        if r.Status != "Success":
            raise PLCError(f"read {self.names[role]}: {r.Status}")
        return r.Value

    def write(self, role, value):
        r = self.comm.Write(self.names[role], value)
        if r.Status != "Success":
            raise PLCError(f"write {self.names[role]} = {value!r}: {r.Status}")

    def close(self):
        self.comm.Close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sim_command(cfg, port=None):
    tags = [f"{cfg['tags'][role]}={kind}" for role, kind in TYPES.items()]
    return [sys.executable, "-m", "cpppo.server.enip", "-a", f"{cfg['host']}:{port or cfg['port']}", "--no-print", *tags]


def start_sim(cfg=None, port=None, wait_s=10.0):
    """Start the simulated PLC in the background; returns once it answers. Stop it with .terminate()."""
    cfg = cfg or load_config()[0]
    proc = subprocess.Popen(sim_command(cfg, port), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            with Tags(cfg, port=port) as t:
                t.read("trigger")
            return proc
        except Exception:
            time.sleep(0.2)
    proc.terminate()
    raise PLCError("simulated PLC didn't start")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    cfg, _ = load_config()
    print(f"Simulated PLC (cpppo) on {cfg['host']}:{cfg['port']} with tags:")
    for role, kind in TYPES.items():
        print(f"  {cfg['tags'][role]:<20} {kind}")
    print("Ctrl+C to stop.")
    try:
        done = subprocess.run(sim_command(cfg), stderr=subprocess.PIPE, text=True)
    except KeyboardInterrupt:
        return
    if done.returncode:
        print(done.stderr.strip().splitlines()[-1] if done.stderr.strip() else "simulator stopped")
        print(f"Port {cfg['port']} busy? An old simulator may still be running: pkill -f cpppo.server.enip")


if __name__ == "__main__":
    main()
