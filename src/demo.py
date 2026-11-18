"""The demo: the PLC's operator panel and the vision station, started together as two separate apps.

    uv run python -m src.demo                                   # camera
    uv run python -m src.demo --images data/images/hardval2     # photos from a folder, no camera

Two processes, like a real cell. Close the station window (or it crashes) and the
PLC panel keeps running: its heartbeat lamp goes dark and RELEASE drops to HOLD by
itself. Start the station again and it reconnects. Ctrl+C here stops both.

In the PLC panel: 1 / 2 = which kit, SPACE = send a kit, a = auto, q = quit.
"""

import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description="PLC panel + vision station, as two separate apps")
    parser.add_argument("--images", help="station uses photos from this folder instead of the camera")
    args = parser.parse_args()

    station_cmd = [sys.executable, "-m", "src.station.run", "--show"] + (["--images", args.images] if args.images else [])
    plc = subprocess.Popen([sys.executable, "-m", "src.plc.panel"])
    time.sleep(1.5)                               # let the panel start the simulated PLC (the station would retry anyway)
    station = subprocess.Popen(station_cmd)
    print("PLC panel and vision station started. Close either window; Ctrl+C here stops both.")
    try:
        while plc.poll() is None or station.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (station, plc):
            if p.poll() is None:
                p.terminate()
        for p in (station, plc):
            p.wait(timeout=10)


if __name__ == "__main__":
    main()
