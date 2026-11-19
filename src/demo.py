"""Run the simulated PLC and vision station with one operator window."""
import argparse
import os
import subprocess
import sys
import tempfile

from src.plc import panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", help="replay photos from a folder instead of the camera")
    args = parser.parse_args()
    ram = "/dev/shm" if os.access("/dev/shm", os.W_OK) else None
    with tempfile.TemporaryDirectory(prefix="kit-demo-", dir=ram) as preview:
        command = [sys.executable, "-m", "src.station.run", "--preview-dir", preview]
        if args.images:
            command.extend(["--images", args.images])
        vision = subprocess.Popen(command)
        try:
            panel.main(["--preview-dir", preview])
        finally:
            if vision.poll() is None:
                vision.terminate()
            vision.wait()


if __name__ == "__main__":
    main()
