"""Open the camera with locked settings, and prove they're locked.

Webcams default to auto exposure and auto white balance, so the "same" scene
looks different from frame to frame. For inspection we want every image taken
under identical settings. This camera also forgets its settings whenever it's
unplugged or the PC reboots, so we apply them on EVERY open and read them back.

Try it (from the repo root):
    uv run python -m src.capture.camera
"""

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import yaml

# Note: expect "Corrupt JPEG data: N extraneous bytes before marker 0xd?" on
# stderr for most frames. The C510 pads its JPEGs with a few junk bytes before
# the restart markers (0xd0-0xd7); the decoder skips them and the image is fine.

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "station.yaml"
WARMUP_FRAMES = 10  # frames to throw away so new settings have taken effect


def quiet_jpeg_warnings():
    """Hide the C510's harmless 'Corrupt JPEG data' spam, keep Python errors.

    The JPEG library writes straight to the process's stderr (file descriptor 2),
    below Python. We point fd 2 at /dev/null, and give Python's own sys.stderr a
    copy of the real stderr so tracebacks still show. Side effect: OpenCV's other
    C-level warnings are hidden too.
    """
    real_stderr = os.dup(2)
    os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
    sys.stderr = os.fdopen(real_stderr, "w", buffering=1)


def load_camera_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text())["camera"]


def set_controls(device, controls):
    """Set camera controls one at a time, in the order given.

    We use the v4l2-ctl tool instead of OpenCV's cap.set() because OpenCV's
    property numbers for exposure differ between camera drivers; v4l2-ctl uses
    the exact control names the camera itself reports.
    """
    for name, value in controls.items():
        subprocess.run(["v4l2-ctl", "-d", device, "--set-ctrl", f"{name}={value}"], check=True)


def read_controls(device, names):
    """Ask the camera what each control is actually set to now."""
    out = subprocess.run(
        ["v4l2-ctl", "-d", device, "--get-ctrl", ",".join(names)],
        check=True, capture_output=True, text=True,
    ).stdout
    # Lines look like "auto_exposure: 1 (Manual Mode)"; keep just the number.
    values = {}
    for line in out.splitlines():
        name, value = line.split(":", 1)
        values[name.strip()] = int(value.split()[0])
    return values


def find_usb_camera(usb_id, video_class=Path("/sys/class/video4linux")):
    """Find the capture interface of this USB camera, including after kernel renumbering."""
    for node in sorted(video_class.glob("video*")):
        try:
            if (node / "index").read_text().strip() != "0":
                continue
            for parent in (node / "device").resolve().parents:
                if (parent / "idVendor").is_file():
                    identity = ":".join((parent / name).read_text().strip() for name in ("idVendor", "idProduct"))
                    if identity.lower() == usb_id.lower():
                        return "/dev/" + node.name
                    break
        except FileNotFoundError:
            continue  # unplugged while enumerating
    return None


def station_camera_config(cfg):
    """Resolve the camera for each open attempt. WSL loses USB forwarding on unplug."""
    usb_id = cfg.get("usb_id")
    if not usb_id:
        return cfg
    device = find_usb_camera(usb_id)
    if device is None:
        tool = shutil.which("usbipd.exe") if "microsoft" in platform.release().lower() else None
        if tool:
            result = subprocess.run([tool, "attach", "--wsl", "--hardware-id", usb_id],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or "waiting for the shared USB webcam")
            device = find_usb_camera(usb_id)
        if device is None:
            raise RuntimeError(f"waiting for USB webcam {usb_id}")
    return dict(cfg, device=device)


def open_camera(cfg=None):
    """Open the camera, lock its settings, verify them, and return the capture."""
    cfg = cfg or load_camera_config()
    device = cfg["device"]

    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Can't open {device}. On WSL, is it attached? (usbipd attach --wsl --busid 8-4)")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg["fourcc"]))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        cap.set(cv2.CAP_PROP_FPS, cfg["fps"])

        # Start the stream BEFORE setting controls. Tested on the C510: the first
        # read() starts the stream and resets gain to 16, wiping whatever we set.
        cap.read()
        set_controls(device, cfg["controls"])
        for _ in range(WARMUP_FRAMES):
            cap.read()

        # Trust, but verify: a control the camera silently ignored is how you end
        # up with a training set shot on auto exposure.
        actual = read_controls(device, list(cfg["controls"]))
        wrong = {k: (v, actual.get(k)) for k, v in cfg["controls"].items() if actual.get(k) != v}
        if wrong:
            raise RuntimeError(f"Camera ignored settings (wanted, got): {wrong}")

        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Camera opened but returned no frame. On WSL, try detach + re-attach.")
        h, w = frame.shape[:2]
        if (w, h) != (cfg["width"], cfg["height"]):
            raise RuntimeError(f"Asked for {cfg['width']}x{cfg['height']}, got {w}x{h}")
    except Exception:
        cap.release()
        raise
    return cap


if __name__ == "__main__":
    cfg = load_camera_config()
    cap = open_camera(cfg)
    ok, frame = cap.read()
    cap.release()
    out = REPO / "reports" / "stage1" / "camera_test.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"Camera OK: {frame.shape[1]}x{frame.shape[0]}, all {len(cfg['controls'])} settings locked and verified")
    print(f"Mean brightness {frame.mean():.0f}/255 (aim for the parts to be clearly visible, not washed out)")
    print(f"Saved {out.relative_to(REPO)}")
