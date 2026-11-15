"""Live camera view with readouts, for setting up the rig and tuning the camera.

Run from the repo root:
    uv run python -m src.capture.preview
    uv run python -m src.capture.preview --size 640x480   # optional smaller view

Size and frame rate come from station.yaml (1280x720 at 15 fps). On WSL the USB
passthrough can't keep up with 720p at 30 fps, so the config uses 15.
Sharpness numbers aren't comparable between sizes or zoom levels.

Keys (click the window first so it gets your key presses). No Shift needed:
Shift+letter doesn't reach OpenCV through WSL's window system.
    1 2 3 4   pick a setting: exposure, white balance, gain, zoom
    -  =      turn the picked setting down / up
    c         toggle centre grid
    s         save current values to configs/station.yaml
    q / Esc   quit

Readouts:
    brightness  mean pixel value, 0-255
    clipped     % of pixels blown out to pure white. Glare on steel shows up here;
                you want it near 0% on the tray area.
    sharpness   how much fine detail is in focus (variance of the Laplacian).
                The camera is fixed-focus: move it up/down and watch this number;
                the height where it peaks is your mount height.

Zoom is the camera cropping its own sensor, which has more pixels than the 720p
it sends, so zoom 2 gives about twice the real detail of cropping in software
(tested Nov 15). Best setup: raise the camera until it's sharp, then zoom in so
the tray fills the frame.
"""

import argparse
import re

import cv2

from src.capture.camera import CONFIG, load_camera_config, open_camera, quiet_jpeg_warnings, set_controls

WINDOW = "kit-inspect camera  (q to quit)"

# The settings you can tune: (name in v4l2, label, min, max, how to step down, how to step up).
# Limits come from v4l2-ctl --list-ctrls. Exposure stops at 660 (= 66 ms): any longer
# than one frame at 15 fps and the camera has to drop its frame rate.
SETTINGS = [
    ("exposure_time_absolute", "exposure", 3, 660, lambda v: v * 0.8, lambda v: v * 1.25 + 1),
    ("white_balance_temperature", "white bal", 2800, 6500, lambda v: v - 200, lambda v: v + 200),
    ("gain", "gain", 16, 255, lambda v: v - 8, lambda v: v + 8),  # below 16 snaps to 16 while streaming
    ("zoom_absolute", "zoom", 1, 5, lambda v: v - 1, lambda v: v + 1),
]


def measure(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clipped = (frame.max(axis=2) >= 250).mean() * 100
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return gray.mean(), clipped, sharpness


def draw_overlay(frame, lines, show_grid):
    out = frame.copy()
    h, w = out.shape[:2]
    if show_grid:  # thirds plus a centre cross, to line the tray up the same way every time
        for x in (w // 3, 2 * w // 3):
            cv2.line(out, (x, 0), (x, h), (0, 255, 255), 1)
        for y in (h // 3, 2 * h // 3):
            cv2.line(out, (0, y), (w, y), (0, 255, 255), 1)
        cv2.drawMarker(out, (w // 2, h // 2), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)
    # Dark see-through panel behind the text. (A thick black "outline" pass doesn't
    # work: OpenCV spaces thicker text wider, so it drifts out from under the white.)
    font, scale, line_h = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 26
    lines = [t for t in lines if t]
    if lines:
        width = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines)
        x2, y2 = min(w, 10 + width + 10), 8 + line_h * len(lines) + 4
        panel = out[0:y2, 0:x2]
        out[0:y2, 0:x2] = cv2.addWeighted(panel, 0.35, panel * 0, 0.65, 0)  # 65% black
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 28 + i * line_h), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_to_config(values):
    """Update just these numbers in station.yaml, keeping its comments intact."""
    text = CONFIG.read_text()
    for name, value in values.items():
        text = re.sub(rf"(\n\s+{name}: )\d+", rf"\g<1>{value}", text)
    CONFIG.write_text(text)


def main():
    parser = argparse.ArgumentParser(description="Live camera preview")
    parser.add_argument("--size", help="WIDTHxHEIGHT, e.g. 640x480 (default: from station.yaml)")
    args = parser.parse_args()

    quiet_jpeg_warnings()
    cfg = load_camera_config()
    if args.size:
        cfg["width"], cfg["height"] = (int(n) for n in args.size.lower().split("x"))
    device = cfg["device"]
    cap = open_camera(cfg)
    values = {name: cfg["controls"][name] for name, *_ in SETTINGS}
    picked = 0  # index into SETTINGS
    show_grid = True
    status = ""

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)  # resizable
    cv2.resizeWindow(WINDOW, cfg["width"], cfg["height"])
    print(__doc__.split("Readouts:")[0])

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost the camera. On WSL: usbipd.exe attach --wsl --busid 8-4")
            break
        brightness, clipped, sharpness = measure(frame)
        setting_text = "   ".join(
            f"{'>' if i == picked else ' '}[{i + 1}] {label} {values[name]}"
            for i, (name, label, *_) in enumerate(SETTINGS)
        )
        lines = [
            setting_text + "    (-/= to change)",
            f"brightness {brightness:5.1f}   clipped {clipped:4.1f}%   sharpness {sharpness:6.0f}",
            status,
        ]
        cv2.imshow(WINDOW, draw_overlay(frame, lines, show_grid))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif ord("1") <= key < ord("1") + len(SETTINGS):
            picked = key - ord("1")
        elif key in (ord("-"), ord("=")):
            name, _, lo, hi, down, up = SETTINGS[picked]
            step = down if key == ord("-") else up
            values[name] = int(max(lo, min(hi, step(values[name]))))
            set_controls(device, {name: values[name]})
        elif key == ord("c"):
            show_grid = not show_grid
        elif key == ord("s"):
            save_to_config(values)
            status = f"saved to station.yaml: {values}"
            print(status)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the X
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
