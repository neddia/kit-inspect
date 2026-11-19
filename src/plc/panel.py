"""The simulated PLC with an operator panel: the line's side of the cell, as its own app.

    PLC panel (this app)                         Vision station (src/station/run.py)
    kit selector, Send kit, auto mode            camera, what it found, why
    RELEASE / HOLD lamp = the real Release_Permit tag
    vision heartbeat and fault lamps

Two separate programs, like a real cell: close the vision app and this one keeps
running, sees the heartbeat stop, and drops the release to HOLD by itself
(Line.watchdog, every scan). Start them in any order; the station reconnects.

Keys (in the panel window):
    1 / 2   which kit the line is building (kits.yaml codes: 1 = KIT-A, 2 = KIT-B)
    SPACE   send the next kit: trigger -> vision inspects -> RELEASE or HOLD
    a       auto: a kit every 4 s (a again = off)
    q / X   quit (stops the simulated PLC if this app started it)

Run:
    uv run python -m src.plc.panel
    uv run python -m src.demo           # this panel and the station together
"""

import argparse
from pathlib import Path
import queue
import threading
import time

import cv2
import numpy as np
import yaml

from src.plc.line import Line
from src.plc.tags import Tags, load_config, start_sim

WINDOW = "PLC (simulated) - operator panel"
GREEN, RED, AMBER, GREY, WHITE = (60, 190, 60), (60, 60, 220), (0, 190, 255), (110, 110, 110), (235, 235, 235)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def kit_specs():
    from src.inspection.pipeline import REPO
    return yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]


def lamp(img, center, on, color, label, radius=22):
    cv2.circle(img, center, radius, color if on else (50, 50, 50), -1)
    cv2.circle(img, center, radius, (200, 200, 200), 2)
    cv2.putText(img, label, (center[0] + radius + 12, center[1] + 8), FONT, 0.7, WHITE, 2, cv2.LINE_AA)


def draw_panel(st, codes, specs):
    img = np.full((470, 760, 3), 28, np.uint8)
    cv2.putText(img, "SIMULATED PLC  -  operator panel", (20, 38), FONT, 0.85, WHITE, 2, cv2.LINE_AA)

    # kit selector
    for i, (code, name) in enumerate(sorted(codes.items())):
        x = 20 + i * 250
        sel = code == st["kit"]
        cv2.rectangle(img, (x, 60), (x + 230, 128), AMBER if sel else GREY, -1 if sel else 2)
        col = (20, 20, 20) if sel else WHITE
        cv2.putText(img, f"[{code}] {name}", (x + 12, 90), FONT, 0.8, col, 2, cv2.LINE_AA)
        spec = specs.get(name, {})
        cv2.putText(img, "  ".join(f"{n} {p}" + ("s" if n != 1 else "") for p, n in spec.items()), (x + 12, 116), FONT, 0.5, col, 1, cv2.LINE_AA)

    # the release lamp: the actual Release_Permit tag
    rel = st["released"]
    cv2.rectangle(img, (20, 150), (370, 300), GREEN if rel else RED, -1)
    cv2.putText(img, "RELEASE" if rel else "HOLD", (55 if rel else 110, 245), FONT, 2.0, (255, 255, 255), 5, cv2.LINE_AA)
    cv2.putText(img, "Release_Permit", (30, 290), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # vision station lamps
    lamp(img, (420, 180), st["heartbeat_ok"], GREEN, "vision heartbeat" + ("" if st["heartbeat_ok"] else " LOST"))
    lamp(img, (420, 235), st["fault"], RED, "vision fault")
    lamp(img, (420, 290), st["busy"], AMBER, "inspecting..." if st["busy"] else "idle")

    # last cycles
    cv2.putText(img, "last kits:", (20, 335), FONT, 0.6, GREY, 1, cv2.LINE_AA)
    for i, (text, released) in enumerate(st["log"][-4:][::-1]):
        cv2.putText(img, text, (20, 362 + i * 24), FONT, 0.55, GREEN if released else (90, 90, 255) if released is False
                    else WHITE, 1, cv2.LINE_AA)
    auto = f"   AUTO: a kit every {st['every']:.0f} s" if st["auto"] else ""
    cv2.putText(img, f"SPACE send kit   1/2 kit   a auto   q quit{auto}", (20, 455), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
    return img


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulated PLC with an operator panel")
    parser.add_argument("--kit", default="KIT-A", help="kit the line starts with")
    parser.add_argument("--every", type=float, default=4.0, help="seconds between kits in auto mode (a)")
    parser.add_argument("--preview-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    view = None
    window = WINDOW
    if args.preview_dir:
        from src.station.demo_view import DemoView
        view = DemoView(args.preview_dir)
        window = "Fastener kit inspection"

    cfg, codes = load_config()
    specs = kit_specs()
    proc = None
    try:
        with Tags(cfg) as t:
            t.read("trigger")                     # a simulated PLC is already running: use it
        print("Using the simulated PLC that's already running.")
    except Exception:
        proc = start_sim(cfg)
        print(f"Started the simulated PLC on {cfg['host']}:{cfg['port']}.")

    by_name = {name: code for code, name in codes.items()}
    st = {"kit": by_name.get(args.kit, min(codes)), "released": False, "heartbeat_ok": False, "fault": False,
          "busy": False, "log": [], "auto": False, "every": args.every, "stop": False}
    sends = queue.Queue()

    def scan():
        """Keep the scan running; a lost connection shows HOLD and an error, never frozen lamps."""
        while not st["stop"]:
            try:
                scan_once()
            except Exception as e:
                st.update(released=False, heartbeat_ok=False, busy=False, connected=False)
                st["log"].append((f"PLC connection error: {type(e).__name__}: {str(e)[:50]} (retrying)", False))
                print(st["log"][-1][0])
                time.sleep(1.0)

    def scan_once():
        """The PLC's scan: a kit when asked (or auto), the watchdog in between."""
        with Tags(cfg) as t:
            line = Line(t, cfg)
            next_auto = time.monotonic() + args.every
            while not st["stop"]:
                send = None
                try:
                    send = sends.get_nowait()
                except queue.Empty:
                    if st["auto"] and time.monotonic() >= next_auto:
                        send = st["kit"]
                if send is not None:
                    next_auto = time.monotonic() + args.every
                    st.update(busy=True, released=False)
                    out = line.cycle(send)
                    st["busy"] = False
                    text = (f"cycle {out['cycle_id']}  {codes.get(send)}  {'RELEASE' if out['released'] else 'HOLD'}"
                            f"  {out['ms']} ms  {out['why']}")
                    st["log"].append((text, out["released"]))
                    print(text)
                w = line.watchdog()
                if w["dropped"]:
                    st["log"].append((f"release dropped: {w['dropped']}", False))
                    print(st["log"][-1][0])
                st.update(connected=True, released=w["released"], heartbeat_ok=w["heartbeat_ok"], fault=w["fault"])
                time.sleep(line.poll_s)

    worker = threading.Thread(target=scan, daemon=True)
    worker.start()
    cv2.namedWindow(window, (cv2.WINDOW_NORMAL if view else cv2.WINDOW_AUTOSIZE) | cv2.WINDOW_GUI_NORMAL)
    if view:
        cv2.resizeWindow(window, 1200, 736)
    try:
        while True:
            cv2.imshow(window, view.render(st, codes) if view else draw_panel(st, codes, specs))
            key = cv2.waitKey(50) & 0xFF
            try:
                closed = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                closed = True
            if key == ord("q") or closed:
                break
            if key == ord(" ") and not st["busy"]:
                sends.put(st["kit"])
            elif key == ord("a"):
                st["auto"] = not st["auto"]
            elif chr(key).isdigit() and int(chr(key)) in codes:
                st["kit"] = int(chr(key))
    except KeyboardInterrupt:
        pass
    finally:
        st["stop"] = True
        worker.join(timeout=3)
        cv2.destroyAllWindows()
        if proc:
            proc.terminate()


if __name__ == "__main__":
    main()
