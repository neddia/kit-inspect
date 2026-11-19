"""Display-only snapshots shared by the vision process and the PLC window."""
import json
import os
import textwrap
import time

import cv2
import numpy as np


def publish(path, frame, **fields):
    payload = b""
    if frame is not None:
        width = min(frame.shape[1], 640)
        img = cv2.resize(frame, (width, round(width * frame.shape[0] / frame.shape[1])))
        ok, encoded = cv2.imencode(".jpg", img)
        if not ok:
            return
        payload = encoded.tobytes()
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_bytes(json.dumps(fields).encode() + b"\n" + payload)
        os.replace(tmp, path)
    except OSError:
        # Preview is optional. A full or closed display directory must not fault the PLC handshake.
        return


class Snapshot:
    def __init__(self, path):
        self.path = path
        self.modified = None
        self.fields = {}
        self.image = None

    def refresh(self):
        try:
            modified = self.path.stat().st_mtime_ns
            if modified == self.modified:
                return
            header, jpeg = self.path.read_bytes().split(b"\n", 1)
        except FileNotFoundError:
            return
        self.fields = json.loads(header)
        self.image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
        self.modified = modified


class DemoView:
    def __init__(self, directory, specs):
        self.specs = specs
        self.live = Snapshot(directory / "preview")
        self.last = Snapshot(directory / "inspection")
        self.had_heartbeat = False

    def render(self, state, codes):
        self.live.refresh()
        self.last.refresh()
        img = np.full((736, 1200, 3), (40, 32, 23), np.uint8)
        white, muted = (247, 246, 239), (186, 184, 163)
        green, amber, red = (160, 224, 117), (104, 203, 255), (96, 104, 253)

        def label(x, y, value, size=.62, color=white, weight=1):
            cv2.putText(img, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, weight, cv2.LINE_AA)

        def photo(frame, x, y, w, h):
            cv2.rectangle(img, (x,y), (x+w,y+h), (24,21,18), -1)
            if frame is not None:
                scale = min(w/frame.shape[1], h/frame.shape[0])
                pw, ph = round(frame.shape[1]*scale), round(frame.shape[0]*scale)
                left, top = x+(w-pw)//2, y+(h-ph)//2
                img[top:top+ph, left:left+pw] = cv2.resize(frame, (pw, ph))

        current = self.live.fields
        fresh = time.monotonic() - current.get("time", 0) < 1.5
        live = self.live.image if fresh else None
        last = self.last.fields
        # Identical 16:9 viewing areas keep the raw and annotated tray at the same scale.
        left, right, width, height = 24, 608, 568, 320
        label(left, 43, "Fastener kit inspection", 1.0, weight=2)
        selected = state["kit"]
        recipe = self.specs[codes[selected]]
        label(462, 23, "SELECTED FOR NEXT INSPECTION", .51, muted)
        parts = " / ".join(f"{n} {part}{'s' if n != 1 else ''}" for part, n in recipe.items())
        label(462, 47, parts, .60, white)
        choices = sorted(codes) if len(codes) <= 2 else [selected]
        for i, code in enumerate(choices):
            x = 792 + i * 196
            active = code == selected
            accent = (247, 207, 111) if code == 1 else (247, 183, 210)
            cv2.rectangle(img, (x, 8), (x + 188, 54), accent if active else (89, 77, 63), -1 if active else 1)
            ink = (38, 32, 24) if active else muted
            cv2.circle(img, (x + 18, 31), 7, ink, 2)
            if active:
                cv2.circle(img, (x + 18, 31), 3, ink, -1)
            label(x + 36, 39, f"{code}   {codes[code]}", .76, ink, 2 if active else 1)
        cv2.line(img, (left, 61), (1176, 61), (91, 82, 65))
        label(left, 93, "PHOTO REPLAY" if current.get("replay") else "LIVE CAMERA", .62, muted)
        caption = "LAST INSPECTION"
        if last:
            caption += f"    /    Cycle {last['cycle']}    /    {last['kit']}"
        label(right, 93, caption, .62, muted)
        photo(live, left, 109, width, height)
        photo(self.last.image, right, 109, width, height)
        result = last.get("result")
        colors = {"PASS": green, "FAIL": red, "REVIEW": amber}
        if not last:
            label(right + 118, 278, "Waiting for first inspection", .74, muted)

        permit = state["released"]
        self.had_heartbeat |= state["heartbeat_ok"]
        if not state.get("connected"):
            health = "PLC disconnected"
        elif fresh and current.get("error"):
            health = "Camera lost" if not current.get("replay") else "Image source fault"
        elif state["fault"]:
            health = "Vision fault"
        elif not state["heartbeat_ok"]:
            health = "Heartbeat lost" if self.had_heartbeat else "Starting vision..."
        else:
            health = "Heartbeat healthy"
        if live is None:
            message = health if health in ("Camera lost", "Heartbeat lost", "Vision fault") else "Waiting for a frame"
            label(left + 38, 278, message, .90, red if fresh and current.get("error") else muted, 2)

        label(left, 467, "TRAY MOVEMENT   /   SIMULATED PLC", .61, muted)
        fault = state["fault"] or not state["heartbeat_ok"] or not state.get("connected")
        hold_color = red if fault or result == "FAIL" else amber if result == "REVIEW" else muted
        cv2.rectangle(img, (left, 486), (left + width, 565), green if permit else hold_color, -1)
        label(left + 16, 541, "RELEASE" if permit else "HOLD", 1.53, (38, 36, 26), 3)
        label(left, 608, health, .81, green if health == "Heartbeat healthy" else red, 2)
        if state["busy"]:
            label(left, 644, "Inspection in progress", .68, amber)

        if last:
            label(right, 469, result, 1.04, colors[result], 3)
            reasons = " / ".join(last["reasons"])
            lines = textwrap.wrap(reasons[:1].upper() + reasons[1:], width=54)
            for i, line in enumerate(lines[:3]):
                label(right, 501 + 21*i, line, .66 if len(lines)==1 else .57, white, 2 if i==0 else 1)
            expected = last.get("expected", {})
            if expected:
                columns = [right + offset for offset in (8, 204, 331, 460)]
                for x, heading in zip(columns, ("Part", "Found", "Unsure", "Required")):
                    label(x, 565, heading, .57, muted)
                for i, (part, need) in enumerate(expected.items()):
                    y = 592 + 27*i
                    found = last["confirmed"].get(part, 0)
                    possible = last["uncertain"].get(part, 0)
                    wrong = found > need or found + possible < need
                    color = red if wrong else amber if possible else white
                    if wrong or possible:
                        cv2.rectangle(img, (right, y-20), (1176, y+5), (56, 44, 48), -1)
                        cv2.rectangle(img, (right, y-20), (right+3, y+5), color, -1)
                    for x, value in zip(columns, ("Foreign object" if part=="other" else part.capitalize(), found, possible, need)):
                        label(x, y, str(value), .64, color, 2)
            else:
                label(right, 578, "Part count skipped", .68, muted)
                tag = last.get("tag")
                if tag:
                    label(right, 612, "Printed tag: " + tag["text"], .74, colors[result], 2)
        else:
            label(right, 503, "Press SPACE to inspect the selected kit.", .69, white)

        cv2.line(img, (24, 690), (1176, 690), (91, 82, 65))
        for x, shortcut, action in [(24,"SPACE","Inspect"), (235,"1 / 2","Select kit"),
                                    (495,"A","Auto " + ("on" if state["auto"] else "off")), (1010,"Q","Quit")]:
            key_width = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, .53, 1)[0][0] + 18
            cv2.rectangle(img, (x,700), (x+key_width,728), (84,72,57), -1)
            label(x+9,720,shortcut,.53)
            label(x+key_width+10,721,action,.60,muted)
        return img
