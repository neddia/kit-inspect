"""Display-only snapshots shared by the vision process and the PLC window."""
import json
import os
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


def operator_reason(reason):
    for part in ("nut", "screw"):
        if reason.startswith(f"check {part} size:"):
            if "(minimum " in reason:
                return f"{part.capitalize()} looks small"
            if "(maximum " in reason:
                return f"{part.capitalize()} looks large"
    return reason


class DemoView:
    def __init__(self, directory, specs):
        self.specs = specs
        self.live = Snapshot(directory / "preview")
        self.last = Snapshot(directory / "inspection")
        self.had_heartbeat = False

    def render(self, state, codes):
        self.live.refresh()
        self.last.refresh()
        last = self.last.fields
        reasons = []
        for reason in last.get("reasons", []):
            lines = [""]
            for word in operator_reason(reason).split():
                trial = (lines[-1] + " " + word).strip()
                if lines[-1] and cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, .64, 1)[0][0] > 540:
                    lines.append(word)
                else:
                    lines[-1] = trial
            lines[0] = lines[0][:1].upper() + lines[0][1:]
            reasons.append(lines)

        counts = []
        for part, need in last.get("expected", {}).items():
            found = last["confirmed"].get(part, 0)
            possible = last["uncertain"].get(part, 0)
            priority = 0 if found > need or found + possible < need else 1 if possible else 2
            name = "Foreign objects" if part == "other" else part.capitalize() + "s"
            size_flags = sum(reason.startswith(f"check {part} size:") for reason in last["reasons"])
            counts.append((priority, name, found, need, possible, size_flags))
        counts.sort(key=lambda row: row[0])
        footer = max(690, 565 + 32 * len(counts), 550 + sum(25 * len(lines) + 10 for lines in reasons))
        img = np.full((footer + 46, 1200, 3), (40, 32, 23), np.uint8)
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
        # Identical 16:9 viewing areas keep the raw and annotated tray at the same scale.
        left, right, width, height = 24, 608, 568, 320
        label(left, 43, "Fastener kit inspection", 1.0, weight=2)
        selected = state["kit"]
        recipe = self.specs[codes[selected]]
        label(462, 23, "SELECTED FOR NEXT INSPECTION", .51, muted)
        parts = ", ".join(f"{n} {part}{'s' if n != 1 else ''}" for part, n in recipe.items())
        label(462, 47, parts, .60, white)
        choices = sorted(codes) if len(codes) <= 2 else [selected]
        for i, code in enumerate(choices):
            x = 792 + i * 196
            active = code == selected
            accent = (236, 209, 146)
            cv2.rectangle(img, (x, 8), (x + 188, 54), accent if active else (89, 77, 63), -1 if active else 1)
            ink = (38, 32, 24) if active else muted
            cv2.circle(img, (x + 18, 31), 7, ink, 2)
            if active:
                cv2.circle(img, (x + 18, 31), 3, ink, -1)
            label(x + 36, 39, f"{code}   {codes[code]}", .76, ink, 2 if active else 1)
        cv2.line(img, (left, 61), (1176, 61), (91, 82, 65))
        result = last.get("result")
        colors = {"PASS": green, "FAIL": red, "REVIEW": amber}
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
        fault = state["fault"] or not state["heartbeat_ok"] or not state.get("connected")
        hold_color = red if fault or result == "FAIL" else amber
        status_color = green if permit else hold_color
        current_cycle = state.get("last_cycle", {})
        if state["busy"]:
            status = "Inspection in progress"
        elif fault or (fresh and current.get("error")):
            status = health
        elif current_cycle and current_cycle["why"] not in ("matched PASS -> release", "REVIEW -> hold", "FAIL -> hold"):
            status = current_cycle["why"].split(" -> ")[0]
        elif not permit and last and result in ("FAIL", "REVIEW") and current_cycle.get("cycle_id") == last.get("cycle"):
            status = "Resolve findings and reinspect"
        else:
            status = health
        card = (57, 47, 36)
        cv2.rectangle(img, (left, 75), (left + width, 151), card, -1)
        cv2.rectangle(img, (left, 75), (left + 5, 151), status_color, -1)
        label(left + 17, 98, "TRAY MOVEMENT   /   SIMULATED PLC", .52, muted)
        label(left + 16, 137, "RELEASE" if permit else "HOLD", 1.07, status_color, 2)
        label(left + 185, 133, status, .60, white)

        cv2.rectangle(img, (right, 75), (right + width, 151), card, -1)
        label(right + 16, 98, "LAST INSPECTION", .52, muted)
        if last:
            label(right + 15, 137, result, 1.07, colors[result], 2)
            label(right + 180, 133, f"Cycle {last['cycle']}    {last['kit']}", .65, white)
        else:
            label(right + 16, 134, "Awaiting first inspection", .74, white)

        label(left, 179, "PHOTO REPLAY" if current.get("replay") else "LIVE CAMERA", .55, muted)
        label(right, 179, "INSPECTED FRAME", .55, muted)
        photo(cv2.rotate(live, cv2.ROTATE_180) if live is not None else None, left, 189, width, height)
        photo(self.last.image, right, 189, width, height)
        if live is None:
            label(left + 38, 354, health if fault or current.get("error") else "Waiting for a frame", .84, white)

        if last:
            label(left, 541, "INSPECTED KIT COUNTS", .55, muted)
            for i, (priority, name, found, need, possible, size_flags) in enumerate(counts):
                y = 572 + 32 * i
                color = (red, amber, muted)[priority]
                if priority < 2:
                    cv2.rectangle(img, (left, y-22), (left+width, y+5), card, -1)
                    cv2.rectangle(img, (left, y-22), (left+3, y+5), color, -1)
                detail = f"{found} of {need} confirmed"
                if possible:
                    detail += f"   +{possible} uncertain"
                if size_flags:
                    label(left + 456, y, f"{size_flags} size flag" + ("s" if size_flags > 1 else ""), .53, amber)
                label(left + 12, y, name, .65, color, 2 if priority < 2 else 1)
                label(left + 175, y, detail, .65, color, 2 if priority < 2 else 1)
            if not counts:
                label(left, 572, "Part count skipped", .67, white)
                tag = last.get("tag")
                if tag:
                    label(left, 604, "Printed tag: " + tag["text"], .65, muted)
            label(right, 541, "INSPECTION RESULT" if result == "PASS" else "HOLD REASONS", .55, muted)
            y = 572
            for lines in reasons:
                cv2.circle(img, (right + 5, y - 7), 4, colors[result], -1)
                for line in lines:
                    label(right + 20, y, line, .64, white)
                    y += 25
                y += 10
        elif not fault and not state["busy"]:
            label(right, 568, "Press SPACE to inspect the selected kit.", .69, white)

        cv2.line(img, (24, footer), (1176, footer), (91, 82, 65))
        for x, shortcut, action in [(24,"SPACE","Inspect"), (235,"1 / 2","Select kit"),
                                    (495,"A",f"Stop auto ({state['every']:g}s)" if state["auto"] else "Start auto"), (1010,"Q","Quit")]:
            key_width = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, .53, 1)[0][0] + 18
            cv2.rectangle(img, (x,footer+10), (x+key_width,footer+38), (84,72,57), -1)
            label(x+9,footer+30,shortcut,.53)
            label(x+key_width+10,footer+31,action,.60,muted)
        return img
