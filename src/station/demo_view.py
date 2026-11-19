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
        width = min(frame.shape[1], 820 if path.name == "inspection" else 640)
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
    def __init__(self, directory):
        self.live = Snapshot(directory / "preview")
        self.last = Snapshot(directory / "inspection")
        self.had_heartbeat = False

    def render(self, state, codes):
        self.live.refresh()
        self.last.refresh()
        img = np.full((690, 1200, 3), (40, 32, 23), np.uint8)
        white, muted = (247, 246, 239), (186, 184, 163)
        green, amber, red = (160, 224, 117), (104, 203, 255), (133, 146, 255)

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
        label(24, 46, "Fastener kit inspection", 1.05, weight=2)
        label(898, 43, "SIMULATED PLC", .72, muted)
        cv2.line(img, (24,65), (1176,65), (91,82,65))
        title = f"LAST INSPECTION   /   CYCLE {last['cycle']}   /   {last['kit']}" if last else "CAMERA PREVIEW"
        label(24, 101, title, .65, muted)
        photo(self.last.image if last else live, 24, 119, 820, 461)
        if not last and live is None:
            label(226, 354, "Waiting for a camera frame", .8, muted)

        x = 870
        label(x, 101, "MAY THE TRAY MOVE ON?", .64, muted)
        permit = state["released"]
        cv2.rectangle(img,(x,118),(1176,201), green if permit else amber,-1)
        label(x+17,177,"RELEASE" if permit else "HOLD",1.55,(38,36,26),3)
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
        label(x,234,health,.68,green if health=="Heartbeat healthy" else red,2)
        label(x,277,"SELECTED ORDER",.55,muted)
        label(x,312,codes[state["kit"]],1.0,weight=2)
        label(x,359,"PHOTO REPLAY" if current.get("replay") else "LIVE CAMERA",.55,muted)
        photo(live,x,375,306,172)
        if live is None:
            label(x+23,468,"No current frame",.66,muted)
        if state["busy"]:
            label(x,578,"Inspecting...",.62,amber)

        if last:
            label(24,610,last['result'],.82,green if last['result']=="PASS" else amber,2)
            for i,line in enumerate(textwrap.wrap(" / ".join(last['reasons']),width=83)[:2]):
                label(132,607+i*23,line,.59)
        else:
            label(24,611,"Press SPACE to inspect the selected kit.",.68,muted)
        cv2.line(img,(24,644),(1176,644),(91,82,65))
        label(24,674,"SPACE  inspect     1  KIT-A     2  KIT-B     A  auto: " + ("ON" if state["auto"] else "OFF") + "     Q  quit",.62)
        return img
