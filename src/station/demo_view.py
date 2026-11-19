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
        label(24, 46, "Fastener kit inspection", 1.05, weight=2)
        label(898, 43, "SIMULATED PLC", .72, muted)
        cv2.line(img, (24,65), (1176,65), (91,82,65))
        title = f"LAST INSPECTION   /   CYCLE {last['cycle']}   /   {last['kit']}" if last else "CAMERA PREVIEW"
        label(24, 101, title, .65, muted)
        result = last.get("result")
        colors = {"PASS": green, "FAIL": red, "REVIEW": amber}
        if last:
            label(24,146,result,1.12,colors[result],3)
            reasons = " / ".join(last["reasons"])
            reasons = reasons[:1].upper() + reasons[1:]
            if len(reasons) <= 42:
                label(176,146,reasons,.86,white,2)
            else:
                for i,line in enumerate(textwrap.wrap(reasons, width=65)[:3]):
                    label(176,125+i*23,line,.63,white,2 if i==0 else 1)
        else:
            label(24,148,"Press SPACE to inspect the selected kit.",.78,white)
        photo(self.last.image if last else live, 24, 185, 820, 461)
        if not last and live is None:
            label(226, 354, "Waiting for a camera frame", .8, muted)

        x = 870
        label(x, 101, "MAY THE TRAY MOVE ON?", .64, muted)
        permit = state["released"]
        fault = state["fault"] or not state["heartbeat_ok"] or not state.get("connected")
        hold_color = red if fault or result == "FAIL" else amber if result == "REVIEW" else muted
        cv2.rectangle(img,(x,118),(1176,201), green if permit else hold_color,-1)
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
        label(x,271,"Next order: " + codes[state["kit"]],.70,white,2)

        expected = last.get("expected", {})
        label(x,319,"INSPECTED TRAY" + ("   /   " + last["kit"] if last else ""),.61,muted)
        if expected:
            for dx, heading in [(0,"Part"),(110,"Found"),(178,"Unsure"),(265,"Need")]:
                label(x+dx,350,heading,.51,muted)
            for i,(part,need) in enumerate(expected.items()):
                y = 380 + 29*i
                found = last["confirmed"].get(part,0)
                possible = last["uncertain"].get(part,0)
                wrong = found > need or found + possible < need
                color = red if wrong else amber if possible else white
                if wrong or possible:
                    cv2.rectangle(img,(x-4,y-20),(1176,y+5),(56,44,48),-1)
                    cv2.rectangle(img,(x-4,y-20),(x-1,y+5),color,-1)
                label(x+4,y,"Foreign" if part=="other" else part.capitalize(),.61,color,2)
                for dx,value in [(127,found),(199,possible),(276,need)]:
                    label(x+dx,y,str(value),.65,color,2)
        elif last:
            label(x,353,"Part count skipped",.68,muted)
            tag = last.get("tag")
            if tag:
                label(x,386,"Tag on tray: " + tag["text"],.70,red if result=="FAIL" else white,2)
        else:
            label(x,353,"Awaiting first inspection",.61,muted)

        label(x,511,"PHOTO REPLAY" if current.get("replay") else "LIVE CAMERA",.55,muted)
        photo(live,x,524,306,150)
        if live is None:
            label(x+23,606,"No current frame",.66,muted)
        if state["busy"]:
            label(24,677,"Inspecting...",.62,amber)
        cv2.line(img,(24,690),(1176,690),(91,82,65))
        label(24,720,"SPACE  inspect     1  KIT-A     2  KIT-B     A  auto: " + ("ON" if state["auto"] else "OFF") + "     Q  quit",.62)
        return img
