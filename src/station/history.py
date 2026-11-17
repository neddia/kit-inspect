"""Every inspection, kept: one SQLite row + the raw photo.

Why keep it all: to answer "why was kit 4127 held?" later, to spot drift (falling
confidence, rising REVIEW rate), and because the station's own photos are the
best future training data (they're exactly the line's conditions).

    data/history/history.db                     one row per inspection (or fault)
    data/history/images/2026-11-17/143012_c42.jpg   the raw photo (annotations can be redrawn)

SQLite is a database in one file, built into Python. WAL mode lets the dashboard
read while the station writes.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "history"

COLUMNS = """
    id INTEGER PRIMARY KEY,
    time TEXT,            -- local time, ISO format
    cycle_id INTEGER,     -- the PLC's Insp_CycleID
    kit_id TEXT,          -- kit name (from the PLC's code)
    result TEXT,          -- PASS / FAIL / REVIEW, or FAULT
    reasons TEXT,         -- JSON list, e.g. ["missing 1 nut"]
    counts TEXT,          -- JSON {sure, unsure, spec}
    detections TEXT,      -- JSON list of {cls, conf, box, in_bowl}
    unknown_objects TEXT, -- JSON list of {box, area}
    scene TEXT,           -- JSON {brightness, rim, floor, problems}
    image TEXT,           -- photo path, relative to data/history
    ms_frame REAL,        -- trigger seen -> fresh photo
    ms_inspect REAL,      -- model + checks + rules
    ms_total REAL,        -- trigger seen -> Done written
    model TEXT,           -- model file and version
    error TEXT            -- what went wrong, for faults
"""


class History:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "history.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"CREATE TABLE IF NOT EXISTS inspections ({COLUMNS})")
        self.db.commit()

    def record(self, r, model=""):
        """Store one result from the station (see VisionSide.handle_trigger). Returns the row id."""
        now = datetime.now()
        image = None
        if r.get("frame") is not None:
            image = Path("images") / now.strftime("%Y-%m-%d") / f"{now:%H%M%S}_c{r['cycle_id']}.jpg"
            (self.root / image).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.root / image), r["frame"])
        row = {"time": now.isoformat(timespec="milliseconds"), "cycle_id": r["cycle_id"], "kit_id": r["kit_id"],
               "image": str(image) if image else None, "model": model, "ms_frame": r.get("ms_frame"),
               "ms_total": r.get("ms")}
        if "error" in r:
            row |= {"result": "FAULT", "error": r["error"]}
        else:
            d, out = r["decision"], r["out"]
            row |= {
                "result": d.result,
                "reasons": json.dumps(d.reasons),
                "counts": json.dumps({"sure": d.sure, "unsure": d.unsure, "spec": d.spec}),
                "detections": json.dumps([{"cls": x.cls, "conf": round(x.conf, 3), "box": [round(v) for v in x.box],
                                           "in_bowl": x.in_bowl} for x in out["detections"]]),
                "unknown_objects": json.dumps(out["unknown_objects"], default=int),
                "scene": json.dumps(out["scene"], default=float),
                "ms_inspect": out["ms"]["total"],
            }
        cur = self.db.execute(f"INSERT INTO inspections ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                              list(row.values()))
        self.db.commit()
        return cur.lastrowid

    def recent(self, n=20):
        return [dict(x) for x in self.db.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (n,))]

    def close(self):
        self.db.close()
