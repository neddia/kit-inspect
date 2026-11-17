"""Every inspection, kept: one SQLite row + the raw photo.

Why keep it all: to answer "why was kit 4127 held?" later, to spot drift (falling
confidence, rising REVIEW rate), and because the station's own photos are the
best future training data (they're exactly the line's conditions).

    data/history/history.db                     one row per inspection (or fault)
    data/history/images/2026-11-17/143012_250_c42.jpg   the raw photo (annotations can be redrawn)

SQLite is a database in one file, built into Python. WAL mode lets the dashboard
read while the station writes.
"""

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path

import cv2

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data" / "history"

COLUMNS = """
    id INTEGER PRIMARY KEY,
    time TEXT,            -- local time, ISO format
    cycle_id INTEGER,     -- the PLC's Insp_CycleID (empty for a bench inspection: SPACE in the window)
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
    error TEXT,           -- what went wrong, for faults
    train INTEGER DEFAULT 0,  -- 1 = marked "save for training" (station window: t toggles, u undoes)
    harvested TEXT        -- training session it was copied into (src/station/harvest.py)
"""


class History:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        (self.root / "images").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / "history.db", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(f"CREATE TABLE IF NOT EXISTS inspections ({COLUMNS})")
        have = {r[1] for r in self.db.execute("PRAGMA table_info(inspections)")}
        for col, kind in (("train", "INTEGER DEFAULT 0"), ("harvested", "TEXT")):   # older history files
            if col not in have:
                self.db.execute(f"ALTER TABLE inspections ADD COLUMN {col} {kind}")
        self.db.commit()
        self.lock = threading.Lock()              # the station thread records, the window thread undoes

    def record(self, r, model="", train=False):
        """Store one result from the station (see VisionSide.handle_trigger). Returns the row id.
        train=True marks it for the next training session (faults are never marked)."""
        now = datetime.now()
        image = None
        if r.get("frame") is not None:
            image = Path("images") / now.strftime("%Y-%m-%d") / (f"{now:%H%M%S}_{now.microsecond // 1000:03d}_"   # ms: two quick inspections never share a name
                + (f"c{r['cycle_id']}" if r["cycle_id"] is not None else "bench") + ".jpg")
            (self.root / image).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(self.root / image), r["frame"])
        row = {"time": now.isoformat(timespec="milliseconds"), "cycle_id": r["cycle_id"], "kit_id": r["kit_id"],
               "image": str(image) if image else None, "model": model, "ms_frame": r.get("ms_frame"),
               "ms_total": r.get("ms"), "train": int(train and "error" not in r and image is not None)}
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
        with self.lock:
            cur = self.db.execute(f"INSERT INTO inspections ({', '.join(row)}) VALUES ({', '.join('?' * len(row))})",
                                  list(row.values()))
            self.db.commit()
        return cur.lastrowid

    def unmark_last(self):
        """Undo: the newest photo marked for training (not yet harvested) is unmarked. Returns its row or None."""
        with self.lock:
            row = self.db.execute("SELECT id, cycle_id FROM inspections WHERE train = 1 AND harvested IS NULL "
                                  "ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                return None
            self.db.execute("UPDATE inspections SET train = 0 WHERE id = ?", (row["id"],))
            self.db.commit()
            return dict(row)

    def marked(self):
        """How many photos are marked for training and not harvested yet."""
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM inspections WHERE train = 1 AND harvested IS NULL").fetchone()[0]

    def recent(self, n=20):
        return [dict(x) for x in self.db.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT ?", (n,))]

    def close(self):
        self.db.close()
