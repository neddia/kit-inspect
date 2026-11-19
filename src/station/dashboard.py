"""Read-only inspection history for the local viewer and the public replay."""

import argparse
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

REPO = Path(__file__).resolve().parents[2]
HISTORY = REPO / "data/history"
SITE = REPO / "docs"
FIELDS = ("id", "cycle_id", "kit_id", "result", "reasons", "label", "time", "counts",
          "detections", "unknown_objects", "ms_total", "image", "error")


def records(root, ids=()):
    db = root / "history.db"
    if not db.is_file():
        return []
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        if ids:
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(f"SELECT * FROM inspections WHERE id IN ({placeholders}) ORDER BY id", ids)
        else:
            rows = conn.execute("SELECT * FROM inspections ORDER BY id DESC LIMIT 100")
        return [dict(row) for row in rows]


def photo(root, row):
    path = (root / row["image"]).resolve() if row.get("image") else None
    return path if path and path.is_relative_to(root.resolve()) and path.is_file() else None


def public_record(row, root):
    data = {key: row.get(key) for key in FIELDS}
    for key in ("reasons", "label", "counts", "detections", "unknown_objects"):
        data[key] = json.loads(data[key]) if data[key] else None
    data["image"] = f"frames/{row['id']}.jpg" if photo(root, row) else None
    return data


def band():
    """The station's confidence band, so the viewer colours boxes exactly like the station window."""
    from src.inspection.pipeline import load_settings
    d = load_settings()[0]["decision"]
    return {"low": d["low"], "high": d["high"]}


def export(root, ids, locate_labels=False):
    rows = records(root, ids)
    if len(rows) != len(set(ids)) or not all(photo(root, row) for row in rows):
        raise SystemExit("Each selected history ID must have a row and a photograph.")
    frames = SITE / "frames"
    frames.mkdir(exist_ok=True)
    reader = None
    if locate_labels:
        import cv2
        from src.inspection.kit_label import inspect_label, make_reader
        from src.inspection.pipeline import load_settings
        settings, kits = load_settings()
        cfg = settings["label"]
        reader = make_reader(cfg["threads"])

    exported = []
    for row in reversed(rows):
        frame_path = photo(root, row)
        shutil.copyfile(frame_path, frames / f"{row['id']}.jpg")
        item = public_record(row, root)
        # Early records saved the read text, but not its polygon. Locate it for
        # display only when a fresh read agrees with the recorded text and status.
        label = item["label"]
        if reader is not None and label and label.get("text") and not label.get("box"):
            scene = json.loads(row["scene"])
            _, replay = inspect_label(reader, cv2.imread(str(frame_path)), scene["rim"], row["kit_id"], kits,
                                      cfg["min_conf"], cfg["near"])
            if replay["label"] and replay["label"]["text"] == label["text"] and replay["status"] == label["status"]:
                label.update(box=replay["label"]["box"], located_on_replay=True)
        exported.append(item)
    data = {"source": "Recorded inspections", "initial_id": ids[0], "band": band(), "records": exported}
    (SITE / "inspections.json").write_text(json.dumps(data, indent=2) + "\n")


def application(root):
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/inspections.json")
    def recent():
        return {"source": "Local inspection history", "band": band(), "records": [public_record(r, root) for r in records(root)]}

    @app.get("/frames/{record_id}.jpg")
    def image(record_id: int):
        rows = records(root, (record_id,))
        path = photo(root, rows[0]) if rows else None
        if not path:
            raise HTTPException(404)
        return FileResponse(path)

    app.mount("/", StaticFiles(directory=SITE, html=True))
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", nargs="+", type=int, metavar="HISTORY_ID",
                        help="write selected inspection records and photographs into docs/")
    parser.add_argument("--locate-labels", action="store_true", help="locate missing OCR outlines during export")
    parser.add_argument("--port", type=int, default=8010)
    args = parser.parse_args()
    if args.export:
        export(HISTORY, args.export, args.locate_labels)
    else:
        import uvicorn
        uvicorn.run(application(HISTORY), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
