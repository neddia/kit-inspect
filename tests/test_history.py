"""History + pulling station photos into a training session, on a scratch folder. No camera, no model, no PLC.

Run: uv run pytest tests/test_history.py
"""

import csv
import sys

import numpy as np

import src.station.harvest as harvest
from src.inspection.decision import Decision
from src.station.history import History


def result(cycle, shade, kit="KIT-A"):
    frame = np.full((90, 160, 3), shade, np.uint8)       # a different shade = a "moved" tray
    out = {"detections": [], "unknown_objects": [], "scene": {"problems": []}, "ms": {"total": 50.0}}
    return {"cycle_id": cycle, "kit_id": kit, "decision": Decision("PASS", ["ok"]), "out": out,
            "frame": frame, "ms_frame": 20.0, "ms": 70.0}


def test_harvest_takes_an_id_range_once_with_the_plcs_kit(tmp_path, monkeypatch):
    h = History(tmp_path / "history")
    ids = [h.record(result(1, 50)),
           h.record(result(2, 100, kit="KIT-B")),
           h.record(result(3, 100, kit="KIT-B")),             # same tray as cycle 2: a near-duplicate
           h.record(result(4, 200))]
    h.record({"cycle_id": 5, "kit_id": "KIT-A", "error": "camera unplugged"})   # faults have no photo: never taken
    images = [r["image"] for r in h.recent()]
    assert len(set(i for i in images if i)) == 4               # every photo its own file, even in the same second

    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    with (data / "manifest.csv").open("w", newline="") as f:
        f.write("image,session,split,tier,kit_id,bolts,nuts,screws,other,expected,scenario\n")
    monkeypatch.setattr(harvest, "DATA", data)
    monkeypatch.setattr(harvest, "ROOT", h.root)
    monkeypatch.setattr(harvest, "History", lambda: History(h.root))
    monkeypatch.setattr(sys, "argv", ["harvest", "--from-id", str(ids[1]), "--to-id", str(ids[3]), "--session", "live02"])
    harvest.main()

    rows = list(csv.DictReader((data / "manifest.csv").open()))
    assert [r["image"] for r in rows] == ["live02/live02_0001.jpg", "live02/live02_0002.jpg"]   # cycles 2 and 4
    assert rows[0]["kit_id"] == "KIT-B" and rows[0]["bolts"] == "2" and rows[0]["nuts"] == "4"  # the PLC's kit
    assert rows[0]["other"] == "-1" and rows[0]["expected"] == ""     # truth unknown until reviewed
    monkeypatch.setattr(sys, "argv", ["harvest", "--from-id", str(ids[1]), "--session", "live03"])
    try:
        harvest.main()
        raise AssertionError("expected 'nothing to take'")
    except SystemExit as e:
        assert "nothing to take" in str(e)                     # never taken twice
