"""History + "save for training" + harvest, on a scratch folder. No camera, no model, no PLC.

Run: uv run pytest tests/test_history.py
"""

import csv
import sys

import numpy as np

import src.station.harvest as harvest
from src.inspection.decision import Decision
from src.station.history import History


def result(cycle, shade):
    frame = np.full((90, 160, 3), shade, np.uint8)       # a different shade = a "moved" tray
    out = {"detections": [], "unknown_objects": [], "scene": {"problems": []}, "ms": {"total": 50.0}}
    return {"cycle_id": cycle, "kit_id": "KIT-A", "decision": Decision("PASS", ["ok"]), "out": out,
            "frame": frame, "ms_frame": 20.0, "ms": 70.0}


def test_mark_undo_and_harvest_once(tmp_path, monkeypatch):
    h = History(tmp_path / "history")
    h.record(result(1, 50), train=False)          # not saved for training
    h.record(result(2, 100), train=True)
    h.record(result(3, 150), train=True)
    h.record(result(4, 150), train=True)          # same tray as cycle 3: a near-duplicate
    h.record(result(5, 200), train=True)          # a slip...
    assert h.marked() == 4
    assert h.unmark_last()["cycle_id"] == 5      # ...undone
    assert h.marked() == 3
    h.record({"cycle_id": 6, "kit_id": "KIT-A", "error": "camera unplugged"}, train=True)
    assert h.marked() == 3                        # faults are never saved for training
    images = [r["image"] for r in h.recent()]
    assert len(set(i for i in images if i)) == 5  # every photo its own file, even in the same second

    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    with (data / "manifest.csv").open("w", newline="") as f:
        f.write("image,session,split,tier,kit_id,bolts,nuts,screws,other,expected,scenario\n")
    monkeypatch.setattr(harvest, "DATA", data)
    monkeypatch.setattr(harvest, "ROOT", h.root)
    monkeypatch.setattr(harvest, "History", lambda: History(h.root))
    monkeypatch.setattr(sys, "argv", ["harvest", "--session", "live01"])
    harvest.main()

    rows = list(csv.DictReader((data / "manifest.csv").open()))
    assert [r["image"] for r in rows] == ["live01/live01_0001.jpg", "live01/live01_0002.jpg"]   # cycles 2, 3
    assert rows[0]["bolts"] == "3" and rows[0]["split"] == "train" and rows[0]["scenario"] == "live"
    assert h.marked() == 0                        # all taken (the duplicate too), so never taken twice
    monkeypatch.setattr(sys, "argv", ["harvest", "--session", "live02"])
    try:
        harvest.main()
        raise AssertionError("expected 'nothing to take'")
    except SystemExit as e:
        assert "nothing to take" in str(e)
