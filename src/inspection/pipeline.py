"""One inspection: photo + kit ID from the PLC -> PASS / FAIL / REVIEW with reasons.

    photo ──> model (bolt / nut / screw boxes)
          ──> scene check (tray found? in position? brightness normal?)
          ──> unknown-object check (anything on the floor the model didn't explain)
          ──> decision rules (src/inspection/decision.py)

Settings come from configs/station.yaml, section `inspection`.

Usage:
    from src.inspection.pipeline import Inspector
    inspector = Inspector()                       # loads the model and settings once
    out = inspector.inspect(frame, "KIT-A")       # frame: a BGR image (cv2.imread / camera)
    out["decision"].result, out["decision"].reasons
"""

import time
from pathlib import Path

import yaml

from src.inspection.decision import Detection, decide
from src.inspection.leftovers import find_leftovers
from src.inspection.scene import check_scene
from src.inspection.tray import in_bowl

REPO = Path(__file__).resolve().parents[2]


def load_settings():
    station = yaml.safe_load((REPO / "configs" / "station.yaml").read_text())
    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]
    return station["inspection"], kits


class Inspector:
    def __init__(self, settings=None, kits=None, model=None):
        loaded, loaded_kits = load_settings()
        self.cfg = settings or loaded
        self.kits = kits or loaded_kits
        if model is None:
            from ultralytics import YOLO
            model = YOLO(str(REPO / self.cfg["model"]))
        self.model = model

    def inspect(self, frame, kit_id):
        t0 = time.perf_counter()
        d_cfg = self.cfg["decision"]
        r = self.model.predict(frame, conf=d_cfg["low"], imgsz=self.cfg["imgsz"], verbose=False)[0]
        t_model = time.perf_counter()
        boxes = r.boxes.xyxy.tolist()
        scene = check_scene(frame, boxes, **self.cfg["scene"])
        detections = [Detection(self.model.names[int(c)], float(s), b,
                                in_bowl(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2), scene["rim"]))
                      for b, c, s in zip(boxes, r.boxes.cls.tolist(), r.boxes.conf.tolist())]
        unknown = []
        if scene["floor"] is not None:
            unknown = find_leftovers(frame, scene["floor"], boxes, **self.cfg["leftovers"])
        decision = decide(kit_id, self.kits, detections, scene["problems"], unknown, **d_cfg)
        t_end = time.perf_counter()
        return {
            "decision": decision,
            "detections": detections,
            "scene": scene,
            "unknown_objects": unknown,
            "ms": {"model": round((t_model - t0) * 1000, 1), "total": round((t_end - t0) * 1000, 1)},
        }
