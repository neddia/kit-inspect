"""`kit`: one command for the whole project. Each subcommand runs an existing module; nothing new underneath.

Run the line (two apps, like a real cell):
    kit demo                      PLC operator panel + vision station together
    kit plc                       just the simulated PLC with its operator panel
    kit station [--show]          just the vision station

Improve the model (the data loop):
    kit camera                    set up the rig: live view, camera settings
    kit capture [--verdict]       guided photo sessions from configs/capture_plan.yaml (records the truth)
    kit harvest                   pull real station photos (by history #id) into a session
    kit label [--session S]       SAM 3 auto-labels anything new, then opens the review tool
    kit train NAME [--classes 4]  export labels -> train -> OpenVINO: a named CANDIDATE in models/
    kit eval [--model M] ...      score PASS / FAIL / REVIEW against the manifest

Checks:
    kit label-check [--kit KIT-B] what the OCR reads on the tray, and its verdict

`kit train` never changes what the station runs: evaluate the candidate with
`kit eval --model ...`, then select it by editing inspection.model in configs/station.yaml.

Run with uv:  uv run kit demo
"""

import importlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

COMMANDS = {
    "demo": "src.demo",
    "plc": "src.plc.panel",
    "station": "src.station.run",
    "camera": "src.capture.preview",
    "capture": "src.capture.guided",
    "harvest": "src.station.harvest",
    "eval": "src.eval.decision_eval",
    "label-check": "src.inspection.kit_label",
}


def run(module, name, args):
    sys.argv = [f"kit {name}"] + args
    importlib.import_module(module).main()


def label(args):
    """SAM 3 outlines every image that has no auto-label yet, then the review tool opens."""
    run("src.labeling.auto_label", "label (auto)", [])
    run("src.labeling.review", "label", args)


def train(args):
    """export_yolo -> train -> copy best.pt to models/ -> OpenVINO export. Same recipe as v1/v2."""
    if not args or args[0].startswith("-"):
        raise SystemExit("usage: kit train NAME [--classes 3|4] [--epochs 100]   e.g. kit train v3")
    name, rest = args[0], args[1:]
    classes = rest[rest.index("--classes") + 1] if "--classes" in rest else "4"
    epochs = rest[rest.index("--epochs") + 1] if "--epochs" in rest else "100"
    out = REPO / "models" / f"kit_yolo26n_640_{name}.pt"
    if out.exists():
        raise SystemExit(f"{out.name} already exists: pick a new name")

    run("src.labeling.export_yolo", "train (export labels)", ["--classes", classes])
    data = "data/yolo_other/data.yaml" if classes == "4" else "data/yolo/data.yaml"
    run("src.train.train", "train", ["model=models/yolo26n.pt", f"data={data}", f"epochs={epochs}", "imgsz=640",
                                     "batch=16", "seed=0", "project=runs/stage5", f"name={name}", "exist_ok=True"])
    best = max(REPO.glob(f"runs/**/stage5/{name}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    shutil.copy2(best, out)
    from ultralytics import YOLO
    YOLO(str(out)).export(format="openvino", imgsz=640, half=False)
    ov = out.with_name(out.stem + "_openvino_model")
    print(f"\nCandidate ready: {ov.relative_to(REPO)}  (the station still runs its current model)")
    print(f"  compare:  uv run kit eval --model {ov.relative_to(REPO)} --session hardval2 --name {name}_hardval2")
    print("  select:   set inspection.model in configs/station.yaml, with a comment saying why")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(__doc__)
        return
    name, args = sys.argv[1], sys.argv[2:]
    if name == "label":
        label(args)
    elif name == "train":
        train(args)
    elif name in COMMANDS:
        run(COMMANDS[name], name, args)
    else:
        raise SystemExit(f"unknown command '{name}'. Commands: {', '.join(['label', 'train'] + list(COMMANDS))}")


if __name__ == "__main__":
    main()
