"""Train YOLO, same as the `yolo detect train` command, minus the JPEG spam.

The C510's JPEGs carry harmless padding that makes libjpeg print
"Corrupt JPEG data" for every image it loads, hundreds per epoch. That's
printed by C code below Python, so it can't be filtered with a setting;
quiet_jpeg_warnings() points it at /dev/null (the data-loading worker
processes inherit that) and keeps Python's own output and errors.

Takes the same key=value arguments as the yolo command:
    uv run python -m src.train.train model=models/yolo26n.pt data=data/yolo/data.yaml \\
        epochs=100 imgsz=640 batch=16 seed=0 project=$PWD/runs/train name=baseline
"""

import sys

from src.capture.camera import quiet_jpeg_warnings


def parse(value):
    """'100' -> 100, '0.5' -> 0.5, 'True' -> True, anything else stays text."""
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(value.lower(), value)


def main():
    args = dict(a.split("=", 1) for a in sys.argv[1:])
    if "model" not in args or "data" not in args:
        raise SystemExit(__doc__)
    quiet_jpeg_warnings()
    from ultralytics import YOLO

    model = YOLO(args.pop("model"))
    model.train(**{k: parse(v) for k, v in args.items()})


if __name__ == "__main__":
    main()
