"""Draw the visual prompt: one box around one example of each part.

YOLOE's visual prompt means "find things that look like THIS". You show it one
photo with a box around one bolt, one nut and one screw; it then looks for
similar objects in every other image.

Run from the repo root:
    uv run python -m src.eval.draw_reference                       # uses pilot_0001.jpg
    uv run python -m src.eval.draw_reference --image pilot/pilot_0005.jpg

For each part it asks for in turn: drag a box tightly around ONE example,
then press ENTER (or SPACE). Press c to cancel and redo that box.
Pick clear examples: not touching anything, fully in view.

Saves configs/visual_prompt.yaml and a picture of your boxes in reports/stage2/.
"""

import argparse
from pathlib import Path

import cv2
import yaml

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Draw visual-prompt boxes")
    parser.add_argument("--image", default="pilot/pilot_0001.jpg", help="path under data/images/")
    args = parser.parse_args()

    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]
    parts = []
    for spec in kits.values():
        parts += [p for p in spec if p not in parts]

    img = cv2.imread(str(REPO / "data" / "images" / args.image))
    if img is None:
        raise SystemExit(f"Can't read data/images/{args.image}")

    boxes = {}
    shown = img.copy()
    for part in parts:
        window = f"Box ONE {part.upper()}, then ENTER"
        x, y, w, h = cv2.selectROI(window, shown, showCrosshair=False)
        cv2.destroyWindow(window)
        if w == 0 or h == 0:
            raise SystemExit(f"No box drawn for {part}; nothing saved.")
        boxes[part] = [int(x), int(y), int(x + w), int(y + h)]  # x1, y1, x2, y2
        cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 200, 0), 2)
        cv2.putText(shown, part, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2, cv2.LINE_AA)

    out = REPO / "configs" / "visual_prompt.yaml"
    out.write_text(yaml.safe_dump({"image": args.image, "boxes": boxes}, sort_keys=False))
    pic = REPO / "reports" / "stage2" / "visual_prompt.jpg"
    pic.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(pic), shown)
    print(f"Saved {out.relative_to(REPO)} and {pic.relative_to(REPO)}: {boxes}")


if __name__ == "__main__":
    main()
