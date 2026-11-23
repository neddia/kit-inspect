"""PatchCore unknown-object check. An experiment, NOT used by the station.

Decided Nov 17: the station uses the OpenCV check (leftovers.py). PatchCore caught
more on val (25/26 vs 23/26) but takes ~2 s per tray on 2 CPU cores (OpenCV: 9 ms)
and +515 MB RAM. See experiments.md.

The OpenCV check (leftovers.py) looks for "darker than its surroundings or
strongly coloured". PatchCore instead learns what a NORMAL tray looks like from
photos alone (no labels, no foreign objects needed), then flags anything that
doesn't look like any part of a normal tray.

How it works (anomalib's PatchCore):
  1. A CNN pretrained on ImageNet (wide_resnet50_2) describes each photo as a
     64 x 64 grid of patches, 1536 numbers per patch (layers 2 and 3).
     Nothing is trained: the CNN's weights stay as they are.
  2. "Fitting" stores the patch descriptions of the normal trays (204 photos x
     4096 patches) in a memory bank, then keeps a 2% sample that still covers
     all of them (coreset sampling: the slow step, ~30 min on the GPU).
  3. Scoring a new photo: for each patch, the distance to the closest patch in
     the memory bank. Those distances are the heatmap; the tray's score is the
     largest one, rescaled to 0-1.

Input is a 512 x 512 square cropped around the bowl (find_tray's rim circle),
so the desk around the tray doesn't matter.

Commands (repo root):
    uv run python -m src.inspection.patchcore prep      # crops -> data/anomaly/
    uv run python -m src.inspection.patchcore fit       # ~30 min -> models/patchcore_tray_v1.ckpt
    uv run python -m src.inspection.patchcore score     # score the test crops, show the separation
    uv run python -m src.inspection.patchcore score data/images/val01/val01_0034.jpg ...
    uv run python -m src.inspection.patchcore heatmap   # picture of 6 interesting trays
    uv run python -m src.inspection.patchcore heatmap data/images/val01/val01_0013.jpg --out x.jpg
"""

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np

from src.inspection.tray import find_tray

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "anomaly"
CKPT = REPO / "models" / "patchcore_tray_v1.ckpt"
SIZE = 512
THRESHOLD = 0.45   # chosen by looking at the test crops' scores (optimistic): good max 0.42, foreign mostly 0.48+
HEATMAP_EXAMPLES = ["test/foreign/val01_0034.jpg", "test/foreign/val01_0036.jpg", "test/foreign/train03_0034.jpg",
                    "test/foreign/train01_0034.jpg", "test/foreign/train04_0035.jpg", "test/good/val01_0006.jpg"]


def crop_bowl(frame, rim, size=SIZE):
    """Square crop around the rim circle (padded black if it runs off the photo), resized to size x size."""
    x, y, r = rim
    half = int(r * 1.02)
    padded = cv2.copyMakeBorder(frame, half, half, half, half, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    crop = padded[int(y):int(y) + 2 * half, int(x):int(x) + 2 * half]
    return cv2.resize(crop, (size, size))


def as_crop(frame):
    """A tray photo -> its bowl crop. Already-cropped 512 x 512 images pass through."""
    if frame.shape[:2] == (SIZE, SIZE):
        return frame
    rim = find_tray(frame)
    if rim is None:
        raise ValueError("no tray found in the photo")
    return crop_bowl(frame, rim)


def load(ckpt=CKPT):
    """The fitted model, ready to score (GPU if there is one)."""
    import torch
    from anomalib.models import Patchcore

    model = Patchcore.load_from_checkpoint(str(ckpt)).eval()
    return model.cuda() if torch.cuda.is_available() else model


def score(model, crop):
    """(score 0-1, heatmap SIZE x SIZE) for one BGR bowl crop. Higher = less like a normal tray."""
    import torch

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().div(255).to(model.device)
    with torch.no_grad():
        out = model(x)
    return float(out.pred_score[0]), out.anomaly_map[0].squeeze().cpu().numpy()


def overlay(crop, heat, label=""):
    """Photo and photo + heatmap side by side (blue = normal, red = unlike anything normal)."""
    colour = cv2.applyColorMap(np.clip(heat / 0.8 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    tile = np.hstack([crop, cv2.addWeighted(crop, 0.55, colour, 0.45, 0)])
    tile = cv2.resize(tile, (640, 320))
    cv2.rectangle(tile, (0, 0), (640, 24), (0, 0, 0), -1)
    cv2.putText(tile, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def prep(out=DATA):
    """Crop train/val photos into anomalib's folder layout.

    train/good    normal trays from pilot + train sessions (the memory bank)
    test/good     normal trays from val01
    test/foreign  every foreign_object tray (never in the memory bank, so safe to test on)
    Stacked trays are left out: hidden parts aren't this check's job.
    """
    from src.capture.camera import quiet_jpeg_warnings

    quiet_jpeg_warnings()
    shutil.rmtree(out, ignore_errors=True)
    with (REPO / "data" / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] in ("train", "val")]
    counts = {}
    for r in rows:
        if r["scenario"] == "stacked":
            continue
        sub = "test/foreign" if r["scenario"] == "foreign_object" else "train/good" if r["split"] == "train" else "test/good"
        frame = cv2.imread(str(REPO / "data" / "images" / r["image"]))
        (out / sub).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / sub / Path(r["image"]).name), as_crop(frame))
        counts[sub] = counts.get(sub, 0) + 1
    print(counts)


def fit(data=DATA, ckpt=CKPT):
    from anomalib.data import Folder
    from anomalib.engine import Engine
    from anomalib.models import Patchcore

    dm = Folder(name="tray", root=str(data), normal_dir="train/good", abnormal_dir="test/foreign",
                normal_test_dir="test/good", train_batch_size=8, eval_batch_size=8, num_workers=4)
    model = Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), coreset_sampling_ratio=0.02,
                      pre_processor=Patchcore.configure_pre_processor(image_size=(SIZE, SIZE)))
    engine = Engine(default_root_dir=str(REPO / "runs" / "anomaly"))
    engine.fit(model=model, datamodule=dm)
    print(engine.test(model=model, datamodule=dm))
    engine.trainer.save_checkpoint(str(ckpt))
    print(f"Saved {ckpt}")


def score_test_set(model):
    """Score every test crop and show how well good and foreign trays separate."""
    scores = {}
    for kind in ("good", "foreign"):
        for p in sorted((DATA / "test" / kind).glob("*.jpg")):
            scores.setdefault(kind, []).append((score(model, cv2.imread(str(p)))[0], p.stem))
    good, foreign = sorted(scores["good"]), sorted(scores["foreign"])
    print(f"good trays    {len(good)}: scores {good[0][0]:.2f} - {good[-1][0]:.2f}, "
          f"above {THRESHOLD}: {sum(s > THRESHOLD for s, _ in good)}  (false alarms)")
    print(f"foreign trays {len(foreign)}: scores {foreign[0][0]:.2f} - {foreign[-1][0]:.2f}, "
          f"above {THRESHOLD}: {sum(s > THRESHOLD for s, _ in foreign)}  (caught)")
    print("\nClosest calls (highest good, lowest foreign):")
    for s, name in good[-3:][::-1]:
        print(f"  good     {name:<16} {s:.3f}")
    for s, name in foreign[:3]:
        print(f"  foreign  {name:<16} {s:.3f}")


def main():
    parser = argparse.ArgumentParser(description="PatchCore unknown-object check (experiment)")
    parser.add_argument("command", choices=["prep", "fit", "score", "heatmap"])
    parser.add_argument("images", nargs="*", help="tray photos or bowl crops (score/heatmap)")
    parser.add_argument("--out", default=str(REPO / "reports" / "eval" / "patchcore" / "heatmaps.jpg"))
    args = parser.parse_args()

    if args.command == "prep":
        return prep()
    if args.command == "fit":
        return fit()

    from src.capture.camera import quiet_jpeg_warnings

    quiet_jpeg_warnings()
    model = load()
    if args.command == "score" and not args.images:
        return score_test_set(model)
    paths = [Path(p) for p in args.images] or [DATA / n for n in HEATMAP_EXAMPLES]
    tiles = []
    for p in paths:
        crop = as_crop(cv2.imread(str(p)))
        s, heat = score(model, crop)
        flag = "UNKNOWN OBJECT" if s > THRESHOLD else "normal"
        print(f"{p.name:<20} score {s:.3f}  {flag}")
        tiles.append(overlay(crop, heat, f"{p.stem}  score {s:.2f}  {flag}"))
    if args.command == "heatmap":
        if len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, sheet)
        print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
