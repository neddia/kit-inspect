"""Zero-shot baseline. How well does YOLOE count parts with NO training?

Runs YOLOE on every image of a session and compares its counts with the
manifest (the true counts you entered while capturing).

Text prompts (just words):
    uv run python -m src.eval.zero_shot --session pilot
    uv run python -m src.eval.zero_shot --session pilot --prompt bolt="hex bolt" --prompt nut="hex nut"

Visual prompt (one reference photo with a box around one of each part;
make it first with: uv run python -m src.eval.draw_reference):
    uv run python -m src.eval.zero_shot --session pilot --visual configs/visual_prompt.yaml

Outputs, in reports/zero_shot/<run name>/:
    summary.txt    the numbers for experiments.md
    per_image.csv  true vs predicted counts for every image, at the best cutoff
    wrong/         annotated images the counts got wrong (for the failure gallery)

How it scores:
    count accuracy   % of images where EVERY part count is exactly right
    decision         PASS if the counts match the kit spec, else FAIL, compared with
                     the manifest's expected result (NOT_PASS accepts FAIL)
    false passes     incomplete/invalid trays the counts would let through: the
                     number that matters most
    false rejects    complete kits it would stop. A model that says FAIL to
                     everything gets 0 false passes, so always read both.

Confidence cutoff: YOLOE only counts boxes above a cutoff, and the right cutoff
isn't known in advance. We report several. The "best" one is picked on these
same images, so it's an OPTIMISTIC number; say so wherever it's quoted.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

from src.capture.camera import quiet_jpeg_warnings

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
CUTOFFS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def load_session(session):
    with (DATA / "manifest.csv").open(newline="") as f:
        return [r for r in csv.DictReader(f) if r["session"] == session]


def count_at(detections, cutoff, parts):
    """detections: list of (part, confidence). Count the ones at or above the cutoff."""
    counts = Counter(part for part, conf in detections if conf >= cutoff)
    return {p: counts.get(p, 0) for p in parts}


def decide(counts, spec):
    """The simplest possible station: PASS only if every count matches the kit."""
    return "PASS" if all(counts[p] == spec.get(p, 0) for p in counts) else "FAIL"


def decision_correct(decision, expected):
    return decision == "PASS" if expected == "PASS" else decision == "FAIL"


def score(rows, all_dets, cutoff, parts, specs):
    exact, correct, false_pass, false_reject, abs_err = 0, 0, 0, 0, Counter()
    for row in rows:
        true = {p: int(row[f"{p}s"]) for p in parts}
        pred = count_at(all_dets[row["image"]], cutoff, parts)
        exact += pred == true
        for p in parts:
            abs_err[p] += abs(pred[p] - true[p])
        d = decide(pred, specs[row["kit_id"]])
        correct += decision_correct(d, row["expected"])
        false_pass += d == "PASS" and row["expected"] != "PASS"
        false_reject += d == "FAIL" and row["expected"] == "PASS"
    n = len(rows)
    return {
        "cutoff": cutoff,
        "count_acc": exact / n,
        "decision_acc": correct / n,
        "false_passes": false_pass,
        "faulty": sum(r["expected"] != "PASS" for r in rows),
        "false_rejects": false_reject,
        "good": sum(r["expected"] == "PASS" for r in rows),
        "mae": {p: abs_err[p] / n for p in parts},
    }


def load_model(args, parts):
    from ultralytics import YOLOE

    model = YOLOE(str(REPO / "models" / args.model))
    if args.visual:
        return model, None
    words = dict(p.split("=", 1) for p in args.prompt)
    prompts = [words.get(p, p) for p in parts]  # e.g. "hex bolt" stands for class "bolt"
    model.set_classes(prompts)
    return model, prompts


def predict_all(model, rows, args, parts):
    """Run once at a very low cutoff and keep every box; cutoffs are applied afterwards."""
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    kw = dict(conf=min(CUTOFFS), device=0, verbose=False, agnostic_nms=True)
    if args.visual:
        ref = yaml.safe_load(Path(args.visual).read_text())
        boxes = [ref["boxes"][p] for p in parts]
        kw |= dict(
            refer_image=str(DATA / "images" / ref["image"]),
            visual_prompts=dict(bboxes=np.array(boxes, dtype=float), cls=np.arange(len(parts))),
            predictor=YOLOEVPSegPredictor,
        )
    all_dets, results = {}, {}
    for row in rows:
        r = model.predict(str(DATA / "images" / row["image"]), **kw)[0]
        # Class i is parts[i] in both modes (we gave the prompts in parts order).
        all_dets[row["image"]] = [(parts[int(c)], float(s)) for c, s in zip(r.boxes.cls, r.boxes.conf)]
        r.names = dict(enumerate(parts))  # visual prompts come back as "object0"; use part names
        results[row["image"]] = r
    return all_dets, results


def main():
    parser = argparse.ArgumentParser(description="Zero-shot YOLOE count baseline")
    parser.add_argument("--session", default="pilot")
    parser.add_argument("--model", default="yoloe-26s-seg.pt", help="file in models/ (downloads if missing)")
    parser.add_argument("--prompt", action="append", default=[],
                        help='override a class\'s prompt text, e.g. --prompt bolt="hex bolt"')
    parser.add_argument("--visual", help="visual-prompt YAML from draw_reference (instead of text)")
    parser.add_argument("--name", help="run name for the output folder (default: built from the options)")
    args = parser.parse_args()

    quiet_jpeg_warnings()  # the camera's harmless JPEG padding warnings
    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]
    rows = load_session(args.session)
    if not rows:
        raise SystemExit(f"No manifest rows for session {args.session!r}")
    parts = []
    for spec in kits.values():
        parts += [p for p in spec if p not in parts]

    model, prompts = load_model(args, parts)
    mode = "visual" if args.visual else "text " + "/".join(prompts)
    name = args.name or f"{args.session}_{Path(args.model).stem}_{'visual' if args.visual else 'text'}"
    out = REPO / "reports" / "zero_shot" / name
    (out / "wrong").mkdir(parents=True, exist_ok=True)

    all_dets, results = predict_all(model, rows, args, parts)
    scores = [score(rows, all_dets, c, parts, kits) for c in CUTOFFS]
    best = max(scores, key=lambda s: (s["count_acc"], -s["false_passes"]))
    default = next(s for s in scores if s["cutoff"] == 0.25)

    lines = [f"Zero-shot baseline: {args.model}, {mode}, session {args.session} ({len(rows)} images)", ""]
    lines.append(f"{'cutoff':>7} {'count acc':>9} {'decision':>8} {'false pass':>11} {'false reject':>13}"
                 "   mean count error per part")
    for s in scores:
        mae = "  ".join(f"{p} {v:.2f}" for p, v in s["mae"].items())
        mark = "  <- best (optimistic: picked on these images)" if s is best else ""
        mark += "  <- default cutoff" if s is default else ""
        lines.append(f"{s['cutoff']:>7} {s['count_acc']:>9.0%} {s['decision_acc']:>8.0%} "
                     f"{s['false_passes']:>5} of {s['faulty']:<3} {s['false_rejects']:>6} of {s['good']:<4}  {mae}{mark}")
    summary = "\n".join(lines)
    print(summary)
    (out / "summary.txt").write_text(summary + "\n")

    # Per-image detail and failure images at the best cutoff
    with (out / "per_image.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "scenario", "expected", *[f"true_{p}" for p in parts],
                    *[f"pred_{p}" for p in parts], "decision", "correct"])
        for row in rows:
            pred = count_at(all_dets[row["image"]], best["cutoff"], parts)
            true = {p: int(row[f"{p}s"]) for p in parts}
            d = decide(pred, kits[row["kit_id"]])
            w.writerow([row["image"], row["scenario"], row["expected"], *true.values(), *pred.values(),
                        d, decision_correct(d, row["expected"])])
            if pred != true:
                r = results[row["image"]]
                keep = r.boxes.conf >= best["cutoff"]
                cv2.imwrite(str(out / "wrong" / Path(row["image"]).name), r[keep].plot())
    print(f"\nSaved {out.relative_to(REPO)}/ (per_image.csv, wrong/ at cutoff {best['cutoff']})")


if __name__ == "__main__":
    main()
