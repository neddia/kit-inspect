"""Stage 5: score a trained model the way the station uses it: whole trays.

The training table (precision, recall, mAP) grades single boxes. The station
needs EVERY count on a tray right. This runs the model on every image of a
session and scores trays, with the same scoring as the zero-shot baseline
(src/eval/zero_shot.py), so the numbers compare directly.

Decision rule here is only "counts match the kit -> PASS, else FAIL" (rules 4
and 5 of the plan). The REVIEW rules (uncertain confidence, uncovered objects,
scene check) come in Stage 6, so foreign objects on a complete kit WILL show
as false passes here; they're listed by scenario so you can see it.

Only detections inside the bowl are counted (a spare on the desk isn't in the kit).

Run from the repo root:
    uv run python -m src.eval.count_eval --weights runs/detect/runs/stage5/baseline-2/weights/best.pt
    uv run python -m src.eval.count_eval --weights ... --session val01 --name baseline

Outputs, in reports/stage5/<name>/:
    summary.txt    numbers for experiments.md
    per_image.csv  true vs predicted counts at the chosen cutoff
    wrong/         annotated images whose counts were wrong (failure gallery)
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import yaml

from src.capture.camera import quiet_jpeg_warnings
from src.eval.zero_shot import CUTOFFS, DATA, REPO, count_at, decide, decision_correct, load_session, score
from src.labeling.auto_label import PARTS, in_tray
from src.labeling.tray import find_tray


def main():
    parser = argparse.ArgumentParser(description="Score a trained model on whole trays")
    parser.add_argument("--weights", required=True, help="path to best.pt")
    parser.add_argument("--session", default="val01")
    parser.add_argument("--name", help="report folder name (default: <run folder>_<session>)")
    parser.add_argument("--imgsz", type=int, default=640, help="must match training")
    args = parser.parse_args()

    quiet_jpeg_warnings()
    from ultralytics import YOLO

    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]
    rows = load_session(args.session)
    if not rows:
        raise SystemExit(f"No manifest rows for session {args.session!r}")
    weights = Path(args.weights)
    name = args.name or f"{weights.parent.parent.name}_{args.session}"
    out = REPO / "reports" / "stage5" / name
    (out / "wrong").mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    names = model.names
    all_dets, results, outside = {}, {}, Counter()
    for row in rows:
        frame = cv2.imread(str(DATA / "images" / row["image"]))
        tray = find_tray(frame)
        r = model.predict(frame, conf=min(CUTOFFS), imgsz=args.imgsz, verbose=False)[0]
        dets = []
        for (x1, y1, x2, y2), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            if in_tray({"center": [(x1 + x2) / 2, (y1 + y2) / 2]}, tray):
                dets.append((names[int(c)], float(s)))
            elif s >= 0.25:
                outside[row["image"]] += 1
        all_dets[row["image"]] = dets
        results[row["image"]] = r

    scores = [score(rows, all_dets, c, PARTS, kits) for c in CUTOFFS]
    best = max(scores, key=lambda s: (s["count_acc"], -s["false_passes"], -abs(s["cutoff"] - 0.25)))

    lines = [f"Trained model: {weights}, session {args.session} ({len(rows)} images)",
             "Decision: counts match -> PASS, else FAIL (no REVIEW rules yet)", "",
             f"{'cutoff':>7} {'count acc':>9} {'decision':>8} {'false pass':>11} {'false reject':>13}"
             "   mean count error per part"]
    for s in scores:
        mae = "  ".join(f"{p} {v:.2f}" for p, v in s["mae"].items())
        mark = "  <- chosen" if s is best else ""
        lines.append(f"{s['cutoff']:>7} {s['count_acc']:>9.0%} {s['decision_acc']:>8.0%} "
                     f"{s['false_passes']:>5} of {s['faulty']:<3} {s['false_rejects']:>6} of {s['good']:<4}  {mae}{mark}")

    # Where the chosen cutoff goes wrong, by scenario
    by_scenario = {}
    with (out / "per_image.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "scenario", "expected", *[f"true_{p}" for p in PARTS],
                    *[f"pred_{p}" for p in PARTS], "decision", "correct"])
        for row in rows:
            pred = count_at(all_dets[row["image"]], best["cutoff"], PARTS)
            true = {p: int(row[f"{p}s"]) for p in PARTS}
            d = decide(pred, kits[row["kit_id"]])
            ok = decision_correct(d, row["expected"])
            w.writerow([row["image"], row["scenario"], row["expected"], *true.values(), *pred.values(), d, ok])
            s = by_scenario.setdefault(row["scenario"], {"n": 0, "counts_right": 0, "decision_right": 0, "false_pass": 0})
            s["n"] += 1
            s["counts_right"] += pred == true
            s["decision_right"] += ok
            s["false_pass"] += d == "PASS" and row["expected"] != "PASS"
            if pred != true or not ok:
                r = results[row["image"]]
                cv2.imwrite(str(out / "wrong" / Path(row["image"]).name), r[r.boxes.conf >= best["cutoff"]].plot())

    lines += ["", f"By scenario at cutoff {best['cutoff']}:",
              f"{'scenario':<16}{'images':>7}{'counts right':>14}{'decision right':>16}{'false pass':>12}"]
    for name_, s in sorted(by_scenario.items()):
        lines.append(f"{name_:<16}{s['n']:>7}{s['counts_right']:>14}{s['decision_right']:>16}{s['false_pass']:>12}")
    if outside:
        lines += ["", f"Ignored (outside the bowl, conf >= 0.25): {sum(outside.values())} detections in {len(outside)} images"]
    lines += ["", "Cutoff chosen on this session: fine for val (that's what val is for), not a test result."]
    summary = "\n".join(lines)
    print(summary)
    (out / "summary.txt").write_text(summary + "\n")
    print(f"\nSaved {out.relative_to(REPO)}/ (per_image.csv, wrong/)")


if __name__ == "__main__":
    main()
