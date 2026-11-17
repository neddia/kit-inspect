"""Stage 6: score the full station decision (PASS / FAIL / REVIEW) against the manifest.

Unlike count_eval (counts only, PASS/FAIL), this runs the whole inspection:
model + scene check + unknown-object check + decision rules, with the settings
in configs/station.yaml. The kit ID is taken from the manifest (standing in for
the PLC).

What counts as right (expected comes from the manifest):
    expected PASS      -> PASS right; REVIEW = a review (costs a human, not a defect);
                          FAIL = false reject
    expected FAIL      -> FAIL or REVIEW right; PASS = FALSE PASS (the headline number)
    expected NOT_PASS  -> FAIL or REVIEW right; PASS = FALSE PASS

Run from the repo root:
    uv run python -m src.eval.decision_eval                       # val01
    uv run python -m src.eval.decision_eval --session val01 --name v1
    uv run python -m src.eval.decision_eval --name int8 --model models/kit_yolo26n_640_v1_int8_openvino_model

Writes reports/stage6/<name>/: summary.txt, per_image.csv, and images/ with an
annotated picture of every tray that wasn't a clean right answer.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2

from src.capture.camera import quiet_jpeg_warnings
from src.inspection.pipeline import REPO, Inspector

DATA = REPO / "data"
COLORS = {"PASS": (60, 200, 60), "FAIL": (60, 60, 230), "REVIEW": (0, 200, 255)}


def outcome(result, expected):
    if expected == "PASS":
        return {"PASS": "right", "REVIEW": "review", "FAIL": "false reject"}[result]
    return "FALSE PASS" if result == "PASS" else "right"


def draw(frame, out):
    img = frame.copy()
    scene = out["scene"]
    if scene["floor"] is not None:
        x, y, r = scene["floor"][:3]
        cv2.circle(img, (int(x), int(y)), int(r), (255, 160, 0), 1)
    for d in out["detections"]:
        x1, y1, x2, y2 = map(int, d.box)
        color = (150, 150, 150) if not d.in_bowl else (60, 200, 60) if d.conf >= 0.6 else (0, 200, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{d.cls} {d.conf:.2f}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    for u in out["unknown_objects"]:
        x1, y1, x2, y2 = u["box"]
        cv2.rectangle(img, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), (255, 0, 255), 2)
    dec = out["decision"]
    cv2.rectangle(img, (0, 0), (img.shape[1], 34 + 22 * len(dec.reasons)), (0, 0, 0), -1)
    cv2.putText(img, dec.result, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[dec.result], 2, cv2.LINE_AA)
    for i, reason in enumerate(dec.reasons):
        cv2.putText(img, reason, (8, 52 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main():
    parser = argparse.ArgumentParser(description="Score the full PASS/FAIL/REVIEW decision")
    parser.add_argument("--session", nargs="+", default=["val01"])
    parser.add_argument("--name", default="val01")
    parser.add_argument("--model", help="model to use instead of station.yaml's (e.g. an OpenVINO export folder)")
    args = parser.parse_args()

    quiet_jpeg_warnings()
    with (DATA / "manifest.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["session"] in args.session]
    out_dir = REPO / "reports" / "stage6" / args.name
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    model = None
    if args.model:
        from ultralytics import YOLO
        model = YOLO(args.model, task="detect")
    inspector = Inspector(model=model)
    results, by_scenario, times = [], {}, []
    for row in rows:
        frame = cv2.imread(str(DATA / "images" / row["image"]))
        out = inspector.inspect(frame, row["kit_id"])
        dec = out["decision"]
        o = outcome(dec.result, row["expected"])
        times.append(out["ms"]["total"])
        results.append({"image": row["image"], "scenario": row["scenario"], "expected": row["expected"],
                        "result": dec.result, "outcome": o, "reasons": "; ".join(dec.reasons)})
        by_scenario.setdefault(row["scenario"], Counter())[dec.result] += 1
        if o != "right" or dec.result == "REVIEW":  # every REVIEW (right or not) and every wrong answer
            cv2.imwrite(str(out_dir / "images" / Path(row["image"]).name), draw(frame, out))

    n = len(results)
    good = [r for r in results if r["expected"] == "PASS"]
    faulty = [r for r in results if r["expected"] != "PASS"]
    tally = Counter(r["outcome"] for r in results)
    reviews = sum(r["result"] == "REVIEW" for r in results)
    model_name = args.model or inspector.cfg["model"]
    lines = [f"Full decision on {', '.join(args.session)} ({n} trays). Model: {model_name}. Settings: configs/station.yaml", "",
             f"False passes:    {tally['FALSE PASS']} of {len(faulty)} faulty trays   <- the headline number",
             f"False rejects:   {tally['false reject']} of {len(good)} good trays (FAIL on a good kit)",
             f"Good kits sent to REVIEW: {tally['review']} of {len(good)}",
             f"Review rate:     {reviews} of {n} trays ({reviews / n:.0%})   target: under 5%",
             f"Time per tray:   median {sorted(times)[n // 2]:.0f} ms (this PC; laptop measured in Stage 7)", "",
             f"{'scenario':<16}{'trays':>6}{'PASS':>6}{'FAIL':>6}{'REVIEW':>8}"]
    for s, c in sorted(by_scenario.items()):
        lines.append(f"{s:<16}{sum(c.values()):>6}{c['PASS']:>6}{c['FAIL']:>6}{c['REVIEW']:>8}")
    lines += ["", "Every REVIEW and every wrong answer, with reasons:"]
    for r in results:
        if r["result"] == "REVIEW" or r["outcome"] not in ("right",):
            lines.append(f"  {r['image']:<26} {r['scenario']:<15} {r['result']:<7} {r['outcome']:<12} {r['reasons']}")
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "summary.txt").write_text(summary + "\n")
    with (out_dir / "per_image.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"\nSaved {out_dir.relative_to(REPO)}/ (annotated pictures of every REVIEW and wrong answer in images/)")


if __name__ == "__main__":
    main()
