"""The decision: PASS, FAIL or REVIEW, with plain-English reasons. Plain code, no model.

The model reports what it sees (class, confidence, box). These rules decide.
They run in order; the first that applies wins:

  1. Scene problem (no tray, too dark/bright, tray off position)   -> REVIEW
  2. Kit ID from the PLC missing or not in kits.yaml                 -> REVIEW
  2b. Kit label on the tray (OCR, if enabled; src/inspection/kit_label.py):
        says a DIFFERENT kit  -> FAIL (wrong kit on the line, whatever the counts)
        missing / unreadable  -> one more REVIEW reason (a FAIL from the counts stays a FAIL)
  3. Sort detections in the bowl by confidence:
        below `low`              ignored (noise)
        `low` up to `high`       UNSURE: might or might not be a real part
        `high` and up            SURE
     A detection overlapping a more confident one (IoU > overlap_iou) counts as
     unsure too: it may be the same part seen twice.
  4. Could the tray be complete? For each part: sure <= spec <= sure + unsure.
        No  -> FAIL. Whatever the unsure ones are, the counts can't match.
               (Reasons say what's off: "missing 1 nut", "1 extra bolt".)
        Yes, but some detections are unsure             -> REVIEW ("unsure about ...")
        Yes, sure counts match, unknown object on floor -> REVIEW ("unknown object")
        Yes, sure counts match, nothing else            -> PASS

Foreign objects the model knows ("other", the 4th class, Nov 17) need no rule of
their own: no kit wants any, so step 4 already does the right thing. A sure one
can't be part of a complete kit -> FAIL ("foreign object"); an unsure one could
be nothing -> REVIEW ("maybe a foreign object"). The OpenCV unknown-object check
stays as the backup for junk the model has never seen.

Why "could it be complete" instead of "any doubt -> REVIEW": an unsure detection
only matters if it could change the answer. A tray clearly missing two bolts is a
FAIL whatever a shaky nut detection turns out to be. That keeps REVIEW (a human's
time) for the cases that need it. FAIL and REVIEW both stop the kit, so this
never risks a false pass.
"""

from dataclasses import dataclass, field

PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"
DEFAULTS = dict(low=0.25, high=0.6, overlap_iou=0.6)


@dataclass
class Detection:
    cls: str
    conf: float
    box: list            # [x1, y1, x2, y2] pixels
    in_bowl: bool = True
    on_label: bool = False   # lettering on the kit label, read as a part (not counted)


@dataclass
class Decision:
    result: str
    reasons: list
    sure: dict = field(default_factory=dict)     # part -> count of sure detections
    unsure: dict = field(default_factory=dict)   # part -> count of unsure detections
    spec: dict = field(default_factory=dict)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def sort_detections(detections, low, high, overlap_iou):
    """Split detections in the bowl into sure and unsure lists (noise below `low` is dropped)."""
    kept = sorted((d for d in detections if d.in_bowl and d.conf >= low), key=lambda d: -d.conf)
    sure, unsure = [], []
    for d in kept:
        overlaps = any(iou(d.box, s.box) > overlap_iou for s in sure + unsure)
        (sure if d.conf >= high and not overlaps else unsure).append(d)
    return sure, unsure


FOREIGN = "other"   # the model's class for junk on the tray


def plural(n, part):
    return f"{n} {part}" + ("s" if n != 1 else "")


def extra(n, part):
    return plural(n, "foreign object") if part == FOREIGN else f"{n} extra {part}" + ("s" if n != 1 else "")


def unsure_about(n, part, confs):
    what = "maybe " + ("a foreign object" if n == 1 else f"{n} foreign objects") if part == FOREIGN \
        else f"unsure about {plural(n, part)}"
    return f"{what} ({confs})"


def decide(kit_id, kits, detections, scene_problems=(), unknown_objects=(), low=0.25, high=0.6, overlap_iou=0.6,
           label=None):
    """label: kit_label.check_label() output, or None when the label check is off."""
    if scene_problems:
        return Decision(REVIEW, list(scene_problems))
    if not kit_id or kit_id not in kits:
        return Decision(REVIEW, [f"kit ID {kit_id!r} not in kits.yaml" if kit_id else "no kit ID from the PLC"])
    if label and label["status"] == "mismatch":
        return Decision(FAIL, [label["reason"]])
    label_doubt = label["reason"] if label and label["status"] == "review" else None

    spec = kits[kit_id]
    sure_list, unsure_list = sort_detections(detections, low, high, overlap_iou)
    parts = list(spec) + sorted({d.cls for d in sure_list + unsure_list} - set(spec))
    sure = {p: sum(d.cls == p for d in sure_list) for p in parts}
    unsure = {p: sum(d.cls == p for d in unsure_list) for p in parts}
    want = {p: spec.get(p, 0) for p in parts}

    could_be_complete = all(sure[p] <= want[p] <= sure[p] + unsure[p] for p in parts)
    if not could_be_complete:
        reasons = []
        for p in parts:
            closest = min(max(want[p], sure[p]), sure[p] + unsure[p])  # the count nearest the spec it could be
            if closest < want[p]:
                reasons.append(f"missing {plural(want[p] - closest, p)}")
            elif closest > want[p]:
                reasons.append(extra(closest - want[p], p))
        if unknown_objects:
            reasons.append(f"also: {plural(len(unknown_objects), 'unknown object')} on the tray")
        if label_doubt:
            reasons.append(f"also: {label_doubt}")
        return Decision(FAIL, reasons, sure, unsure, want)

    reasons = []
    for p in parts:
        if unsure[p]:
            reasons.append(unsure_about(unsure[p], p, ", ".join(f"{d.conf:.2f}" for d in unsure_list if d.cls == p)))
    if unknown_objects:
        reasons.append(f"{plural(len(unknown_objects), 'unknown object')} on the tray")
    if label_doubt:
        reasons.append(label_doubt)
    if reasons:
        return Decision(REVIEW, reasons, sure, unsure, want)
    return Decision(PASS, ["counts match " + kit_id], sure, unsure, want)
