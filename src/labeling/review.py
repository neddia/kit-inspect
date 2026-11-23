"""Labelling step 2: review every auto-label by eye and fix what's wrong.

Shows each image with its outlines (red bolt, green nut, blue screw; grey =
outside the bowl, not counted) and a side panel comparing the manifest's
counts with what's labelled. Flagged images (counts don't match) come first.

Run from the repo root:
    uv run python -m src.labeling.review                  # flagged first, then the rest
    uv run python -m src.labeling.review --flagged        # only the flagged ones
    uv run python -m src.labeling.review --session val01  # one session
    uv run python -m src.labeling.review --junk           # reviewed junk photos with no "other" box yet
Already-reviewed images are skipped, so you can quit and come back.

Junk (Nov 17: the 4th class): box every foreign object as "other" (key 4), the
same way as a part: drag a box, SAM outlines it, press 4. A coin cell SAM
already outlined as a nut: click it, press 4. Junk outside the bowl gets a box too.

Mouse:
    click a part          select it
    drag a box            SAM 3 outlines the part inside the box (a missed part,
                          or one half of two parts merged into one outline)
    right-click           hint for the SELECTED part: "NOT this" (e.g. on a nut its
                          bolt's outline swallowed). SAM redraws the outline.
    middle-click          hint for the SELECTED part: "INCLUDE this" (e.g. a screw
                          head that was cut off). SAM redraws the outline.
                          Hints add up; clicking a part starts them over.
    n / i, then click     the same hints from the keyboard: n = NOT this, i = INCLUDE this
    r, then drag a box    YOUR box, no SAM (last resort; loses the diagonal shape): replaces the selected part's outline
                          (or adds a new part if none is selected). For when SAM
                          keeps merging touching parts or cuts off a screw head.
Keys:
    1 2 3 4    set the selected part to bolt / nut / screw / other (junk)
    j, then click   JOIN: the selected part and the one you click are one part
               (a screw split in two by a bolt lying across it)
    d          delete the selected outline (a bad outline, a merged pair; NOT junk: junk is 4)
    o          toggle the selected part inside/outside the bowl
    u          undo the last change on this image
    l          hide/show outlines, to see the photo clearly
    r          next drag is a plain box you draw (see above); r again cancels
    SPACE      labels are right: save and go to the next image
    b          back to the previous image
    h          the manifest is right but a part is HIDDEN in the photo:
               marks the image stacked (expected NOT_PASS); counts unchanged
    m          the manifest COUNT was wrong (I miscounted): set it to what's labelled
    q / Esc    quit (everything saved so far is kept)

Labels describe what you can SEE: box a part if you can tell what it is, even
if partly covered; no box for a part you can't see at all.

Writes data/labels/reviewed/<session>/<image>.json, and logs every manifest
change to data/manifest_changes.csv (old value, new value, why).
"""

import argparse
import copy
import csv
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.capture.camera import quiet_jpeg_warnings
from src.capture.capture import DATA, MANIFEST, REPO, expected_for, scenario_for
from src.labeling.auto_label import AUTO, PARTS, REPORT, describe, in_tray, shape_class

REVIEWED = DATA / "labels" / "reviewed"
CHANGES = DATA / "manifest_changes.csv"
CLASSES = PARTS + ["other"]   # "other" = junk / foreign object; keys 1-4
COLORS = {"bolt": (40, 40, 255), "nut": (40, 200, 40), "screw": (255, 140, 0), "other": (255, 0, 255)}
GREY = (150, 150, 150)
PANEL_W = 400
DRAG_MIN = 8  # px; a smaller mouse movement is a click, not a box


# ---------- manifest edits (logged) ----------

def read_manifest():
    with MANIFEST.open(newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def update_manifest(image, changes, reason):
    """Change some fields of one manifest row, and log old -> new."""
    columns, rows = read_manifest()
    row = next(r for r in rows if r["image"] == image)
    log = [(field, row[field], str(value)) for field, value in changes.items() if row[field] != str(value)]
    if not log:
        return row
    for field, value in changes.items():
        row[field] = str(value)
    with MANIFEST.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    new_log = not CHANGES.exists()
    with CHANGES.open("a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["when", "image", "field", "old", "new", "reason"])
        for field, old, new in log:
            w.writerow([datetime.now().isoformat(timespec="seconds"), image, field, old, new, reason])
    return row


def fix_hidden(row, spec):
    """Counts were right but a part is hidden: it's a stacked tray."""
    counts = {p: int(row[f"{p}s"]) for p in PARTS} | {"other": int(row["other"])}
    return {"scenario": scenario_for(counts, spec, stacked=True),
            "expected": expected_for(counts, spec, row["tier"], stacked=True)}


def fix_counts(row, labelled, spec, other=None):
    """I miscounted: the tray really held what's labelled. other = junk boxes labelled (same count as the panel)
    (None keeps the manifest's junk count, for labels made before the 4th class)."""
    other = int(row["other"]) if other is None else other
    counts = labelled | {"other": other}
    stacked = row["scenario"] == "stacked"
    return {**{f"{p}s": labelled[p] for p in PARTS}, "other": other,
            "scenario": scenario_for(counts, spec, stacked),
            "expected": expected_for(counts, spec, row["tier"], stacked)}


# ---------- labels ----------

def label_path(root, image):
    return root / Path(image).with_suffix(".json")


def load_label(image):
    reviewed = label_path(REVIEWED, image)
    return json.loads((reviewed if reviewed.exists() else label_path(AUTO, image)).read_text())


def counts_inside(objects):
    return {p: sum(o["cls"] == p and o["inside"] for o in objects) for p in PARTS}


def hit(objects, x, y):
    """Index of the smallest part whose outline contains (x, y), or None."""
    hits = [(cv2.contourArea(np.array(o["polygon"], np.int32)), i) for i, o in enumerate(objects)
            if cv2.pointPolygonTest(np.array(o["polygon"], np.int32), (float(x), float(y)), False) >= 0]
    return min(hits)[1] if hits else None


def junk_queue(session=None):
    """Reviewed images whose manifest says junk is on the tray, but with no "other" box yet
    (reviewed before the 4th class existed, when junk was just deleted)."""
    _, rows = read_manifest()
    todo = []
    for r in rows:
        if r["split"] not in ("train", "val") or r["tier"] != "normal" or int(r["other"]) < 1:
            continue
        if session and r["session"] != session:
            continue
        reviewed = label_path(REVIEWED, r["image"])
        if reviewed.exists() and not any(o["cls"] == "other" for o in json.loads(reviewed.read_text())["objects"]):
            todo.append(r["image"])
    return todo


def queue(args):
    if args.junk:
        todo = junk_queue(args.session)
        return todo, set(todo)
    with REPORT.open(newline="") as f:
        report = list(csv.DictReader(f))
    if args.session:
        report = [r for r in report if r["session"] == args.session]
    flagged = [r["image"] for r in report if r["flagged"] == "True"]
    rest = [r["image"] for r in report if r["flagged"] != "True"]
    images = flagged if args.flagged else flagged + rest
    todo = [i for i in images if not label_path(REVIEWED, i).exists()]
    return todo, set(flagged)


def sam_mask(sam, image_path, **prompts):
    """SAM's mask for one prompt, or None if it found nothing (or failed). Never raises."""
    try:
        r = sam.predict(str(image_path), verbose=False, **prompts)[0]
    except Exception as e:  # a SAM hiccup must not crash the tool and lose the image's edits
        print(f"SAM error: {e}")
        return None
    if r.masks is None or len(r.masks.data) == 0:
        return None
    mask = r.masks.data[0].cpu().numpy() > 0.5
    return mask if mask.sum() >= 50 else None


# ---------- drawing ----------

def draw(frame, objects, selected, show, drag, info, hints=()):
    img = frame.copy()
    for x, y, include in hints:
        cv2.circle(img, (x, y), 6, (0, 220, 0) if include else (255, 80, 0), -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
    if show:
        for i, o in enumerate(objects):
            color = COLORS[o["cls"]] if o["inside"] else GREY
            poly = np.array(o["polygon"], np.int32)
            cv2.polylines(img, [poly], True, color, 4 if i == selected else 2, cv2.LINE_AA)
            if i == selected:
                x, y = o["box"][:2]
                cv2.putText(img, o["cls"] + ("" if o["inside"] else " (outside)"), (x, max(18, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    if drag:
        cv2.rectangle(img, drag[0], drag[1], (0, 255, 255), 2)
    panel = np.full((img.shape[0], PANEL_W, 3), 32, np.uint8)
    y = 30
    for text, color in info:
        cv2.putText(panel, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
        y += 26
    return np.hstack([img, panel])


def panel_lines(image, n, total, is_flagged, row, objects, selected, status):
    labelled = counts_inside(objects)
    manifest = {p: int(row[f"{p}s"]) for p in PARTS}
    white, red, green, grey = (230, 230, 230), (90, 90, 255), (120, 230, 120), (150, 150, 150)
    lines = [(f"{n}/{total}  {image}", white),
             ("FLAGGED: counts differ" if is_flagged else "counts matched", red if is_flagged else green),
             ("", white),
             (f"{'':8}{'manifest':>9}{'labelled':>10}", grey)]
    for p in PARTS:
        lines.append((f"{p + 's':8}{manifest[p]:>9}{labelled[p]:>10}", green if manifest[p] == labelled[p] else red))
    junk = sum(o["cls"] == "other" for o in objects)
    lines += [(f"other   {row['other']:>9}{junk:>10}",
               green if (int(row["other"]) > 0) == (junk > 0) or int(row["other"]) < 0 else red),
              (f"scenario: {row['scenario']}", white),
              (f"expected: {row['expected']}", white),
              ("", white)]
    outside = sum(not o["inside"] for o in objects)
    if outside:
        lines.append((f"{outside} outside the bowl (grey)", grey))
    if selected is not None:
        o = objects[selected]
        lines.append((f"selected: {o['cls']}  len {o['length']:.0f}  w {o['width']:.0f}", white))
        lines.append((f"  SAM said '{o.get('sam_word', '-')}', source {o['source']}", grey))
    lines += [("", white),
              ("click select  drag = SAM box", grey),
              ("selected part: R-click NOT this,", grey),
              ("   middle-click INCLUDE this", grey),
              ("   (or n / i, then click)", grey),
              ("1 bolt 2 nut 3 screw 4 other(junk)", grey),
              ("d delete outline", grey),
              ("j then click = join two pieces", grey),
              ("o in/out  u undo  l outlines", grey),
              ("r then drag = your own box", grey),
              ("SPACE save+next   b back", grey),
              ("h part hidden  m I miscounted", grey),
              ("", white),
              (status, (0, 220, 255))]
    return lines


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Review and fix auto-labels")
    parser.add_argument("--flagged", action="store_true", help="only images whose counts didn't match")
    parser.add_argument("--session", help="only this session")
    parser.add_argument("--junk", action="store_true",
                        help='reopen reviewed junk photos that have no "other" box yet')
    args = parser.parse_args()

    import yaml
    from ultralytics import SAM

    quiet_jpeg_warnings()
    todo, flagged = queue(args)
    if not todo:
        raise SystemExit("Nothing left to review.")
    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())["kits"]
    print(f"{len(todo)} images to review ({sum(i in flagged for i in todo)} flagged). Loading SAM 3 for box prompts...")
    sam = SAM(str(REPO / "models" / "sam3.pt"))

    window = "review (SPACE save+next, q quit)"
    # GUI_NORMAL: no Qt toolbar or right-click menu, so right/middle clicks reach us
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    mouse = {"down": None, "now": None, "click": None, "box": None, "hint": None}

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["down"], mouse["now"] = (x, y), (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["down"]:
            mouse["now"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and mouse["down"]:
            (x0, y0), mouse["down"] = mouse["down"], None
            if abs(x - x0) < DRAG_MIN and abs(y - y0) < DRAG_MIN:
                mouse["click"] = (x, y)
            else:
                mouse["box"] = [min(x0, x), min(y0, y), max(x0, x), max(y0, y)]
        elif event == cv2.EVENT_RBUTTONUP:
            mouse["hint"] = (x, y, 0)
        elif event == cv2.EVENT_MBUTTONUP:
            mouse["hint"] = (x, y, 1)

    cv2.setMouseCallback(window, on_mouse)

    k = 0
    while 0 <= k < len(todo):
        image = todo[k]
        frame = cv2.imread(str(DATA / "images" / image))
        w = frame.shape[1]
        label = load_label(image)
        objects = label["objects"]
        history, selected, show, status, manual = [], None, True, "", False
        hints, hint_box = [], None  # SAM hints for the selected part, and the box they started from
        hint_key = None             # n or i pressed: the next left-click is a hint, not a selection
        join_key = False            # j pressed: the next left-click picks the part to join
        _, rows = read_manifest()
        row = next(r for r in rows if r["image"] == image)
        spec = kits[row["kit_id"]]
        step = 0  # +1 next, -1 back, None quit

        while True:
            drag = (mouse["down"], mouse["now"]) if mouse["down"] else None
            info = panel_lines(image, k + 1, len(todo), image in flagged, row, objects, selected, status)
            cv2.imshow(window, draw(frame, objects, selected, show, drag, info, hints))
            key = cv2.waitKey(20) & 0xFF

            if mouse["click"]:
                x, y = mouse["click"]
                mouse["click"] = None
                if hint_key is not None:
                    mouse["hint"], hint_key = (x, y, hint_key), None
                elif join_key:
                    join_key = False
                    other = hit(objects, x, y)
                    if other is None or other == selected:
                        status = "join cancelled: click a different part"
                    else:
                        history.append(copy.deepcopy(objects))
                        mask = np.zeros(frame.shape[:2], np.uint8)
                        for i in (selected, other):
                            cv2.fillPoly(mask, [np.array(objects[i]["polygon"], np.int32)], 1)
                        obj = describe(mask > 0)
                        obj.update(cls=objects[selected]["cls"], inside=objects[selected]["inside"], source="review_join")
                        objects[selected] = obj
                        objects.pop(other)
                        selected = selected if selected < other else selected - 1
                        status = f"joined into one {obj['cls']}"
                elif x < w:
                    selected = hit(objects, x, y)
                    hints, hint_box = [], None  # any click starts the hints over
            if mouse["hint"]:
                x, y, include = mouse["hint"]
                mouse["hint"] = None
                if selected is None:
                    status = "select a part first, then right/middle-click"
                elif x < w:
                    hints.append((x, y, include))
                    if hint_box is None:
                        hint_box = list(objects[selected]["box"])
                    for hx, hy, inc in hints:  # an INCLUDE point outside the box widens it
                        if inc:
                            hint_box = [min(hint_box[0], hx - 15), min(hint_box[1], hy - 15),
                                        min(max(hint_box[2], hx + 15), w - 1), max(hint_box[3], hy + 15)]
                    status = "SAM is redrawing with your hints..."
                    cv2.imshow(window, draw(frame, objects, selected, show, None,
                                            panel_lines(image, k + 1, len(todo), image in flagged, row, objects,
                                                        selected, status), hints))
                    cv2.waitKey(1)
                    mask = sam_mask(sam, DATA / "images" / image, bboxes=[hint_box],
                                    points=[[[hx, hy] for hx, hy, _ in hints]],
                                    labels=[[inc for _, _, inc in hints]])
                    if mask is None:
                        status = "SAM found nothing with those hints: click the part to start over"
                    else:
                        history.append(copy.deepcopy(objects))
                        obj = describe(mask)
                        obj.update(cls=objects[selected]["cls"], inside=objects[selected]["inside"],
                                   source="review_sam+hints")
                        objects[selected] = obj
                        status = f"redrawn with {len(hints)} hint(s); u = undo"
            if mouse["box"]:
                box, mouse["box"] = mouse["box"], None
                box[2] = min(box[2], w - 1)
                if manual:
                    manual = False
                    history.append(copy.deepcopy(objects))
                    mask = np.zeros(frame.shape[:2], bool)
                    mask[box[1]:box[3] + 1, box[0]:box[2] + 1] = True
                    obj = describe(mask)
                    if selected is not None:  # keep the part's class, replace its shape
                        obj.update(cls=objects[selected]["cls"], source="manual_box",
                                   inside=objects[selected]["inside"])
                        objects[selected] = obj
                        status = f"{obj['cls']} box redrawn by hand"
                    else:
                        obj.update(cls="bolt", source="manual_box", inside=in_tray(obj, label["tray"]))
                        objects.append(obj)
                        selected = len(objects) - 1
                        status = "new part (your box): press 1/2/3/4 to set its class"
                    continue
                status = "SAM is outlining..."
                cv2.imshow(window, draw(frame, objects, selected, show, None,
                                        panel_lines(image, k + 1, len(todo), image in flagged, row, objects, selected, status)))
                cv2.waitKey(1)
                mask = sam_mask(sam, DATA / "images" / image, bboxes=[box])
                if mask is None:
                    status = "SAM found nothing in that box"
                else:
                    history.append(copy.deepcopy(objects))
                    obj = describe(mask)
                    obj.update(cls=shape_class(obj), source="review_sam", inside=in_tray(obj, label["tray"]))
                    objects.append(obj)
                    selected = len(objects) - 1
                    status = f"added {obj['cls']}: press 1/2/3/4 if wrong"

            if key in (ord("q"), 27):
                step = None
                break
            elif key in (ord("1"), ord("2"), ord("3"), ord("4")) and selected is not None:
                history.append(copy.deepcopy(objects))
                objects[selected]["cls"] = CLASSES[key - ord("1")]
                objects[selected]["source"] = objects[selected]["source"].split("+")[0] + "+fixed"
            elif key == ord("d") and selected is not None:
                history.append(copy.deepcopy(objects))
                status = f"deleted {objects.pop(selected)['cls']}"
                selected = None
            elif key == ord("o") and selected is not None:
                history.append(copy.deepcopy(objects))
                objects[selected]["inside"] = not objects[selected]["inside"]
            elif key == ord("u") and history:
                objects[:] = history.pop()
                selected, status, hints, hint_box = None, "undone", [], None
            elif key == ord("l"):
                show = not show
            elif key == ord("j"):
                if selected is None:
                    status = "select a part first, then j, then click the other piece"
                else:
                    join_key, status = True, "click the other piece of this part"
            elif key in (ord("n"), ord("i")):
                if selected is None:
                    status = "select a part first, then n or i"
                else:
                    hint_key = 1 if key == ord("i") else 0
                    status = f"click where SAM should {'INCLUDE' if hint_key else 'NOT include'}"
            elif key == ord("r"):
                manual = not manual
                status = ("draw YOUR box: replaces the selected part" if selected is not None
                          else "draw YOUR box: adds a new part") if manual else "box mode off"
            elif key == ord("h"):
                row = update_manifest(image, fix_hidden(row, spec), "review: part hidden in photo")
                status = f"manifest: {row['scenario']}, expected {row['expected']}"
            elif key == ord("m"):
                junk = sum(o["cls"] == "other" for o in objects)   # junk on the KIT tag (outside the floor) is still on the tray
                row = update_manifest(image, fix_counts(row, counts_inside(objects), spec, junk), "review: miscounted at capture")
                status = f"manifest counts set to labelled ({row['scenario']})"
            elif key == ord(" "):
                label.update(objects=objects, reviewed=True, reviewed_at=datetime.now().isoformat(timespec="seconds"))
                out = label_path(REVIEWED, image)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(label))
                labelled = counts_inside(objects)
                manifest = {p: int(row[f"{p}s"]) for p in PARTS}
                print(f"saved {image}" + ("" if labelled == manifest or row["scenario"] == "stacked"
                                          else f"  (labelled {labelled} != manifest {manifest})"))
                step = 1
                break
            elif key == ord("b"):
                step = -1
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                step = None
                break

        if step is None:
            break
        k = max(k + step, 0)  # going back reopens the saved version, which you can fix and save again

    cv2.destroyAllWindows()
    done = sum(1 for _ in REVIEWED.rglob("*.json")) if REVIEWED.exists() else 0
    print(f"{done} images reviewed in total.")


if __name__ == "__main__":
    main()
