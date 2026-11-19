"""Guided capture: walks you through every session in configs/capture_plan.yaml.

You don't type counts or session names. For each shot the screen says what
should be on the tray; you shake, set it down, hands out, SPACE. When the
parts change (a bolt out, an extra nut in) it says so in yellow. When a session
ends, it tells you what to change in the room and waits for ENTER.

Read the plan first (no camera needed):
    uv run python -m src.capture.guided --list

Shoot (picks up where you left off; it counts what's already in the manifest):
    uv run python -m src.capture.guided
    uv run python -m src.capture.guided --start train03    # jump to a session first,
                                                          # e.g. daylight while it's light
Keys (click the window first):
    SPACE    save this shot              ENTER    confirm a setup change
    u        undo the last save (and go back to that shot)
    1 2 3..  pick a part   0  pick other    - =   adjust the picked count
             (only if the tray really differs from the plan; the row records
              what's REALLY there)
    h        toggle stacked              c        toggle grid
    q / Esc  quit (progress is kept)

Reuses the rules and manifest code from capture.py, so the manifest rows are
exactly what the free-form capture script would write.
"""

import argparse
import csv
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import yaml

from src.capture.camera import load_camera_config, open_camera, quiet_jpeg_warnings
from src.capture.capture import (
    DATA, KITS, MANIFEST, REPO, all_parts, append_row, expected_for, grab_fresh, is_jpeg,
    manifest_columns, next_index, prepare_manifest, remove_row, scenario_for, write_session_info,
)
from src.capture.preview import draw_overlay, measure

PLAN = REPO / "configs" / "capture_plan.yaml"

# One per shot, in rotation, so layouts don't drift into a habit.
LAYOUTS = [
    "scatter wide", "parts touching", "cluster in the middle", "near the tray edge",
    "flip parts over", "bolts pointing different ways", "nut against a bolt head",
    "screws crossing other parts", "everything to one side", "spread in a line",
]
NOTES = {
    "foreign_object": "add ONE foreign object (coin, clip, key, zip tie)",
    "stacked": "COVER part of one part: nut lying on a bolt, screw under a bolt (all lying flat)",
    "empty": "EMPTY tray: take every part out",
}


def note_for(scenario):
    """The extra instruction for shots that need more than adding or removing parts."""
    if scenario.startswith("swap_"):
        part = scenario.removeprefix("swap_")
        return f"swap one {part} for its SMALLER sibling (the small one counts as 'other')"
    if scenario.startswith("lookalike_"):
        part = scenario.removeprefix("lookalike_")
        return f"REPLACE one {part} with a LOOK-ALIKE that isn't one (coin cell, washer, coin): counts as 'other'"
    return NOTES.get(scenario)


def is_swap(scenario):
    """One real part out, something else in its place: the counts can look complete."""
    return scenario.startswith(("swap_", "lookalike_"))


# ---------- the plan: turn a mix into an ordered shot list ----------

def session_mix(session, plan):
    mix = session.get("mix", "default")
    return plan[f"{mix}_mix"] if isinstance(mix, str) else mix


def build_shots(mix, spec, parts):
    """[{scenario, counts, stacked}, ...] in mix order. Counts are what the tray should hold.

    Each shot is checked against capture.py's scenario_for, so the plan can't
    ask for a tray that would be recorded under a different scenario name.
    """
    base = {p: spec.get(p, 0) for p in parts} | {"other": 0}
    kit_parts = [p for p in parts if spec.get(p, 0) > 0]
    shots = []
    for scenario, n in mix.items():
        for i in range(n):
            counts, stacked = dict(base), False
            if scenario == "complete":
                pass
            elif scenario.startswith("missing_"):
                part = scenario.removeprefix("missing_")
                if part not in kit_parts:
                    raise SystemExit(f"Plan asks for {scenario}, but {part!r} isn't in the kit {kit_parts}")
                counts[part] -= 1
            elif scenario == "extra_part":
                counts[kit_parts[i % len(kit_parts)]] += 1          # rotate which part is extra
            elif scenario == "two_faults":
                a, b = kit_parts[i % len(kit_parts)], kit_parts[(i + 1) % len(kit_parts)]
                counts[a] -= 1                                       # even: two missing,
                counts[b] += -1 if i % 2 == 0 else 1                 # odd: one missing + one extra
            elif is_swap(scenario):
                # One real part out, something in its place: its smaller sibling (swap_) or a
                # look-alike such as a coin cell for a nut (lookalike_). If the model counts the
                # stand-in as the real part, the counts "match": the false-pass case.
                part = scenario.split("_", 1)[1]
                if part not in kit_parts:
                    raise SystemExit(f"Plan asks for {scenario}, but {part!r} isn't in the kit {kit_parts}")
                counts[part] -= 1
                counts["other"] = 1
            elif scenario == "foreign_object":
                counts["other"] = 1
            elif scenario == "stacked":
                stacked = True
            elif scenario == "empty":
                counts = {k: 0 for k in counts}
            else:
                raise SystemExit(f"Unknown scenario {scenario!r} in the capture plan")
            got = scenario_for(counts, spec, stacked)
            want = "two_faults" if is_swap(scenario) else scenario  # what the rules call a swap
            if got != want or min(counts.values()) < 0:
                raise SystemExit(f"Plan asks for {scenario} but that tray would be {got}: {counts}")
            shots.append({"scenario": scenario, "counts": counts, "stacked": stacked})
    return shots


def change_text(before, after):
    """'bolt -1, nut +1' : what to add or take away since the last tray."""
    return ", ".join(f"{p} {after[p] - before[p]:+d}" for p in after if after[p] != before[p])


def tray_text(counts, parts):
    return "  ".join(f"{counts[p]} {p}s" for p in parts) + (f"  + {counts['other']} other" if counts["other"] else "")


def done_by_session():
    if not MANIFEST.exists() or MANIFEST.stat().st_size == 0:
        return {}
    done = {}
    with MANIFEST.open(newline="") as f:
        for r in csv.DictReader(f):
            done[r["session"]] = done.get(r["session"], 0) + 1
    return done


def position(order, shots, done):
    """(session index, shot index) of the next shot to take, or None when everything's done."""
    for i, s in enumerate(order):
        if done.get(s["name"], 0) < len(shots[s["name"]]):
            return i, done.get(s["name"], 0)
    return None


# ---------- drawing ----------

TOP_ROWS, BOTTOM_ROWS = 5, 5   # fixed strip heights, so the window doesn't jump between shots


def strip(width, lines, color, scale, line_h, thick, rows):
    """Text on a black strip, wrapped to the width. The strips sit above and below the camera
    picture, never on it, so the whole tray stays visible."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    wrapped = []
    for text in lines:
        if cv2.getTextSize(text, font, scale, thick)[0][0] <= width - 28:
            wrapped.append(text)                  # fits: keep its spacing as written
            continue
        line = ""
        for word in text.split():
            trial = f"{line} {word}".strip()
            if line and cv2.getTextSize(trial, font, scale, thick)[0][0] > width - 28:
                wrapped.append(line)
                line = word
            else:
                line = trial
        wrapped.append(line)
    img = np.zeros((line_h * max(rows, len(wrapped)) + 14, width, 3), np.uint8)
    for i, text in enumerate(wrapped):
        cv2.putText(img, text, (14, line_h - 6 + i * line_h), font, scale, color, thick, cv2.LINE_AA)
    return img


def banner(width, lines, color):
    """What to do for this shot: the strip under the picture."""
    return strip(width, lines, color, 0.85, 38, 2, BOTTOM_ROWS)


def section_for(session, shot):
    """The short name of the part of the session this shot is in (the text before ':' of the last pause)."""
    pauses = {int(k): v for k, v in (session.get("pauses") or {}).items() if int(k) <= shot}
    return pauses[max(pauses)].split(":")[0] if pauses else None


YELLOW, GREEN, CYAN = (0, 230, 255), (80, 230, 80), (255, 220, 120)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Guided capture of every planned session")
    parser.add_argument("--plan", default=str(PLAN))
    parser.add_argument("--start", help="session to do first (the rest follow in plan order)")
    parser.add_argument("--list", action="store_true", help="print the plan and progress, no camera")
    parser.add_argument("--show", metavar="SESSION", help="print every shot of one session, no camera")
    parser.add_argument("--verdict", action="store_true",
                        help="after each shot, show what the current model decides (find what fools it)")
    args = parser.parse_args()

    plan = yaml.safe_load(Path(args.plan).read_text())
    kits = yaml.safe_load(KITS.read_text())["kits"]
    kit_id = plan["kit"]
    spec, parts = kits[kit_id], all_parts(kits)
    columns = manifest_columns(parts)
    order = plan["sessions"]
    if args.start:
        names = [s["name"] for s in order]
        if args.start not in names:
            raise SystemExit(f"No session {args.start!r} in the plan: {names}")
        k = names.index(args.start)
        order = order[k:] + order[:k]
    shots = {s["name"]: build_shots(session_mix(s, plan), spec, parts) for s in order}
    done = done_by_session()

    if args.show:
        if args.show not in shots:
            raise SystemExit(f"No session {args.show!r} in the plan: {list(shots)}")
        session = next(s for s in order if s["name"] == args.show)
        before = {p: spec.get(p, 0) for p in parts} | {"other": 0}
        print(f"{args.show}: {session['setup']}")
        for k, shot in enumerate(shots[args.show]):
            change = change_text(before, shot["counts"])
            pause = (session.get("pauses") or {}).get(k + 1)
            if pause:
                print(f"     --- pause: {pause} ---")
            print(f"{k + 1:>4}  {shot['scenario']:<15} {tray_text(shot['counts'], parts):<42}"
                  f"{'stacked  ' if shot['stacked'] else ''}{('CHANGE: ' + change) if change else ''}")
            before = shot["counts"]
        return

    if args.list:
        for s in order:
            n, d = len(shots[s["name"]]), done.get(s["name"], 0)
            mark = "done" if d >= n else f"{d}/{n}"
            print(f"{s['name']:<15} {s['split']:<5} {s.get('tier', 'normal'):<7} {mark:>6}  {s['setup']}")
            if s.get("pauses"):
                for at, text in s["pauses"].items():
                    print(f"{'':>37}  at shot {at}: {text}")
        print(f"\n{sum(len(v) for v in shots.values())} shots planned; "
              f"{sum(min(done.get(k, 0), len(v)) for k, v in shots.items())} done.")
        return

    prepare_manifest(MANIFEST, columns)
    quiet_jpeg_warnings()
    inspector = None
    if args.verdict:
        from src.inspection.draw import draw
        from src.inspection.pipeline import Inspector
        inspector = Inspector()
        verdict_window = "model's verdict on the last shot"
    cam_cfg = load_camera_config()
    cap = open_camera(cam_cfg)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # the camera's own JPEG bytes

    window = "guided capture (SPACE save, ENTER confirm, u undo, q quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    total_h = cam_cfg["height"] + (26 * TOP_ROWS + 14) + (38 * BOTTOM_ROWS + 14)
    k = min(1.0, 1000 / total_h)                       # picture + both strips, fits a 1080p screen
    cv2.resizeWindow(window, int(cam_cfg["width"] * k), int(total_h * k))

    saved = []            # (image, session) saved in this run, for undo
    status, status_until = "", 0.0
    show_grid = False
    picked = 0
    pickable = [*parts, "other"]
    current = None        # (session index, shot index) the screen is showing
    waiting = None        # setup text to confirm with ENTER, or None

    def go_to(pos, confirm_setup):
        """Load the planned tray for this shot and decide whether to stop for a setup change."""
        nonlocal current, counts, stacked, waiting
        si, k = pos
        session = order[si]
        shot = shots[session["name"]][k]
        counts, stacked = dict(shot["counts"]), shot["stacked"]
        pause = (session.get("pauses") or {}).get(k + 1)
        if confirm_setup and (current is None or current[0] != si or k == 0):
            waiting = session["setup"] + (f"  THEN: {pause}" if pause else "")
        elif confirm_setup and pause:
            waiting = pause
        current = pos

    counts, stacked = {}, False
    pos = position(order, shots, done)
    if pos is None:
        raise SystemExit("Every session in the plan is done. (--list to check)")
    go_to(pos, confirm_setup=True)
    if current[1] > 0:
        waiting += "  (resuming)"

    while True:
        ok, raw = cap.read()
        if not ok:
            print("Lost the camera. On WSL: usbipd.exe attach --wsl --busid 8-4")
            break
        frame = cv2.imdecode(raw, cv2.IMREAD_COLOR) if is_jpeg(raw) else None
        if frame is None:
            continue

        si, k = current
        session = order[si]
        name, tier = session["name"], session.get("tier", "normal")
        plan_shot = shots[name][k]
        scenario = scenario_for(counts, spec, stacked)
        as_planned = counts == plan_shot["counts"] and stacked == plan_shot["stacked"]
        if as_planned and is_swap(plan_shot["scenario"]):
            scenario = plan_shot["scenario"]  # the rules can't tell a swap from two faults; the plan can
        expected = expected_for(counts, spec, tier, stacked)
        brightness, clipped, _ = measure(frame)

        sessions_left = sum(done.get(s["name"], 0) < len(shots[s["name"]]) for s in order)
        top = [
            f"{name} ({session['split']}, {tier})   shot {k + 1}/{len(shots[name])}   "
            f"sessions left: {sessions_left}",
            "TRAY: " + "  ".join(
                f"{'>' if pickable[picked] == p else ' '}[{(i + 1) % len(pickable)}] {p}s {counts[p]}"
                for i, p in enumerate(pickable)) + f"   stacked {'YES' if stacked else 'no'} [h]",
            f"-> {scenario}, expected {expected}"
            + ("" if as_planned else "   (differs from plan: recorded as what you set)"),
            f"light: brightness {brightness:.0f}, clipped {clipped:.1f}%"
            + ("   TOO BRIGHT? lots of pure white" if clipped > 2 else ""),
            status if time.time() < status_until else "",
        ]
        picture = draw_overlay(frame, [], show_grid)    # grid only: the text goes in strips around it
        width = picture.shape[1]
        header = strip(width, [t for t in top if t], (255, 255, 255), 0.6, 26, 1, TOP_ROWS)

        if waiting:
            footer = banner(width, [f"SET UP: {waiting}", "then press ENTER"], CYAN)
        else:
            before = shots[name][k - 1]["counts"] if k else {p: spec.get(p, 0) for p in parts} | {"other": 0}
            change = change_text(before, plan_shot["counts"])
            layout = LAYOUTS[k % len(LAYOUTS)]
            if session.get("loose"):
                layout += ", tray down loose"
            lines = [f"SHOT: {plan_shot['scenario'].replace('_', ' ').upper()}   ->  {tray_text(plan_shot['counts'], parts)}"]
            note = note_for(plan_shot["scenario"])
            if change:
                lines.append(f"CHANGE PARTS: {change}")
            if note:
                lines.append(note.upper())
            section = section_for(session, k + 1)
            if section or session.get("hint"):
                lines.append(section or session["hint"])
            lines.append(f"layout: {layout}. Hands out, SPACE")
            footer = banner(width, lines, YELLOW if change or note else GREEN)
        cv2.imshow(window, np.vstack([header, picture, footer]))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

        elif key in (13, 10) and waiting:
            waiting = None

        elif key == ord(" ") and not waiting:
            fresh = grab_fresh(cap)
            if fresh is None:
                status, status_until = "SAVE FAILED: no frame from camera", time.time() + 3
                continue
            session_dir = DATA / "images" / name
            session_dir.mkdir(parents=True, exist_ok=True)
            write_session_info(session_dir, SimpleNamespace(
                session=name, split=session["split"], tier=tier, kit=kit_id, notes=session["setup"]), cam_cfg)
            file = f"{name}_{next_index(session_dir):04d}.jpg"
            (session_dir / file).write_bytes(fresh.tobytes())
            image = f"{name}/{file}"
            append_row(MANIFEST, {
                "image": image, "session": name, "split": session["split"], "tier": tier,
                "kit_id": kit_id, **{f"{p}s": counts[p] for p in parts},
                "other": counts["other"], "expected": expected, "scenario": scenario,
            }, columns)
            saved.append((image, name))
            done[name] = done.get(name, 0) + 1
            status, status_until = f"SAVED {image} ({scenario}, {expected})", time.time() + 2
            if inspector:
                photo = cv2.imdecode(fresh, cv2.IMREAD_COLOR)
                result = inspector.inspect(photo, kit_id)
                dec = result["decision"]
                if expected == "PASS":
                    note = {"PASS": "", "REVIEW": "  (sent a good kit to review)", "FAIL": "  <-- FALSE REJECT"}[dec.result]
                else:
                    note = "  <-- FALSE PASS!" if dec.result == "PASS" else ""
                status += f"   MODEL: {dec.result}{note}"
                status_until = time.time() + 4
                shown = draw(photo, result, inspector.cfg["decision"]["high"])
                cv2.imshow(verdict_window, cv2.resize(shown, (shown.shape[1] * 3 // 4, shown.shape[0] * 3 // 4)))
            print(status)
            pos = position(order, shots, done)
            if pos is None:
                print("All planned sessions done.")
                break
            if pos[0] != si:
                print(f"Session {name} done.")
            go_to(pos, confirm_setup=True)

        elif key == ord("u"):
            if not saved:
                status, status_until = "nothing to undo in this run", time.time() + 2
                continue
            image, name = saved.pop()
            (DATA / "images" / image).unlink(missing_ok=True)
            remove_row(MANIFEST, image, columns)
            done[name] -= 1
            waiting = None
            go_to(position(order, shots, done), confirm_setup=False)
            status, status_until = f"UNDID {image}: retake this shot", time.time() + 3
            print(status)

        elif ord("0") <= key <= ord("9") and not waiting:
            digit = key - ord("0")
            index = len(pickable) - 1 if digit == 0 else digit - 1
            if index < len(pickable):
                picked = index
        elif key in (ord("-"), ord("=")) and not waiting:
            part = pickable[picked]
            counts[part] = max(0, counts[part] + (1 if key == ord("=") else -1))
        elif key == ord("h") and not waiting:
            stacked = not stacked
        elif key == ord("c"):
            show_grid = not show_grid

        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"This run: {len(saved)} images saved.")


if __name__ == "__main__":
    main()
