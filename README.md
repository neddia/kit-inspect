# kit-inspect

A fixed-camera inspection station for fastener kits. It counts the bolts, nuts and screws on a tray, spots foreign objects, reads the kit label, decides **PASS / FAIL / REVIEW**, and hands the result to a **simulated** Allen-Bradley PLC. The PLC releases the kit only on a PASS for the current cycle; anything else, including silence, holds it.

| Kit | Bolts | Nuts | Screws |
| --- | --- | --- | --- |
| KIT-A (code 1) | 3 | 3 | 2 |
| KIT-B (code 2) | 2 | 4 | 2 |

> Work in progress. Sections marked *(Stage 8)* get filled in when the test sessions are done.

## Run the demo

```bash
uv sync                          # once (training PC). Station laptop: uv sync --no-default-groups --group prod
uv run kit demo                  # PLC operator panel + vision station, as two separate apps
```

In the **PLC panel**: `1` / `2` picks the kit the line is building, `SPACE` sends a kit, `a` sends one every 4 s.
The big lamp is the PLC's real `Release_Permit` tag. The **station window** shows the camera, what it found and why.

Things to try:
- A correct kit → **RELEASE**. Take a nut out → **HOLD**, "missing 1 nut".
- Put the KIT-A label on a KIT-B order → **HOLD**, "label says KIT-A, PLC expects KIT-B".
- Close the station window while the lamp says RELEASE → the PLC sees the heartbeat stop and drops to **HOLD** by itself.
- Unplug the camera → the station raises a fault at once; plug it back in and it recovers.

No camera? `uv run kit demo --images data/images/hardval2` replays photos instead.

## How it works

```
  PLC operator panel (simulated PLC)                 Vision station
  ──────────────────────────────────                 ──────────────
  kit code, cycle id, trigger        ───────────>    fresh photo after the trigger
                                                     detector (YOLO26n: bolt, nut, screw, other)
                                                     scene check (tray found, position, light)
                                                     unknown-object check (OpenCV)
                                                     kit label check (OCR)
                                                     rules -> PASS / FAIL / REVIEW + reasons
  release only on a matched PASS     <───────────    result + cycle id, done
  watchdog: heartbeat lost -> HOLD   <───────────    heartbeat every 500 ms, fault flag
```

The model reports what it sees; plain code decides (`src/inspection/decision.py`). Every inspection is kept (photo + database row) in `data/history/`.

## Improve the model

| Step | Command | What it does |
| --- | --- | --- |
| Set up | `uv run kit camera` | Live view and locked camera settings |
| Photograph | `uv run kit capture` | Guided sessions from `configs/capture_plan.yaml`; records the truth for every photo |
| Recover real failures | `uv run kit harvest` | Pull the exact station frame that fooled it (by history `#id`) into a session |
| Label | `uv run kit label` | SAM 3 outlines every new photo, then a review tool to fix them by eye |
| Train | `uv run kit train NAME` | A named **candidate** model (PyTorch + OpenVINO). The station keeps its current model |
| Evaluate | `uv run kit eval --model ...` | PASS / FAIL / REVIEW scored against the recorded truth |

A candidate becomes the station's model only when you set `inspection.model` in `configs/station.yaml`. Every experiment and decision is logged in `experiments.md`.

## Results
*(Stage 8)*: every number here will come from a script in this repo, run on test sessions shot after the model and thresholds were frozen.

## Repo layout

| Folder | What's in it |
| --- | --- |
| `configs/` | Kit specs, station settings, capture plan |
| `data/` | `manifest.csv` (one row per photo, with the truth), labels; photos and history not in Git |
| `src/capture` | Camera, guided capture |
| `src/labeling` | SAM 3 auto-labelling, review tool, YOLO export |
| `src/train` | Training |
| `src/inspection` | Detector pipeline, scene check, unknown-object check, kit label OCR, decision rules |
| `src/plc` | Simulated PLC, its operator panel, tag map |
| `src/station` | The vision station, camera feed, PLC handshake, history |
| `src/eval` | Scoring, plus the earlier experiments (zero-shot baseline, PatchCore) |
| `tests/` | Decision rules, PLC handshake and fail-safes, label check, unknown-object check |

## Known limits
*(Stage 8)*

- The PLC is simulated (cpppo + pylogix). No real controller was used.
- Junk the model has never seen relies on the OpenCV check, which can't see objects inside a part's box.

## License note
Ultralytics YOLO models are AGPL-3.0 licensed.
