# kit-inspect

A fixed-camera inspection station that checks a fastener kit (2 bolts, 2 nuts, 4 washers), reads its kit ID, decides PASS / FAIL / REVIEW, and hands the result to a **simulated** Allen-Bradley PLC. The kit is released only on PASS.

> Work in progress. Sections marked *(Stage N)* get filled in when that stage is done.

## Results
*(Stage 8)*: every number here comes from a script in this repo, run on real captured images.

## Architecture
*(Stage 7)*

## How to run

Setup (needs [uv](https://docs.astral.sh/uv/) and x86-64 Linux). Pick the line for your machine:

```bash
uv sync                                    # training PC: CUDA PyTorch (~6 GB)
uv sync --no-default-groups --group prod   # station: CPU-only PyTorch (~1.6 GB)

uv run python scripts/check_env.py         # confirm the environment works
```

## Repo layout

| Folder | What's in it |
| --- | --- |
| `configs/` | Kit specs and station settings |
| `data/` | `manifest.csv` (one row per image), images (not in Git), labels |
| `src/capture` | Capture script |
| `src/labeling` | SAM 3 auto-labeling, mask-to-box, export |
| `src/train` | Training, validation, OpenVINO export |
| `src/inspection` | Scene checks, detector, OCR, decision logic |
| `src/plc` | Simulated PLC and tag client |
| `src/station` | FastAPI app, dashboard, history |
| `src/eval` | Metrics, failure gallery, report |
| `scripts/` | One-off utilities (environment check) |
| `tests/` | Decision logic and handshake tests |

## Known limits
*(Stage 8)*

- The PLC is simulated (cpppo). No real controller was used.

## License note
Ultralytics YOLO models are AGPL-3.0 licensed.
