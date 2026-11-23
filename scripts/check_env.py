"""Environment check: is every library installed and working in its most basic form?

Run from the repo root:
    uv run python scripts/check_env.py

Each check does one small real thing (not just an import) and prints PASS or FAIL.
The checks that must pass are 1 and 3:
CUDA works, and a pretrained YOLO26n saves an annotated image.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "reports" / "setup"


def check_torch_cuda():
    import torch

    if torch.version.cuda is None:
        # CPU-only build: what the laptop (prod group) should have.
        return f"torch {torch.__version__}, CPU-only build (correct for the station laptop)"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA build of torch, but torch.cuda.is_available() is False")
    # Do a tiny bit of math on the GPU to prove it really runs there.
    x = torch.rand(1000, 1000, device="cuda")
    y = (x @ x).sum().item()
    gpu = torch.cuda.get_device_name(0)
    return f"torch {torch.__version__}, CUDA {torch.version.cuda}, {gpu}, matmul ok ({y:.0f})"


def check_opencv():
    import cv2
    import numpy as np

    # Draw a white circle on a black image and count the white pixels.
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(img, (50, 50), 20, (255, 255, 255), -1)
    white = int((img[:, :, 0] == 255).sum())
    return f"opencv {cv2.__version__}, drew a circle of {white} px"


def check_yolo26n():
    import cv2
    import torch
    from ultralytics import YOLO, __version__
    from ultralytics.utils import ASSETS

    device = 0 if torch.cuda.is_available() else "cpu"  # 0 = the first GPU

    # Pretrained on COCO (80 everyday classes). It knows "person" and "bus",
    # not "washer" - teaching it washers takes fine-tuning.
    model = YOLO(str(REPO / "models" / "yolo26n.pt"))  # downloads (~5 MB) the first time
    result = model(ASSETS / "bus.jpg", device=device, verbose=False)[0]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "yolo26n_bus.jpg"
    cv2.imwrite(str(out_path), result.plot())  # plot() draws the boxes and returns an image

    names = [result.names[int(c)] for c in result.boxes.cls]
    ms = result.speed["inference"]
    return f"ultralytics {__version__} on {'GPU' if device == 0 else 'CPU'}, found {len(names)} objects {sorted(set(names))} in {ms:.0f} ms, saved {out_path.relative_to(REPO)}"


def check_openvino():
    import openvino as ov

    devices = ov.Core().available_devices
    if "CPU" not in devices:
        raise RuntimeError(f"no CPU device, got {devices}")
    return f"openvino {ov.__version__}, devices {devices}"


def check_transformers():
    import transformers
    from transformers import Florence2ForConditionalGeneration  # noqa: F401  (OCR model class)

    return f"transformers {transformers.__version__}, Florence-2 class available (weights download on first use)"


def check_fastapi():
    import fastapi
    from fastapi import FastAPI

    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True}

    routes = [r.path for r in app.routes if r.path == "/health"]
    return f"fastapi {fastapi.__version__}, built an app with route {routes}"


def check_plc_round_trip():
    import subprocess
    import sys
    import time
    from importlib.metadata import version

    from pylogix import PLC

    # Start the SIMULATED PLC (cpppo) in the background with two tags,
    # then write each one from the vision side (pylogix) and read it back.
    server = subprocess.Popen(
        [sys.executable, "-m", "cpppo.server.enip", "-a", "127.0.0.1:44818",
         "--no-print", "Insp_CycleID=DINT", "Insp_Trigger=BOOL"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(2)  # give the server a moment to start listening
        with PLC("127.0.0.1") as comm:
            for tag, value in [("Insp_CycleID", 42), ("Insp_Trigger", True)]:
                comm.Write(tag, value)
                got = comm.Read(tag)
                if got.Status != "Success" or got.Value != value:
                    raise RuntimeError(f"{tag}: wrote {value}, read {got.Value} ({got.Status})")
    finally:
        server.terminate()  # always stop the simulator, even if the check failed
        server.wait()
    return f"pylogix {version('pylogix')} wrote and read back DINT and BOOL tags on cpppo {version('cpppo')}"


def check_yaml_configs():
    import yaml

    kits = yaml.safe_load((REPO / "configs" / "kits.yaml").read_text())
    return f"pyyaml read kits.yaml: KIT-A = {kits['kits']['KIT-A']}"


def check_pandas_manifest():
    import pandas as pd

    df = pd.read_csv(REPO / "data" / "manifest.csv")
    return f"pandas {pd.__version__} read manifest.csv: {len(df.columns)} columns, {len(df)} rows"


CHECKS = [
    ("1. PyTorch + CUDA", check_torch_cuda),
    ("2. OpenCV", check_opencv),
    ("3. YOLO26n predict", check_yolo26n),
    ("4. OpenVINO", check_openvino),
    ("5. Transformers", check_transformers),
    ("6. FastAPI", check_fastapi),
    ("7. PLC round trip", check_plc_round_trip),
    ("8. YAML configs", check_yaml_configs),
    ("9. Manifest CSV", check_pandas_manifest),
]


def main():
    failures = 0
    for name, check in CHECKS:
        try:
            print(f"PASS  {name}: {check()}")
        except Exception as e:  # one broken library shouldn't hide the others
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print()
    print("All checks passed." if failures == 0 else f"{failures} check(s) failed.")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
