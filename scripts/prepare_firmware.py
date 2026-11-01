"""Generate an actual train-only normalized NHWC fixture and copy a verified .espdl."""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from agritinyvision.data.dataset import open_rgb
from agritinyvision.data.transforms import preprocessing
from agritinyvision.quality import abstention_bounds, decide, load_quality, quality_gate
from agritinyvision.train import load_checkpoint
from agritinyvision.utils import read_json, sha256, write_json


def main():
    torch.set_num_threads(4)
    run = Path("runs/baseline160")
    model, cp = load_checkpoint(run / "best.pt")
    c = cp["config"]
    report = read_json(run / "tinyvision.espdl.quantization.json")
    if not report["status"].startswith("SUCCESS") or report["sha256"] != sha256(run / "tinyvision.espdl"):
        raise ValueError("Actual successful ESP32-S3 conversion is required")
    # Use a train calibration source, never disclose or tune based on a test sample.
    path = Path(c["dataset_root"]) / report["calibration_paths"][0]
    im = open_rgb(path)
    x = preprocessing(c["input_size"])(im).unsqueeze(0)
    with torch.inference_mode():
        p = model(x).softmax(1)[0, 1].item()
    destination = Path("firmware_example/main")
    destination.mkdir(parents=True, exist_ok=True)
    x.permute(0, 2, 3, 1).contiguous().numpy().astype("<f4").tofile(destination / "fixture_input.bin")
    q = load_quality()
    gate = quality_gate(im, q)
    lower, upper = abstention_bounds(cp["threshold"], q["abstention_margin"])
    text = f"#pragma once\n#define FIXTURE_WIDTH {c['input_size']}\n#define FIXTURE_HEIGHT {c['input_size']}\n#define FIXTURE_THRESHOLD {cp['threshold']:.9g}f\n#define FIXTURE_ABSTENTION_MARGIN {q['abstention_margin']:.9g}f\n#define FIXTURE_NORMAL_UPPER_BOUND {lower:.9g}f\n#define FIXTURE_SUSPECT_LOWER_BOUND {upper:.9g}f\n#define FIXTURE_FP32_PROBABILITY {p:.9g}f\n#define FIXTURE_QUALITY_PASSED {int(gate['passed'])}\n"
    (destination / "fixture_config.h").write_text(text)
    shutil.copy2(run / "tinyvision.espdl", destination / "tinyvision.espdl")
    write_json(
        "reports/firmware_fixture_final_patch.json",
        {
            "source_split": "train",
            "source_path": str(path),
            "layout": "NHWC float32 normalized little endian",
            "fp32_probability": p,
            "quality": gate,
            "state_logic": decide(p, cp["threshold"], gate, q["abstention_margin"]),
            "model_sha256": sha256(destination / "tinyvision.espdl"),
            "sha256": sha256(destination / "fixture_input.bin"),
            "hardware_status": "PENDING HARDWARE VALIDATION",
        },
    )
    print("Prepared verified .espdl + train-only golden input; hardware validation remains pending")


if __name__ == "__main__":
    main()
