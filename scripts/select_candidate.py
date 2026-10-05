"""Freeze candidate from validation and compute before looking at final test metrics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.benchmark import compute
from agritinyvision.threshold import selection_key
from agritinyvision.train import load_checkpoint
from agritinyvision.utils import now, read_json, sha256, write_json


def main():
    out = Path("reports/candidate_selection.json")
    if out.exists():
        raise RuntimeError("Selection already frozen. Do not reselect after test evaluation.")
    experiments = []
    for name in ("baseline160", "baseline224"):
        run = Path("runs") / name
        if (run / "test_metrics.json").exists():
            raise RuntimeError("Test results already exist; validation-only selection must happen first")
        result = read_json(run / "metrics.json")
        model, cp = load_checkpoint(run / "best.pt")
        experiments.append(
            {
                "run": str(run),
                "input_size": cp["config"]["input_size"],
                "validation": result["best_validation"],
                "parameters": result["parameters"],
                "fp32_parameter_bytes": result["fp32_parameter_bytes"],
                "conv_linear_macs": compute(model, cp["config"]["input_size"]),
                "training_seconds": result["training_seconds"],
                "peak_vram_allocated_bytes": result["peak_vram_allocated_bytes"],
                "epochs_completed": result["epochs_completed"],
                "checkpoint_sha256": sha256(run / "best.pt"),
            }
        )
    chosen = max(experiments, key=lambda r: (*selection_key(r["validation"]), -r["conv_linear_macs"]))
    write_json(
        out,
        {
            "created_at": now(),
            "selected_run": chosen["run"],
            "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
            "criterion": "Validation recall/FPR constraints then F1; tied validation scores resolved by lower MACs. No test metrics used.",
            "experiments": experiments,
            "hardware_status": "PENDING HARDWARE VALIDATION",
        },
    )
    print("Frozen candidate:", chosen["run"])


if __name__ == "__main__":
    main()
