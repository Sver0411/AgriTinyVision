"""Summarize measured PC, transport and device timings without changing the model."""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from agritinyvision.pc_camera.metrics import generate_report, write_json


def performance_report(run):
    run = Path(run)
    summary = generate_report(run)
    with (run / "frame_results.csv").open(encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r["error"]]
    residuals = [float(r["serial_roundtrip_ms"]) - float(r["total_us"]) / 1000 for r in rows
                 if r.get("serial_roundtrip_ms") and r.get("total_us")]
    result = {"frames": len(rows), "memory": summary["memory"], "latency_trend": summary["latency_trend"],
              "serial_minus_device_total_ms": {"count": len(residuals), "p50": float(np.median(residuals)),
                                               "p95": float(np.percentile(residuals, 95))} if residuals else "PENDING",
              "interpretation": "Residual includes USB, serialization and PC receive overhead; it is not pure wire time.",
              "latency": json.loads((run / "latency_report.json").read_text(encoding="utf-8")),
              "quantile_policy": "Compute residuals per frame; do not subtract or add marginal percentiles."}
    write_json(run / "performance_analysis.json", result)
    (run / "PERFORMANCE.md").write_text("# Performance analysis\n\n```json\n" + json.dumps(result, indent=2) +
        "\n```\n\nPC capture and preprocessing are wall-clock milliseconds; device fields are microseconds. "
        "The end-to-end clock starts at capture (or fixed-image processing) and ends at parsed result. "
        "Old v0.2 records measured send-to-result and retain that historical definition.\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    print(json.dumps(performance_report(p.parse_args().run), indent=2))
