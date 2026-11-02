"""Regenerate a local report or prepare a separate sanitized experiment for review."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.pc_camera.experiment_export import public_experiment
from agritinyvision.pc_camera.metrics import generate_report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--public-output", help="new sanitized directory; nothing is uploaded")
    a = p.parse_args()
    if a.public_output:
        print(public_experiment(a.run, a.public_output))
    else:
        print(generate_report(a.run))
