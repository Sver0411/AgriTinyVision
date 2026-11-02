"""Compare identical RGB inputs across frozen ONNX, PPQ and ESP32 backends."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.pc_camera.serial_client import SerialClient
from agritinyvision.quantization_analysis import analyze

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--bundle", default="releases/v0.1.0")
    p.add_argument("--output", required=True, help="new output directory")
    p.add_argument("--port", help="physical ESP32 port; omit for desktop-only analysis")
    p.add_argument("--host-model", help="frozen .espdl with adjacent trusted .native")
    p.add_argument("--host-report", help="hash-bound .quantization.json for that native graph")
    args = p.parse_args()
    if bool(args.host_model) != bool(args.host_report):
        p.error("--host-model and --host-report are required together")
    client = SerialClient(args.port, bundle=args.bundle) if args.port else None
    print(json.dumps(analyze(args.manifest, args.bundle, args.output, client,
                             args.host_model, args.host_report), indent=2))
