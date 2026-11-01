import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.pc_camera.deployment import prepare

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify real .espdl and prepare camera firmware")
    parser.add_argument("--bundle", default="releases/v0.1.0")
    parser.add_argument("--destination", default="firmware_example/main")
    parser.add_argument("--quality", default="configs/quality.yaml")
    args = parser.parse_args()
    print(json.dumps(prepare(args.bundle, args.destination, args.quality), indent=2))
