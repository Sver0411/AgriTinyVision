"""Run from the checkout without importing the training/ONNX pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.pc_camera.cli import main

if __name__ == "__main__":
    main()
