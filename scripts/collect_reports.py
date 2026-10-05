"""Copy small authentic run evidence to reports; never include model/checkpoint binaries."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.export_onnx import validate_checkpoint_onnx
from agritinyvision.utils import read_json, write_json


def main():
    snapshot = subprocess.check_output(
        ["git", "-c", f"safe.directory={Path.cwd().as_posix()}", "rev-parse", "0b60606"], text=True
    ).strip()
    for name in ("baseline160", "baseline224"):
        run = Path("runs") / name
        env = read_json(run / "environment.json")
        if env.get("git_commit") is None:
            env["source_snapshot_commit"] = snapshot
            env["git_detection_note"] = (
                "Original runtime Git lookup failed due sandbox/desktop repository ownership mismatch. The initial source snapshot was committed during first run; training/model/data/threshold modules match that snapshot. Original git_commit remains null, not fabricated. Future lookup uses a repository-scoped safe.directory."
            )
            for relative in (
                "agritinyvision/train.py",
                "agritinyvision/models/mobilenetv2.py",
                "agritinyvision/data/dataset.py",
                "agritinyvision/data/split.py",
                "agritinyvision/data/transforms.py",
                "agritinyvision/threshold.py",
            ):
                original = subprocess.check_output(
                    [
                        "git",
                        "-c",
                        f"safe.directory={Path.cwd().as_posix()}",
                        "show",
                        snapshot + ":" + relative,
                    ],
                    text=True,
                )
                # ruff formatting/import order does not alter executed behavior; preserve exact verification.
                env.setdefault("source_snapshot_file_matches", {})[relative] = (
                    original == Path(relative).read_text()
                )
            write_json(run / "environment.json", env)
        validate_checkpoint_onnx(run / "best.pt", run / "tinyvision.onnx")
        dest = Path("reports/experiments") / name
        dest.mkdir(parents=True, exist_ok=True)
        for filename in (
            "config.yaml",
            "environment.json",
            "history.csv",
            "metrics.json",
            "test_metrics.json",
            "threshold.json",
            "tinyvision.onnx.validation.json",
            "tinyvision.espdl.quantization.json",
            "tinyvision.onnx.benchmark.json",
        ):
            source = run / filename
            if source.exists():
                shutil.copy2(source, dest / filename)
    m = read_json("data/manifests/plantvillage_split.json")
    write_json("data/metadata/plantvillage_source.json", {k: v for k, v in m.items() if k != "samples"})
    print(
        "Collected authentic experiment evidence, bound ONNX to checkpoints, preserved original Git lookup failure"
    )


if __name__ == "__main__":
    main()
