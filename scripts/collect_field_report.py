"""Collect aggregate field results and source references, excluding original photographs."""

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.field_validation.capture import read_json, read_samples
from agritinyvision.field_validation.reporting import REPORT_FILES
from agritinyvision.pc_camera.metrics import write_json
from agritinyvision.utils import sha256


def collect(batch, output):
    batch, output = Path(batch), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = read_json(batch / "batch.json")
    for name in sorted(REPORT_FILES | {"SOURCE_AUDIT.json", "FIELD_SPLIT.json"}):
        source = batch / "reports" / name
        if source.exists():
            shutil.copy2(source, output / name)
    sources = []
    for sample in read_samples(batch):
        record = read_json(batch / sample["metadata"])
        context = record["context"]
        sources.append({"sample_id": sample["sample_id"], "rgb_sha256": record["rgb_sha256"],
                        **{k: context.get(k) for k in ("source_url", "source_path", "source_label", "label",
                            "source_split", "sha256", "git_blob_sha1", "crop_type", "excluded_from_evaluation")}})
    write_json(output / "SOURCE_MANIFEST.json", {"batch_id": meta["batch_id"], "source": meta["context"],
               "batch_sha256": sha256(batch / "batch.json"), "samples": sources})
    (output / "README.md").write_text(
        "# Public agricultural image evaluation\n\n"
        "[Experiment summary](EXPERIMENT_SUMMARY.md) · [Inference parity](INFERENCE_PARITY.md)\n\n"
        "Images: PlantDoc, Singh et al. (2020), https://github.com/pratikkayal/PlantDoc-Dataset, "
        "revision " + str(meta["context"].get("revision")) + ". Repository license: CC-BY-4.0.\n\n"
        "This folder contains aggregate measurements, diagrams and source references. Original photos and "
        "lossless model inputs remain in the ignored local batch. Labels come from the dataset authors, "
        "not project predictions. Human quality and source capture-session attributes were not supplied.\n",
        encoding="utf-8")
    print(json.dumps({"batch_id": meta["batch_id"], "aggregates": str(output), "images_copied": 0}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", required=True)
    p.add_argument("--output", default="reports/field_validation")
    a = p.parse_args()
    collect(a.batch, a.output)
