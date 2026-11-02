"""Evaluate, compare, label or partition saved field batches with the frozen policy."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agritinyvision.field_validation.annotations import annotation_csv, import_annotations
from agritinyvision.field_validation.parity import compare_batch
from agritinyvision.field_validation.reporting import generate_batch_report

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("report", "parity", "annotation-template", "import-annotations"):
        item = commands.add_parser(name)
        item.add_argument("--batch", required=True, type=Path)
        if name == "report":
            item.add_argument("--labels", choices=("manual", "source"), default="manual")
        elif name == "parity":
            item.add_argument("--bundle", default="releases/v0.1.0")
            item.add_argument("--host-model")
            item.add_argument("--host-report")
        else:
            item.add_argument("--file", required=True, type=Path)
    item = commands.add_parser("split")
    item.add_argument("--batches", nargs="+", required=True)
    item.add_argument("--output", required=True)
    item.add_argument("--seed", type=int, default=41)
    a = p.parse_args()
    if a.command == "report":
        result = generate_batch_report(a.batch, a.labels)
        print(json.dumps(result["dataset"], indent=2))
    elif a.command == "parity":
        if bool(a.host_model) != bool(a.host_report):
            p.error("host-model and host-report are required together")
        print(json.dumps(compare_batch(a.batch, a.bundle, a.host_model, a.host_report)["summary"], indent=2))
    elif a.command == "annotation-template":
        a.file.write_text(annotation_csv(a.batch), encoding="utf-8-sig")
    elif a.command == "import-annotations":
        print(import_annotations(a.batch, a.file))
    else:
        from agritinyvision.field_validation.split import split_batches

        print(json.dumps(split_batches(a.batches, a.output, a.seed)["counts"], indent=2))
