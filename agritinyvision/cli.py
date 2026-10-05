import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser(description="AgriTinyVision: RGB image → NORMAL / SUSPECT / UNKNOWN")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("doctor")
    d.add_argument("--output", default="reports/environment.json")
    a = sub.add_parser("audit")
    a.add_argument("--manifest", default="data/manifests/plantvillage_split.json")
    a.add_argument("--root", default="data/raw/plantvillage_color")
    a.add_argument("--output", default="reports/dataset_audit")
    s = sub.add_parser("prepare-data")
    s.add_argument("--manifest", default="data/manifests/plantvillage_raw.json")
    s.add_argument("--root", default="data/raw/plantvillage_color")
    s.add_argument("--output", default="data/manifests/plantvillage_split.json")
    s.add_argument("--seed", type=int, default=41)
    t = sub.add_parser("train")
    t.add_argument("--config", required=True)
    t.add_argument("--allow-cpu", action="store_true")
    t.add_argument("--resume")
    t.add_argument("--output")
    e = sub.add_parser("evaluate")
    e.add_argument("--checkpoint", required=True)
    e.add_argument("--model", help="Optional ONNX FP32/INT8 artifact for final host evaluation")
    e.add_argument("--output")
    e.add_argument("--allow-cpu", action="store_true")
    ex = sub.add_parser("export")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--format", choices=["onnx"], default="onnx")
    ex.add_argument("--output")
    pre = sub.add_parser("preflight")
    pre.add_argument("--config", default="configs/train_mobilenetv2_160.yaml")
    q = sub.add_parser("quantize")
    q.add_argument("--model", required=True)
    q.add_argument("--checkpoint", required=True)
    q.add_argument("--backend", choices=["espdl", "onnx"], default="espdl")
    q.add_argument("--output")
    q.add_argument("--allow-cpu", action="store_true")
    b = sub.add_parser("benchmark")
    b.add_argument("--model", required=True)
    b.add_argument("--iterations", type=int, default=100)
    b.add_argument("--output")
    r = sub.add_parser("release")
    r.add_argument("--checkpoint", required=True)
    r.add_argument("--onnx", required=True)
    r.add_argument("--metrics", required=True)
    r.add_argument("--espdl")
    r.add_argument("--version", default="0.1.0")
    r.add_argument("--output")
    i = sub.add_parser("infer")
    i.add_argument("--checkpoint", required=True)
    i.add_argument("--image", required=True)
    i.add_argument("--quality", default="configs/quality.yaml")
    i.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args()
    try:
        if args.command == "doctor":
            from .doctor import main as run

            run(args.output)
        elif args.command == "audit":
            from .data.audit import audit

            report, _ = audit(args.manifest, args.root, args.output)
            print(json.dumps(report, indent=2))
            if not report["passed"]:
                raise ValueError("Dataset audit failed")
        elif args.command == "prepare-data":
            from .data.split import prepare

            print(prepare(args.manifest, args.root, args.output, args.seed))
        elif args.command == "train":
            from .train import train

            print(train(args.config, args.allow_cpu, args.resume, args.output))
        elif args.command == "evaluate":
            from .evaluate import evaluate

            evaluate(args.checkpoint, args.model, args.allow_cpu, args.output)
        elif args.command == "export":
            from .export_onnx import export

            print(export(args.checkpoint, args.output))
        elif args.command == "preflight":
            from .export_onnx import preflight

            print(preflight(args.config))
        elif args.command == "quantize":
            from .quantize import quantize

            print(quantize(args.model, args.checkpoint, args.backend, args.output, args.allow_cpu))
        elif args.command == "benchmark":
            from .benchmark import benchmark

            benchmark(args.model, args.iterations, args.output)
        elif args.command == "release":
            from .release import release

            print(release(args.checkpoint, args.onnx, args.metrics, args.espdl, args.version, args.output))
        elif args.command == "infer":
            import torch

            from .data.dataset import open_rgb
            from .data.transforms import preprocessing
            from .doctor import device
            from .quality import decide, load_quality, quality_gate
            from .train import load_checkpoint

            q = load_quality(args.quality)
            gate = quality_gate(args.image, q)
            if not gate["passed"]:
                answer = {"state": "UNKNOWN", "reason": "quality_gate", "quality": gate}
            else:
                dev = device(args.allow_cpu)
                model, cp = load_checkpoint(args.checkpoint, dev)
                x = preprocessing(cp["config"]["input_size"])(open_rgb(args.image)).unsqueeze(0).to(dev)
                with torch.inference_mode():
                    score = model(x).softmax(1)[0, 1].item()
                answer = decide(score, cp["threshold"], gate, q["min_confidence"])
            print(json.dumps(answer, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError, ImportError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
