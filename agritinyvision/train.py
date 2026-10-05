import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data.audit import audit
from .data.dataset import ManifestDataset
from .data.transforms import preprocessing, training_transform
from .doctor import device, diagnose
from .models.mobilenetv2 import build
from .threshold import selection_key, tune
from .utils import commit, load_yaml, read_json, sha256, validate_config, write_json


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


@torch.inference_mode()
def predict(model, loader, dev):
    model.eval()
    labels, scores = [], []
    for x, y in loader:
        probs = model(x.to(dev)).float().softmax(1)[:, 1]
        labels.extend(y.tolist())
        scores.extend(probs.cpu().tolist())
    return labels, scores


def load_checkpoint(path, dev="cpu"):
    # Only load checkpoints produced by this project / a trusted source.
    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    c = validate_config(checkpoint["config"])
    model = build(c, download_weights=False).to(dev)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval(), checkpoint


def train(config_path, allow_cpu=False, resume=None, output=None):
    c = validate_config(load_yaml(config_path))
    os.environ.setdefault("TORCH_HOME", str(Path(".cache/torch").resolve()))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dev = device(allow_cpu)
    torch.set_num_threads(4)
    seed = c["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = c["deterministic"]
    torch.use_deterministic_algorithms(c["deterministic"], warn_only=True)
    audit_result, _ = audit(c["manifest"], c["dataset_root"], "reports/dataset_audit")
    if not audit_result["passed"]:
        raise ValueError("Dataset audit failed; refusing training")
    manifest = read_json(c["manifest"])
    manifest_sha = sha256(c["manifest"])
    experiment = f"mobilenetv2_{c['input_size']}_{time.strftime('%Y%m%dT%H%M%S')}"
    run = Path(output or (Path(resume).parent if resume else Path("runs") / experiment))
    if run.exists() and not resume and any(run.iterdir()):
        raise ValueError(f"Refusing to overwrite experiment: {run}")
    run.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    train_data = ManifestDataset(
        c["manifest"], c["dataset_root"], "train", training_transform(c["input_size"], c["augmentation"])
    )
    val_data = ManifestDataset(c["manifest"], c["dataset_root"], "val", preprocessing(c["input_size"]))
    loader_args = dict(
        batch_size=c["batch_size"],
        num_workers=c["workers"],
        pin_memory=dev.type == "cuda",
        worker_init_fn=seed_worker,
    )
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model = build(c, download_weights=not bool(resume)).to(dev)
    if c["optimizer"] == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=c["learning_rate"], momentum=0.9, weight_decay=c["weight_decay"]
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c["epochs"])
    use_amp = c["amp"] and dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    criterion = torch.nn.CrossEntropyLoss()
    start_epoch, best_key, bad_epochs, history, elapsed = 0, (-1,), 0, [], 0.0
    best_tuned = None
    if resume:
        cp = torch.load(resume, map_location="cpu", weights_only=False)
        if cp["config"] != c or cp["manifest_sha256"] != manifest_sha:
            raise ValueError("Resume config/dataset differs from original experiment")
        model.load_state_dict(cp["state_dict"])
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        scaler.load_state_dict(cp["scaler"])
        start_epoch, best_key, bad_epochs = cp["epoch"] + 1, tuple(cp["best_key"]), cp["bad_epochs"]
        history, best_tuned, elapsed = cp["history"], cp["best_tuned"], cp["elapsed_seconds"]
        random.setstate(cp["random_state"])
        np.random.set_state(cp["numpy_state"])
        torch.set_rng_state(cp["torch_rng"].cpu())
        generator.set_state(cp["loader_rng"].cpu())
        if dev.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in cp["cuda_rng"]])
    else:
        (run / "config.yaml").write_text(yaml.safe_dump(c, sort_keys=False), encoding="utf-8")
        env = diagnose()
        env.update(
            experiment_id=run.name,
            git_commit=commit(),
            dataset_version=manifest["version"],
            dataset_hash=manifest["dataset_hash"],
            train_split_hash=manifest["train_split_hash"],
            manifest_sha256=manifest_sha,
            pretrained_weights="IMAGENET1K_V1" if c["pretrained"] else None,
            amp_enabled=use_amp,
            determinism="best effort, unsupported deterministic kernels warn",
        )
        write_json(run / "environment.json", env)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for epoch in range(start_epoch, c["epochs"]):
        model.train()
        total_loss = 0.0
        epoch_start = time.perf_counter()
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(y)
        y, p = predict(model, val_loader, dev)
        tuned = tune(y, p, c["threshold"])
        key = selection_key(tuned)
        improved = key > best_key
        if improved:
            best_key, best_tuned, bad_epochs = key, tuned, 0
        else:
            bad_epochs += 1
        scheduler.step()
        record = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_data),
            "val_recall": tuned["selected"]["suspect_recall"],
            "val_fnr": tuned["selected"]["false_negative_rate"],
            "val_f1": tuned["selected"]["f1"],
            "val_fpr": tuned["selected"]["false_positive_rate"],
            "threshold": tuned["selected"]["decision_threshold"],
            "constraints_met": tuned["constraints_met"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        duration = elapsed + time.perf_counter() - started
        cp = {
            "config": c,
            "state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "history": history,
            "best_key": best_key,
            "best_tuned": best_tuned,
            "bad_epochs": bad_epochs,
            "threshold": tuned["selected"]["decision_threshold"],
            "manifest_sha256": manifest_sha,
            "elapsed_seconds": duration,
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if dev.type == "cuda" else [],
        }
        torch.save(cp, run / "last.pt")
        if improved:
            torch.save(cp, run / "best.pt")
            write_json(run / "threshold.json", tuned)
        with open(run / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writeheader()
            writer.writerows(history)
        write_json(
            run / "metrics.json",
            {
                "status": "validation_only",
                "best_validation": best_tuned,
                "training_seconds": duration,
                "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated()
                if dev.type == "cuda"
                else None,
                "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved() if dev.type == "cuda" else None,
                "parameters": sum(p.numel() for p in model.parameters()),
                "fp32_parameter_bytes": sum(p.numel() * 4 for p in model.parameters()),
                "epochs_completed": epoch + 1,
            },
        )
        print(f"{run.name}: {record}", flush=True)
        if bad_epochs >= c["patience"]:
            print("Early stopping on validation selection criterion", flush=True)
            break
    return run
