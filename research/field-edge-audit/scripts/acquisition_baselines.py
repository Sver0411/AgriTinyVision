"""Published review scores and a label-free shift trigger for the acquisition study."""
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import struct
import zipfile

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold, StratifiedKFold

from edge_adaptation_wire import I8, normalized, predict_from_head
from published_defer import binary_head, train_weights, mao_examples
from published_cascade import HEADER, SHAPES, forward, probability_features, train_routes, unpack
from published_gatekeeper import decode_head, optimize

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/acquisition_source_reference.csv"
METHODS = ("edgeboost_rank", "edgeboost_fixed", "ova", "compare_confidence", "mao", "diff01", "gatekeeper", "ce")
EDGEBOOST_CONFIG = struct.Struct("<ff")


def download_byte_count(method, width):
    """Quote the fixed wire layout without access to any label, feature or expert answer."""
    if method not in (*METHODS, "diffprob") or not isinstance(width, (int, np.integer)) or not 0 < width < 65536:
        raise ValueError("Unsupported packet type or activation width")
    head = I8.size + int(width)
    if method in ("ova", "compare_confidence", "mao"):
        return 2 * head
    if method in ("diff01", "diffprob"):
        return head + HEADER.size + 4 * sum(int(np.prod(shape)) for shape in SHAPES)
    if method == "edgeboost_fixed":
        return head + EDGEBOOST_CONFIG.size
    return head


def codes(matrix):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.dtype != np.int8:
        raise ValueError("Expected a two-dimensional signed-byte activation matrix")
    return matrix


def edge_fit(q, y):
    return binary_head(normalized(q), y, train_weights(y, "balanced"))


def held_out_logits(q, y, seed):
    if len(y) < 2:
        raise ValueError("Fixed-threshold calibration needs at least two paid labels")
    minority = np.bincount(y, minlength=2).min()
    if minority >= 2:
        split = StratifiedKFold(min(4, int(minority)), shuffle=True, random_state=seed)
        folds = split.split(q, y)
    else:
        folds = KFold(min(4, len(q)), shuffle=True, random_state=seed).split(q)
    result = np.empty(len(y))
    for train, hold in folds:
        result[hold] = predict_from_head("local_int8", edge_fit(q[train], y[train]), q[hold])
    return result


@dataclass(frozen=True)
class ReviewModel:
    method: str
    classifier: bytes
    correctness: bytes = b""
    temperature: float = 1.0
    oof_confidence: np.ndarray | None = None

    @property
    def download_bytes(self):
        config = EDGEBOOST_CONFIG.size if self.method == "edgeboost_fixed" else 0
        return len(self.classifier) + len(self.correctness) + config

    def score(self, q):
        """Higher review priority means the current photo is more eligible to send."""
        q = codes(q)
        z = predict_from_head("local_int8", self.classifier, q)
        if self.method in ("edgeboost_rank", "gatekeeper", "ce"):
            # Analytic rank before tanh might numerically saturate. Every T>0 preserves it.
            return z, -np.abs(z)
        if self.method == "edgeboost_fixed":
            return z, -np.tanh(np.abs(z) / (2 * self.temperature))
        if self.method in ("diff01", "diffprob"):
            _, _, priorities = forward(probability_features(expit(z)), unpack(self.correctness))
            return z, priorities
        expert_logit = predict_from_head("local_int8", self.correctness, q)
        if self.method == "mao":
            return z, expert_logit
        if self.method == "ova":
            return z, expert_logit - np.abs(z)
        return z, expit(expert_logit) - expit(np.abs(z))

    def fixed_cutoff(self, calibration_lengths, desired_fraction):
        """Use priority > returned threshold. Only paid calibration lengths enter."""
        if self.method != "edgeboost_fixed" or self.oof_confidence is None:
            raise ValueError("Fixed cutoff is specific to the calibrated EdgeBoost gate")
        lengths = np.asarray(calibration_lengths, dtype=np.int64)
        if lengths.shape != self.oof_confidence.shape or not np.all(lengths > 0):
            raise ValueError("Calibration packet lengths do not match the paid labels")
        if not np.isfinite(desired_fraction) or not 0 <= desired_fraction <= 1:
            raise ValueError("Desired byte fraction must lie in [0,1]")
        confidence = self.oof_confidence
        thresholds = np.unique(np.r_[np.float32(0), np.nextafter(confidence.astype(np.float32),
                                                                  np.float32(np.inf))])
        used = np.array([lengths[confidence < t].sum() for t in thresholds])
        threshold = thresholds[np.argmin(np.abs(used / lengths.sum() - desired_fraction))]
        return -float(threshold)


def fit_router(method, qfit, yfit, cloudfit, seed=0):
    """Fit only on purchased annotated photographs. No evaluation scores or labels enter."""
    if method not in METHODS:
        raise ValueError(f"Unknown published method {method}")
    q = codes(qfit)
    y = np.asarray(yfit)
    if len(q) == 0 or y.shape != (len(q),) or not np.all(np.isin(y, [0, 1])):
        raise ValueError("One paid binary reference is required per training record")
    y = y.astype(np.int32)
    head = edge_fit(q, y)
    if method == "edgeboost_rank":
        return ReviewModel(method, head)
    if method == "edgeboost_fixed":
        oof = held_out_logits(q, y, seed)
        def nll(log_t):
            scores = oof / np.exp(log_t)
            return float(np.mean(np.logaddexp(0, scores) - y * scores))
        result = minimize_scalar(nll, bounds=(-6.0, 6.0), method="bounded")
        if not result.success:
            raise ValueError("Unable to calibrate EdgeBoost temperature")
        temp = float(np.float32(np.exp(result.x)))
        return ReviewModel(method, head, temperature=temp, oof_confidence=np.tanh(np.abs(oof) / (2 * temp)))
    if method in ("gatekeeper", "ce"):
        design = np.column_stack((normalized(q), np.ones(len(q))))
        fitted = optimize(design, y, decode_head(head), .5 if method == "gatekeeper" else None,
                          100, method == "ce")
        return ReviewModel(method, fitted)
    observed = np.asarray(cloudfit)
    if observed.shape != y.shape or not np.all(np.isin(observed, [0, 1])):
        raise ValueError("Learned deferral requires cloud outcomes on the purchased photographs")
    if method == "mao":
        oof = held_out_logits(q, y, 20261127 + seed)
        design, targets, weights = mao_examples(normalized(q), y, oof >= 0, observed, train_weights(y, "balanced"))
        return ReviewModel(method, head, binary_head(design, targets, weights))
    if method == "diff01":
        oof = held_out_logits(q, y, 3071 + seed)
        target = (observed == y).astype(float) - ((oof >= 0) == y)
        router = train_routes(probability_features(expit(oof)), target, y, seed, balanced=False, steps=20)[20]
        return ReviewModel(method, head, router)
    correctness = binary_head(normalized(q), np.equal(observed, y).astype(int), train_weights(y, "balanced"))
    return ReviewModel(method, head, correctness)


def fit_probability_router(qfit, yfit, cloud_probability, seed=0):
    """Diff-Prob uses a continuous cloud probability only on already purchased photos."""
    q = codes(qfit)
    y = np.asarray(yfit, dtype=np.int32)
    p = np.asarray(cloud_probability, dtype=np.float64)
    if y.shape != (len(q),) or p.shape != y.shape or not np.all((p >= 0) & (p <= 1)):
        raise ValueError("Expected one calibration label and cloud probability for each purchased photo")
    oof = held_out_logits(q, y, 3071 + seed)
    local = expit(oof)
    target = (2.0*y-1)*(p-local)
    router = train_routes(probability_features(local), target, y, seed, balanced=False, steps=20)[20]
    return ReviewModel("diffprob", edge_fit(q,y), router)


def bbsd_soft(reference_prob, prefix_prob):
    """BBSD-soft with the authors' two-sided KS and Bonferroni across both outputs."""
    src = np.asarray(reference_prob, dtype=np.float64)
    tgt = np.asarray(prefix_prob, dtype=np.float64)
    for p in (src, tgt):
        if p.ndim != 1 or len(p) == 0 or not np.all((p >= 0) & (p <= 1)):
            raise ValueError("Expected finite class-1 probabilities in [0,1]")
    calls = [ks_2samp(src, tgt), ks_2samp(1-src, 1-tgt)]
    p = min(1.0, 2.0 * min(float(r.pvalue) for r in calls))
    return p < .05, p, max(float(r.statistic) for r in calls)


def reference_scores(backbone):
    """Factory-installed scores on the same 121 clean PlantDoc holdouts for each camera."""
    name = backbone.lower()
    if name not in ("v2", "v4", "v2grid"):
        raise ValueError("Backbone must be v2, v4, or v2grid")
    with REFERENCE.open(newline="") as f:
        values = np.asarray([float(row[name + "_prob"]) for row in csv.DictReader(f)], np.float32)
    if values.shape != (121,):
        raise ValueError("Incomplete factory reference")
    return values


def build_reference(quantization_config):
    from PIL import Image
    from esp_quantized_replay import graph, prepare
    from second_backbone_features import checked_model, preprocess
    from edge_adaptation_wire import quantized

    with (ROOT / "results/plantdoc_logits.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["keep_dedup"] == r["training_crop"] == "1"]
    assert len(rows) == len({r["source_path"] for r in rows}) == 121
    session, hidden, _, _, cfg = checked_model(2)
    grid = graph(quantization_config)
    grid_meta = json.loads((ROOT / "data/raw/tinyvision-metadata.json").read_text())["preprocessing"]
    with zipfile.ZipFile(ROOT / "results/second_backbone_heads.zip") as z:
        head = z.read("source.bin")
    results = []
    for start in range(0, len(rows), 12):
        group = rows[start:start+12]
        arrays, grid_probs = [], []
        for row in group:
            path = row["source_path"]
            raw = (ROOT / "data/raw/plantdoc" / path).read_bytes()
            _, _, _, tensor = preprocess((path, 0, row["file_sha256"], raw), cfg)
            arrays.append(tensor)
            with Image.open(ROOT / "data/raw/plantdoc" / path) as image:
                quant_input, rgb_digest = prepare(image, grid_meta)
            if rgb_digest != row["rgb160_sha256"]:
                raise ValueError("Grid replay RGB differs from the archived FP32 input")
            logits, = grid.run([grid.get_outputs()[0].name], {"image": quant_input})
            grid_probs.append(float(expit(np.diff(logits.ravel()).item())))
        features, = session.run([hidden], {session.get_inputs()[0].name: np.stack(arrays)})
        z = predict_from_head("local_int8", head, quantized(features)[0])
        for row, prob, grid_prob in zip(group, expit(z), grid_probs, strict=True):
            results.append(dict(source_path=row["source_path"], source_sha256=row["file_sha256"],
                                v2_prob=float(np.float32(row["p_suspect"])), v4_prob=float(np.float32(prob)),
                                v2grid_prob=float(np.float32(grid_prob))))
    with REFERENCE.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantization-config", type=Path, default=ROOT / "data/raw/esp-ppq/tinyvision.json")
    build_reference(parser.parse_args().quantization_config)
    for name in ("v2", "v4", "v2grid"):
        print(name, len(reference_scores(name)), "clean source holdout scores")
