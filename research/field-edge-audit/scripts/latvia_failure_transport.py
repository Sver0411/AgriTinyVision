"""Change the transport size while keeping the trained expert and its centre crop fixed."""

import argparse
import hashlib
import io
from pathlib import Path
import time
import zipfile

import numpy as np
from PIL import Image, ImageOps

from directional_policy import probabilities_from_logits
from edge_adaptation_wire import normalized
from evaluate_upload import rank_selection
from field_specialist import MODEL_DIR, checked_model, tensor_from_image
from latvia_failure import ROOT, RESULTS, SETUP, MARGINS, auc, ba, describe, save
from server_backend import jpeg_transport

SCORES = RESULTS / "latvia_failure_transport_scores.npz"
METHODS = ("jpeg224_center", "jpeg448_center", "original_center", "jpeg448_full")


def extract(archive_path):
    from latvia_blind_features import session

    cfg = checked_model()
    model = session(str(MODEL_DIR / "model.onnx"), 4)
    with np.load(ROOT / "data/features/dino_plantdoc_balanced_head.npz") as z:
        w = z["coef"][0]
        bias = z["intercept"].item()
    with np.load(ROOT / "data/features/latvia.npz") as z:
        paths = z["source_path"]
        y = z["y"]
        sizes = [z["jpeg_bytes"], z["jpeg448_bytes"]]
        sums, source_sums = z["jpeg448_sha256"], z["source_sha256"]
        centre224 = z["dino_balanced_margin"]
        full448 = normalized(z["multiview"][:, :384]) @ w + bias
    centres = []
    original_sizes = []
    start = time.monotonic()
    with zipfile.ZipFile(archive_path) as archive:
        for first in range(0, len(y), 4):
            tensors = []
            for i in range(first, min(first + 4, len(y))):
                raw = archive.read(paths[i])
                if hashlib.sha256(raw).hexdigest() != source_sums[i]:
                    raise ValueError(f"Unexpected original {paths[i]}")
                original_sizes.append(len(raw))
                with Image.open(io.BytesIO(raw)) as photo:
                    rgb = ImageOps.exif_transpose(photo).convert("RGB")
                jpeg = jpeg_transport(rgb, 448)
                if (
                    len(jpeg) != sizes[1][i]
                    or hashlib.sha256(jpeg).hexdigest() != sums[i]
                ):
                    raise ValueError(
                        f"JPEG differs from the recorded 448px transmission {paths[i]}"
                    )
                with Image.open(io.BytesIO(jpeg)) as photo:
                    tensors.append(tensor_from_image(photo.convert("RGB"), cfg))
                tensors.append(tensor_from_image(rgb, cfg))
            cls = model.run(
                ["last_hidden_state"], {"pixel_values": np.ascontiguousarray(tensors)}
            )[0][:, 0, :]
            centres.extend((normalized(cls) @ w + bias).reshape(-1, 2).tolist())
            if (first + 4) % 120 == 0:
                print(
                    f"{first + 4}/{len(y)}, {time.monotonic() - start:.1f}s", flush=True
                )
    c448, original = np.array(centres).T
    scores = probabilities_from_logits(np.array([centre224, c448, original, full448]))
    np.savez_compressed(
        SCORES,
        source_path=paths,
        y=y,
        scores=scores,
        methods=METHODS,
        application_bytes=np.array([sizes[0], sizes[1], original_sizes, sizes[1]]) + 20,
    )


def analyze():
    with np.load(SCORES) as z:
        paths, y, scores, costs = (
            z["source_path"],
            z["y"],
            z["scores"],
            z["application_bytes"],
        )
        assert list(z["methods"]) == list(METHODS)
    with np.load(RESULTS / "latvia_decisions.npz") as z:
        assert np.array_equal(z["source_path"], paths) and np.array_equal(
            scores[0], z["remote_probability"]
        )
    with np.load(MARGINS) as z:
        assert np.array_equal(z["latvia_paths"], paths)
        margins = z["latvia_scores"]
    cap = int(costs[0].sum()) // 4
    rows = []
    for seed, margin in enumerate(margins):
        order = np.lexsort((paths, abs(margin)))
        fixed = rank_selection(order, costs[0], cap - SETUP)
        for i, method in enumerate(METHODS):
            cloud = scores[i] >= 0.5
            for mode, q in [
                ("full_expert", np.ones(len(y), bool)),
                ("same_selected_images", fixed),
                ("same_byte_ceiling", rank_selection(order, costs[i], cap - SETUP)),
            ]:
                pred = np.where(q, cloud, margin >= 0)
                rows.append(
                    dict(
                        seed=seed,
                        method=method,
                        comparison=mode,
                        n=len(y),
                        positives=int(y.sum()),
                        fp=int(np.count_nonzero(pred[y == 0])),
                        fn=int(np.count_nonzero(~pred[y == 1])),
                        sensitivity=float(pred[y == 1].mean()),
                        specificity=float(1 - pred[y == 0].mean()),
                        balanced_accuracy=ba(y, pred),
                        expert_auc=auc(y, scores[i]),
                        queries=int(q.sum()),
                        total_application_bytes=int(costs[i, q].sum())
                        + (0 if mode == "full_expert" else SETUP),
                    )
                )
    save("latvia_failure_transport_runs.csv", rows)
    save("latvia_failure_transport.csv", describe(rows, ("method", "comparison")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extract",
        type=Path,
        help="Rebuild scores using the Latvian ZIP and frozen ONNX weights.",
    )
    args = parser.parse_args()
    if args.extract:
        extract(args.extract)
    analyze()
