"""Export a non-restricted numeric C99 fixture (1 real BASA EQ01 head + 16 quantized rows) for ESP32-S3 benchmarking."""

from pathlib import Path
import numpy as np

from basa_controller import build_basa_plan, load_factory_source_heads
from edge_adaptation_wire import I8, predict_from_head, quantized
from purchase_cohorts import load
from purchase_experiment import FULL_RATE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "esp32s3_bench/main/basa_fixture.h"


def main() -> None:
    d = load("cornell", "v2grid")
    w0 = load_factory_source_heads(ROOT)["v2grid"]
    n = len(d["test_y"])
    cap_bytes = n * (FULL_RATE * 25 // 100)
    plan = build_basa_plan(d, seed=0, cap_bytes=cap_bytes, label_cap=32, grant_mode="earned", w0=w0)
    assert not plan.fallback_frozen and len(plan.head_blob) == 14 + 1280

    q_test = quantized(d["test_x"][:16])[0]
    w_i8 = np.frombuffer(plan.head_blob, dtype=np.int8, offset=I8.size).astype(np.int32)
    scores = predict_from_head("local_int8", plan.head_blob, q_test)
    dots = (q_test.astype(np.int32) @ w_i8).tolist()
    sqs = np.sum(q_test.astype(np.int32) ** 2, axis=1).tolist()
    preds = [int(s >= 0.0) for s in scores]

    lines = [
        "/* Auto-generated numeric fixture for ESP32-S3 bit-exact BASA parity benchmark. */",
        "#ifndef BASA_FIXTURE_H",
        "#define BASA_FIXTURE_H",
        "#include <stdint.h>",
        "#define BASA_DIM 1280",
        "#define BASA_FIXTURE_ROWS 16",
        "",
        "static const uint8_t BASA_HEAD_BLOB[1294] = {",
    ]
    blob_bytes = list(plan.head_blob)
    for i in range(0, len(blob_bytes), 16):
        lines.append("    " + ", ".join(str(b) for b in blob_bytes[i : i + 16]) + ",")
    lines.append("};")
    lines.append("")
    lines.append("static const int8_t BASA_FIXTURE_X[BASA_FIXTURE_ROWS][BASA_DIM] = {")
    for r in range(16):
        vals = q_test[r].tolist()
        lines.append("    {")
        for i in range(0, len(vals), 20):
            lines.append("        " + ", ".join(str(v) for v in vals[i : i + 20]) + ",")
        lines.append("    },")
    lines.append("};")
    lines.append(f"static const int32_t BASA_FIXTURE_DOT[BASA_FIXTURE_ROWS] = {{{', '.join(map(str, dots))}}};")
    lines.append(f"static const int32_t BASA_FIXTURE_SQ[BASA_FIXTURE_ROWS] = {{{', '.join(map(str, sqs))}}};")
    lines.append(
        "static const double BASA_FIXTURE_SCORE[BASA_FIXTURE_ROWS] = {"
        + ", ".join(f"{float(s):.17g}" for s in scores)
        + "};"
    )
    lines.append(f"static const int BASA_FIXTURE_PRED[BASA_FIXTURE_ROWS] = {{{', '.join(map(str, preds))}}};")
    lines.append("#endif /* BASA_FIXTURE_H */")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
