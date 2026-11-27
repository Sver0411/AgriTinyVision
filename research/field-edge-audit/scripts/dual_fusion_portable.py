"""Standard-library deployment for the preselected seed 0 / 450-label rule.

Export once with `python scripts/dual_fusion_portable.py`. On the edge, call
local_decision(rule, released_edge_margin). A bool is final. None requests
the actual 224px Q75 JPEG's source-balanced DINO margin. Complete the call
with remote_decision(rule, released_edge_margin, returned_dino_margin).
The same functions can run every record in dual_fusion_rules.json.
"""

import hashlib
import json
import math
from pathlib import Path
import struct


BASE = Path(__file__).resolve().parents[1]
FROZEN_SHA256 = "57887ed913aecdadf3064de738a6ceb5faf8bb84aeaafffe31eff8ed70032e47"


def term(rule, index, score):
    if not math.isfinite(score):
        raise ValueError("Invalid model score")
    bounded = min(max(score, rule["lo"][index]), rule["hi"][index])
    return (bounded - rule["mean"][index]) / rule["scale"][index] * rule["w"][index]


def local_decision(rule, edge_margin):
    if rule["mode"] == "original":
        return edge_margin >= math.log(0.3 / 0.7)
    edge = term(rule, 0, edge_margin)
    endpoints = [term(rule, 1, rule[which][1]) for which in ("lo", "hi")]
    if edge + min(endpoints) >= rule["threshold"]:
        return True
    if edge + max(endpoints) < rule["threshold"]:
        return False
    return None


def remote_decision(rule, edge_margin, dino_margin):
    # The PRD1 response contains a little-endian float32 signed source margin.
    wire_margin = struct.unpack("<f", struct.pack("<f", dino_margin))[0]
    return term(rule, 0, edge_margin) + term(rule, 1, wire_margin) >= rule["threshold"]


def export():
    original = (BASE / "results/dual_fusion_rules.json").read_bytes()
    if hashlib.sha256(original).hexdigest() != FROZEN_SHA256:
        raise ValueError("Frozen pool-only rules changed")
    source = json.loads(original)
    selected = next(r for r in source["rules"] if r["budget"] == 450 and r["seed"] == 0)
    rule = {k: selected[k] for k in ("mode", "lo", "hi", "scale", "mean", "w", "threshold")}
    path = BASE / "results/dual_fusion_deployment.json"
    # Source weights reside with the server. The inline SHA identifies the required
    # unmodified head; the device stores these two-score fusion parameters only.
    blob = json.dumps(dict(**rule, seed=0, budget=450, source_head_sha256=source["source_head_sha256"],
                           frozen_rules_sha256=FROZEN_SHA256),
                      separators=(",", ":"), allow_nan=False).encode() + b"\n"
    path.write_bytes(blob)
    print(path.name, len(blob), hashlib.sha256(blob).hexdigest())


if __name__ == "__main__":
    export()
