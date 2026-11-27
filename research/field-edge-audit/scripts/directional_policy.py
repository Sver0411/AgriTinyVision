"""Deploy a locked two-direction edge/DINO policy, including its actual wire formats."""

from dataclasses import dataclass
import struct
import numpy as np

CONFIG = struct.Struct("<4sBBffff")
REQUEST = struct.Struct("<4sI")
REPLY = struct.Struct("<4sIf")
CALIBRATION = struct.Struct("<4sIfB")
EDGE_CUTOFF = np.float32(0.3)


@dataclass(frozen=True)
class Rule:
    rescue: bool = False
    retract: bool = False
    edge_lower: float = 0.0
    edge_upper: float = 1.0
    rescue_cutoff: float = 0.5
    retract_cutoff: float = 0.5


def pack(rule):
    return CONFIG.pack(
        b"DD01",
        1,
        int(rule.rescue) + 2 * int(rule.retract),
        rule.edge_lower,
        rule.edge_upper,
        rule.rescue_cutoff,
        rule.retract_cutoff,
    )


def unpack(blob):
    magic, version, flags, a, b, c, d = CONFIG.unpack(blob)
    if (
        (magic, version) != (b"DD01", 1)
        or flags > 3
        or not (0 <= a <= EDGE_CUTOFF <= b <= 1 and 0.5 <= c <= 1 and 0 <= d <= 0.5)
    ):
        raise ValueError("Invalid directional rule")
    return Rule(bool(flags & 1), bool(flags & 2), a, b, c, d)


def probabilities_from_logits(logits):
    z = np.asarray(logits, dtype=np.float64)
    return np.exp(-np.logaddexp(0.0, -z)).astype(np.float32)


def select_queries(edge_probability, rule):
    e = np.asarray(edge_probability, dtype=np.float32)
    if not np.isfinite(e).all() or np.any((e < 0) | (e > 1)):
        raise ValueError("Invalid edge probability")
    edge = e >= EDGE_CUTOFF
    ask_rescue = rule.rescue & ~edge & (e >= np.float32(rule.edge_lower))
    ask_retract = rule.retract & edge & (e <= np.float32(rule.edge_upper))
    return ask_rescue | ask_retract


def decide(edge_probability, remote_probability, rule):
    """Nonqueried remote entries may be NaN; an answered query must be a valid probability."""
    e, d = np.broadcast_arrays(
        np.asarray(edge_probability, dtype=np.float32),
        np.asarray(remote_probability, dtype=np.float32),
    )
    queried = select_queries(e, rule)
    if np.any(queried & (~np.isfinite(d) | (d < 0) | (d > 1))):
        raise ValueError("Missing or invalid remote answer")
    original = e >= EDGE_CUTOFF
    return np.where(
        queried & ~original & (d >= np.float32(rule.rescue_cutoff)),
        True,
        original & ~(queried & original & (d < np.float32(rule.retract_cutoff))),
    )
