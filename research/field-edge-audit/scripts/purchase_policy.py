"""Choose an affordable deployment action using only purchased calibration outcomes."""

from dataclasses import dataclass
from math import sqrt
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class Estimate:
    action: str
    incremental_setup: int
    affordable: bool
    negative_gains: int
    negative_losses: int
    positive_gains: int
    positive_losses: int
    observed_ba_gain: float
    observed_sensitivity_gain: float
    estimated_ba_gain: float
    gain_sd: float
    learning_ba_gain: float
    learning_gain_sd: float
    expansion_credit: float
    score: float
    eligible: bool


@dataclass(frozen=True)
class Choice:
    action: str
    reason: str
    negatives: int
    positives: int
    estimates: tuple[Estimate, ...]


def binary_vector(value, count=None):
    a = np.asarray(value)
    if a.ndim != 1 or not np.all((a == 0) | (a == 1)):
        raise ValueError("Expected a one-dimensional binary vector")
    if count is not None and a.size != count:
        raise ValueError("Audit identities must be paired across actions")
    return a.astype(bool, copy=False)


def bytes_count(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("Expected a nonnegative integer byte cost")
    return int(value)


def contrast(wins, losses, count):
    """Mean and variance of P(gain)-P(loss) under a paired Dirichlet model."""
    total = count + 1.5
    mean = (wins - losses) / total
    variance = ((wins + losses + 1.0) / total - mean**2) / (total + 1.0)
    return mean, variance


def learning_contrast(y, pair):
    small, large = (binary_vector(v, len(y)) for v in pair)
    correct_small = small == y
    correct_large = large == y
    means, variances = [], []
    for mask in (~y, y):
        wins = np.count_nonzero(mask & correct_large & ~correct_small)
        losses = np.count_nonzero(mask & ~correct_large & correct_small)
        mean, variance = contrast(wins, losses, np.count_nonzero(mask))
        means.append(mean)
        variances.append(variance)
    return sum(means) / 2, sqrt(sum(variances)) / 2


def choose_action(paid_y, outcomes, remaining_bytes, setup_costs, *, frozen="freeze",
                  rule="protected", learning_pair=None, max_labels=32):
    """Audit complete actions on the same, already purchased independent pilot.

    Array elements are final predicted classes on pilot identities, inclusive of
    budget-matched routing. Input predictions must be honest on their pilot rows.
    Any not-yet-bought human supervision or post-pilot deployment costs must be
    included in setup_costs. Held-out outcomes are deliberately absent from the
    API. The comparison named frozen must require no additional bytes.
    An optional pair contrasts nested small/large honest OOF edge predictions.
    A dictionary with "plain" and "guard" pairs supports distinct guard gains.
    A single pair only gives expansion credit to the unguarded action.
    """
    if rule not in ("protected", "paired_erm", "without_recall_guard"):
        raise ValueError("Unknown action-selection rule")
    if max_labels not in (32, 128):
        raise ValueError("The locked supervision allowances are 32 and 128")
    remaining_bytes = bytes_count(remaining_bytes)
    if set(outcomes) != set(setup_costs) or frozen not in outcomes:
        raise ValueError("Every candidate needs pilot predictions and a price")
    fees = {name: bytes_count(value) for name, value in setup_costs.items()}
    if fees[frozen] != 0:
        raise ValueError("The reference action must be free after the pilot")
    y = binary_vector(paid_y)
    preds = {name: binary_vector(value, y.size) for name, value in outcomes.items()}
    npos = int(y.sum())
    nneg = int(y.size - npos)
    if min(npos, nneg) < 2:
        return Choice(frozen, "insufficient_class_support", nneg, npos, ())
    reference = preds[frozen] == y
    if learning_pair is None:
        learning_pair = {}
    elif not isinstance(learning_pair, dict):
        learning_pair = {"plain": learning_pair}
    if set(learning_pair) - {"plain", "guard"}:
        raise ValueError("Use plain or guard learning-curve evidence")
    evidence = {tag: learning_contrast(y, pair) for tag, pair in learning_pair.items()}
    # Saturating n**(-1/2) learning curve. Extrapolate at most the observed increment.
    factor = min(1., max(0., (1. - sqrt(len(y) / max_labels)) / (sqrt(2.) - 1.)))
    estimates = []
    for name, pred in preds.items():
        if name == frozen:
            continue
        wins = (pred == y) & ~reference
        losses = (pred != y) & reference
        ng, nl = int(np.count_nonzero(wins & ~y)), int(np.count_nonzero(losses & ~y))
        pg, pl = int(np.count_nonzero(wins & y)), int(np.count_nonzero(losses & y))
        raw_sens = (pg - pl) / npos
        raw_ba = ((ng - nl) / nneg + raw_sens) / 2
        neg_mean, neg_var = contrast(ng, nl, nneg)
        pos_mean, pos_var = contrast(pg, pl, npos)
        mean = (neg_mean + pos_mean) / 2
        sd = sqrt((neg_var + pos_var) / 4)
        learn_mean, learn_sd, credit = 0., 0., 0.
        if name in ("expand", "expand_guard") and rule != "paired_erm":
            tag = "guard" if name == "expand_guard" else "plain"
            learn_mean, learn_sd = evidence.get(tag, (0., 0.))
            credit = factor * max(0., learn_mean - learn_sd)
        score = raw_ba if rule == "paired_erm" else mean - sd + credit
        affordable = fees[name] <= remaining_bytes
        eligible = affordable and score > 0 and (rule != "protected" or pg >= pl)
        estimates.append(Estimate(name, fees[name], affordable, ng, nl, pg, pl, raw_ba,
                                  raw_sens, mean, sd, learn_mean, learn_sd, credit, score, eligible))
    choices = [row for row in estimates if row.eligible]
    if not choices:
        reason = "no_supported_gain" if any(row.affordable for row in estimates) else "unaffordable"
        return Choice(frozen, reason, nneg, npos, tuple(estimates))
    best = min(choices, key=lambda row: (-row.score, row.incremental_setup, row.action))
    return Choice(best.action, "paid_evidence", nneg, npos, tuple(estimates))
