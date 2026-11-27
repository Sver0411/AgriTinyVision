"""Fixed-look equal-tailed finite-population inference on paired binary predictions."""
import numpy as np
from scipy.stats import hypergeom


def interval(D, n, x, alpha=0.05):
    """Return inclusive endpoints for the total number of repairs among D disagreements."""
    if not (0 <= x <= n <= D and 0 < alpha < 1):
        raise ValueError('invalid audit sample or tail probability')
    if n == 0:
        return 0, D
    # The upper-tail probability increases with the unknown number of successes.
    left, right = x, D - (n-x)
    while left < right:
        mid = (left+right)//2
        if hypergeom.sf(x-1, D, mid, n) > alpha/2 + 1e-14:
            right = mid
        else:
            left = mid+1
    low = left
    # The lower-tail probability decreases as the number of successes increases.
    left, right = low, D - (n-x)
    while left < right:
        mid = (left+right+1)//2
        if hypergeom.cdf(x, D, mid, n) > alpha/2 + 1e-14:
            left = mid
        else:
            right = mid-1
    return low, left


def correct_sign_power(N, D, R, budget, design='disagreement', alpha=0.05):
    """Retrospective probability that a fixed-look interval establishes the true strict sign."""
    if not (0 <= R <= D <= N and N > 0 and budget >= 0 and 0 < alpha < 1):
        raise ValueError('invalid population or budget')
    if design not in ('uniform', 'disagreement'):
        raise ValueError('unknown sampling design')
    if D == 0 or 2*R == D or budget == 0:
        return 0.0
    good = max(R, D-R)
    if design == 'disagreement':
        draws, weights = np.array([min(budget,D)]), np.ones(1)
    else:
        b = min(budget,N)
        draws = np.arange(max(0,b-(N-D)),min(b,D)+1)
        weights = hypergeom.pmf(draws,N,D,b)
    # Under either sign the favorable direction must defeat the nearest null R=floor(D/2).
    null = D//2
    cutoff = hypergeom.isf(alpha/2,D,null,draws).astype(int)+1
    # Resolve round-off at an exact tail boundary by checking the adjacent candidate.
    adj = hypergeom.sf(cutoff-2,D,null,draws)
    cutoff -= adj <= alpha/2 + 1e-14
    cutoff += hypergeom.sf(cutoff-1,D,null,draws) > alpha/2 + 1e-14
    return float(np.clip(np.dot(weights, hypergeom.sf(cutoff-1,D,good,draws)),0,1))
