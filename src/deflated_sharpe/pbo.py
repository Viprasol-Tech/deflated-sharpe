"""Probability of Backtest Overfitting via Combinatorially Symmetric Cross-Validation.

Part of deflated-sharpe by Viprasol Tech Private Limited (https://viprasol.com).

Reference: Bailey, Borwein, Lopez de Prado & Zhu (2015), "The Probability of
Backtest Overfitting", Journal of Computational Finance.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from statistics import median

MAX_SPLITS = 16  # C(16, 8) = 12,870 combinations; beyond this it gets slow


@dataclass(frozen=True)
class PBOResult:
    pbo: float  # share of splits where the in-sample winner is below the OOS median
    n_combinations: int
    median_logit: float
    rows_used: int
    rows_dropped: int


def _sharpe(total: float, total_sq: float, n: int) -> float:
    mean = total / n
    var = (total_sq - total * total / n) / (n - 1)
    if var <= 1e-18:
        return 0.0 if mean == 0.0 else math.copysign(math.inf, mean)
    return mean / math.sqrt(var)


def probability_of_backtest_overfitting(
    trials: Sequence[Sequence[float]], n_splits: int = 10
) -> PBOResult:
    """Estimate PBO for a set of strategy trials.

    Args:
        trials: one return series per trial, all the same length (columns of a
            T x N matrix). These should be *every* configuration you tried.
        n_splits: number of equal row blocks S (even, 4..16). Every way of
            choosing S/2 blocks as in-sample is evaluated.
    """
    n = len(trials)
    if n < 2:
        raise ValueError("need at least 2 trials")
    length = len(trials[0])
    if any(len(t) != length for t in trials):
        raise ValueError("all trials must have the same number of returns")
    if n_splits % 2 or not 4 <= n_splits <= MAX_SPLITS:
        raise ValueError(f"n_splits must be even and between 4 and {MAX_SPLITS}")
    block = length // n_splits
    if block < 2:
        raise ValueError(f"need at least {2 * n_splits} returns for {n_splits} splits")
    used = block * n_splits

    # Per-block sums so each combination costs O(S * N), not O(T * N).
    sums = [[math.fsum(t[b * block : (b + 1) * block]) for t in trials] for b in range(n_splits)]
    sqs = [
        [math.fsum(x * x for x in t[b * block : (b + 1) * block]) for t in trials]
        for b in range(n_splits)
    ]

    half = n_splits // 2
    rows = half * block
    logits: list[float] = []
    for is_blocks in combinations(range(n_splits), half):
        oos_blocks = [b for b in range(n_splits) if b not in is_blocks]
        is_sr = [
            _sharpe(sum(sums[b][j] for b in is_blocks), sum(sqs[b][j] for b in is_blocks), rows)
            for j in range(n)
        ]
        oos_sr = [
            _sharpe(sum(sums[b][j] for b in oos_blocks), sum(sqs[b][j] for b in oos_blocks), rows)
            for j in range(n)
        ]
        best = max(range(n), key=is_sr.__getitem__)
        target = oos_sr[best]
        below = sum(1 for j in range(n) if oos_sr[j] < target)
        ties = sum(1 for j in range(n) if j != best and oos_sr[j] == target)
        omega = (1 + below + 0.5 * ties) / (n + 1)
        logits.append(math.log(omega / (1.0 - omega)))

    overfit = sum(1 for lam in logits if lam <= 0.0)
    return PBOResult(
        pbo=overfit / len(logits),
        n_combinations=len(logits),
        median_logit=median(logits),
        rows_used=used,
        rows_dropped=length - used,
    )
