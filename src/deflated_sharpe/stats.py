"""Sharpe-ratio statistics that correct for non-normal returns and multiple testing.

Part of deflated-sharpe by Viprasol Tech Private Limited (https://viprasol.com).

References:
    Bailey & Lopez de Prado (2012), "The Sharpe Ratio Efficient Frontier".
    Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality".

All Sharpe ratios here are *per period* (not annualized): if your returns are
daily, so is the Sharpe ratio. Use :func:`annualize` only for display.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


@dataclass(frozen=True)
class Moments:
    """Summary statistics of a return series."""

    n: int
    mean: float
    std: float
    sharpe: float
    skew: float
    kurtosis: float  # raw (non-excess) kurtosis; 3.0 for a normal distribution


def moments(returns: Sequence[float]) -> Moments:
    """Sample Sharpe ratio (ddof=1) plus population skewness and raw kurtosis."""
    n = len(returns)
    if n < 3:
        raise ValueError(f"need at least 3 returns, got {n}")
    mean = math.fsum(returns) / n
    dev = [r - mean for r in returns]
    m2 = math.fsum(d * d for d in dev) / n
    if m2 <= 0.0:
        raise ValueError("returns have zero variance; Sharpe ratio is undefined")
    m3 = math.fsum(d**3 for d in dev) / n
    m4 = math.fsum(d**4 for d in dev) / n
    std = math.sqrt(m2 * n / (n - 1))
    return Moments(
        n=n,
        mean=mean,
        std=std,
        sharpe=mean / std,
        skew=m3 / m2**1.5,
        kurtosis=m4 / m2**2,
    )


def _sr_std_error(sharpe: float, skew: float, kurtosis: float) -> float:
    """sqrt(1 - skew*SR + (kurt-1)/4 * SR^2): the non-normality adjustment."""
    var = 1.0 - skew * sharpe + (kurtosis - 1.0) / 4.0 * sharpe**2
    if var <= 0.0:
        raise ValueError("moment combination gives a non-positive Sharpe variance")
    return math.sqrt(var)


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    """P(true Sharpe > benchmark) given an observed per-period Sharpe over n periods."""
    if n < 2:
        raise ValueError("n must be at least 2")
    z = (sharpe - benchmark) * math.sqrt(n - 1) / _sr_std_error(sharpe, skew, kurtosis)
    return _N.cdf(z)


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe among n_trials skill-less strategies.

    Uses the Gumbel approximation from Bailey & Lopez de Prado (2014). With one
    trial there is no selection, so the benchmark is 0.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if sharpe_variance < 0.0:
        raise ValueError("sharpe_variance must be non-negative")
    if n_trials == 1:
        return 0.0
    a = _N.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sharpe_variance) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe_ratio(
    sharpe: float,
    n: int,
    n_trials: int,
    sharpe_variance: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """PSR against the Sharpe you'd expect from the best of n_trials pure-noise trials."""
    sr0 = expected_max_sharpe(n_trials, sharpe_variance)
    return probabilistic_sharpe_ratio(sharpe, n, skew, kurtosis, benchmark=sr0)


def min_track_record_length(
    sharpe: float,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """Periods needed before PSR(benchmark) reaches the given confidence."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if sharpe <= benchmark:
        return math.inf
    z = _N.inv_cdf(confidence)
    adj = _sr_std_error(sharpe, skew, kurtosis) ** 2
    return 1.0 + adj * (z / (sharpe - benchmark)) ** 2


def annualize(sharpe: float, periods_per_year: float) -> float:
    """Scale a per-period Sharpe ratio to annual (iid assumption)."""
    return sharpe * math.sqrt(periods_per_year)


def sample_variance(values: Sequence[float]) -> float:
    """Unbiased sample variance (ddof=1); 0.0 for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    return math.fsum((v - mean) ** 2 for v in values) / (n - 1)
