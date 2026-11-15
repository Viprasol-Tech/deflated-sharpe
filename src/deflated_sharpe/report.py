"""Load trial returns from CSV and build an overfitting report.

Part of deflated-sharpe by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .pbo import PBOResult, probability_of_backtest_overfitting
from .stats import (
    annualize,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
    sample_variance,
)

CONFIDENCE = 0.95


@dataclass(frozen=True)
class Report:
    best_trial: str
    n_periods: int
    n_trials: int
    sharpe: float
    sharpe_annual: float
    skew: float
    kurtosis: float
    psr: float
    expected_max_sharpe: float
    dsr: float
    min_track_record: float
    pbo: PBOResult | None
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_trials(path: Path) -> dict[str, list[float]]:
    """Read a CSV with a header row; one numeric column per trial.

    Columns that are not entirely numeric (e.g. a date column) are ignored.
    """
    with path.open(newline="", encoding="utf-8-sig") as fh:
        rows = [r for r in csv.reader(fh) if any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise ValueError(f"{path}: need a header row and at least one data row")
    header, data = rows[0], rows[1:]
    trials: dict[str, list[float]] = {}
    for i, name in enumerate(header):
        try:
            values = [float(r[i]) for r in data]
        except (ValueError, IndexError):
            continue
        if all(math.isfinite(v) for v in values):
            trials[name.strip() or f"col{i}"] = values
    if not trials:
        raise ValueError(f"{path}: no fully numeric columns found")
    return trials


def build_report(
    trials: Mapping[str, Sequence[float]],
    periods_per_year: float = 252.0,
    n_splits: int = 10,
    n_trials: int | None = None,
    sharpe_variance: float | None = None,
) -> Report:
    """Pick the best trial by Sharpe, then deflate it for everything you tried."""
    stats = {name: moments(r) for name, r in trials.items()}
    best = max(stats, key=lambda k: stats[k].sharpe)
    m = stats[best]
    total_trials = n_trials if n_trials is not None else len(trials)
    if total_trials < len(trials):
        raise ValueError("n_trials cannot be smaller than the number of columns")
    var = (
        sharpe_variance
        if sharpe_variance is not None
        else sample_variance([s.sharpe for s in stats.values()])
    )
    if total_trials > 1 and var == 0.0:
        raise ValueError("need --sharpe-variance when only one trial's returns are given")

    pbo: PBOResult | None = None
    if len(trials) >= 2 and m.n >= 2 * n_splits:
        pbo = probability_of_backtest_overfitting(list(trials.values()), n_splits)

    dsr = deflated_sharpe_ratio(m.sharpe, m.n, total_trials, var, m.skew, m.kurtosis)
    genuine = dsr >= CONFIDENCE and (pbo is None or pbo.pbo <= 0.5)
    return Report(
        best_trial=best,
        n_periods=m.n,
        n_trials=total_trials,
        sharpe=m.sharpe,
        sharpe_annual=annualize(m.sharpe, periods_per_year),
        skew=m.skew,
        kurtosis=m.kurtosis,
        psr=probabilistic_sharpe_ratio(m.sharpe, m.n, m.skew, m.kurtosis),
        expected_max_sharpe=expected_max_sharpe(total_trials, var),
        dsr=dsr,
        min_track_record=min_track_record_length(
            m.sharpe, m.skew, m.kurtosis, confidence=CONFIDENCE
        ),
        pbo=pbo,
        verdict="LIKELY GENUINE" if genuine else "LIKELY OVERFIT / NOT SIGNIFICANT",
    )


def noise_trials(
    n_trials: int, n_periods: int, seed: int = 7, vol: float = 0.01
) -> dict[str, list[float]]:
    """Skill-less strategies: iid normal returns with zero mean."""
    rng = random.Random(seed)
    return {
        f"trial_{i:03d}": [rng.gauss(0.0, vol) for _ in range(n_periods)] for i in range(n_trials)
    }


def write_trials(path: Path, trials: Mapping[str, Sequence[float]]) -> None:
    names = list(trials)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for row in zip(*(trials[n] for n in names), strict=True):
            w.writerow(f"{v:.8f}" for v in row)
