"""deflated-sharpe: tell a real backtest edge from multiple-testing luck.

Part of deflated-sharpe by Viprasol Tech Private Limited (https://viprasol.com).
"""

from .pbo import PBOResult, probability_of_backtest_overfitting
from .report import Report, build_report, load_trials
from .stats import (
    Moments,
    annualize,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
)

__version__ = "0.1.0"

__all__ = [
    "Moments",
    "PBOResult",
    "Report",
    "__version__",
    "annualize",
    "build_report",
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "load_trials",
    "min_track_record_length",
    "moments",
    "probabilistic_sharpe_ratio",
    "probability_of_backtest_overfitting",
]
