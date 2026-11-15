import math
import random

import pytest

from deflated_sharpe import (
    annualize,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    min_track_record_length,
    moments,
    probabilistic_sharpe_ratio,
)


def test_moments_of_symmetric_series():
    m = moments([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert m.n == 5
    assert m.mean == 0.0
    assert m.std == pytest.approx(math.sqrt(2.5))
    assert m.sharpe == 0.0
    assert m.skew == pytest.approx(0.0)
    assert m.kurtosis == pytest.approx(1.7)  # m4/m2^2 = 6.8 / 4


def test_moments_rejects_bad_input():
    with pytest.raises(ValueError):
        moments([1.0, 2.0])
    with pytest.raises(ValueError):
        moments([0.5, 0.5, 0.5])


def test_psr_is_half_at_benchmark():
    assert probabilistic_sharpe_ratio(0.1, 500, benchmark=0.1) == pytest.approx(0.5)


def test_psr_matches_closed_form_under_normality():
    sr, n = 0.1, 253
    z = sr * math.sqrt(n - 1) / math.sqrt(1 + sr**2 / 2)
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert probabilistic_sharpe_ratio(sr, n) == pytest.approx(expected)


def test_negative_skew_and_fat_tails_lower_psr():
    base = probabilistic_sharpe_ratio(0.08, 500)
    assert probabilistic_sharpe_ratio(0.08, 500, skew=-1.0, kurtosis=8.0) < base


def test_expected_max_matches_known_value_for_100_normals():
    # E[max of 100 iid N(0,1)] = 2.5076; the Gumbel approximation is close.
    assert expected_max_sharpe(100, 1.0) == pytest.approx(2.5076, abs=0.05)


def test_expected_max_grows_with_trials_and_is_zero_for_one():
    assert expected_max_sharpe(1, 1.0) == 0.0
    values = [expected_max_sharpe(n, 0.01) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)


def test_expected_max_monte_carlo():
    rng = random.Random(1)
    n, sims = 20, 4000
    mc = sum(max(rng.gauss(0, 1) for _ in range(n)) for _ in range(sims)) / sims
    assert expected_max_sharpe(n, 1.0) == pytest.approx(mc, abs=0.08)


def test_dsr_is_below_psr_with_many_trials():
    psr = probabilistic_sharpe_ratio(0.1, 750)
    dsr = deflated_sharpe_ratio(0.1, 750, n_trials=100, sharpe_variance=0.0013)
    assert dsr < psr
    assert deflated_sharpe_ratio(0.1, 750, 1, 0.0) == pytest.approx(psr)


def test_min_track_record_closed_form_and_consistency():
    sr = 0.1
    trl = min_track_record_length(sr)
    z = 1.6448536269514722
    assert trl == pytest.approx(1 + (1 + sr**2 / 2) * (z / sr) ** 2)
    # At exactly MinTRL periods, PSR equals the confidence level.
    assert probabilistic_sharpe_ratio(sr, trl) == pytest.approx(0.95, abs=1e-9)  # type: ignore[arg-type]
    assert min_track_record_length(-0.1) == math.inf


def test_input_validation():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(0.1, 1)
    with pytest.raises(ValueError):
        expected_max_sharpe(0, 1.0)
    with pytest.raises(ValueError):
        expected_max_sharpe(5, -1.0)
    with pytest.raises(ValueError):
        min_track_record_length(0.1, confidence=1.0)


def test_annualize():
    assert annualize(0.1, 252) == pytest.approx(0.1 * math.sqrt(252))
