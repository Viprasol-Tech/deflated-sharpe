import json
import random

import pytest

from deflated_sharpe import build_report, load_trials, probability_of_backtest_overfitting
from deflated_sharpe.cli import main
from deflated_sharpe.report import noise_trials, write_trials


def test_pbo_high_for_pure_noise():
    trials = list(noise_trials(30, 600, seed=3).values())
    res = probability_of_backtest_overfitting(trials, n_splits=10)
    assert res.n_combinations == 252
    assert 0.3 <= res.pbo <= 0.9


def test_pbo_low_when_one_trial_has_real_edge():
    rng = random.Random(5)
    trials = [[rng.gauss(0, 0.01) for _ in range(600)] for _ in range(20)]
    trials[7] = [rng.gauss(0.004, 0.01) for _ in range(600)]
    res = probability_of_backtest_overfitting(trials, n_splits=10)
    assert res.pbo < 0.05
    assert res.median_logit > 0


def test_pbo_drops_remainder_rows_and_validates():
    trials = list(noise_trials(4, 105).values())
    res = probability_of_backtest_overfitting(trials, n_splits=10)
    assert (res.rows_used, res.rows_dropped) == (100, 5)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(trials[:1])
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(trials, n_splits=7)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting([trials[0], trials[1][:50]])


def test_report_flags_best_of_noise():
    r = build_report(noise_trials(50, 756))
    assert r.n_trials == 50
    assert r.dsr < r.psr
    assert r.verdict.startswith("LIKELY OVERFIT")


def test_report_accepts_genuine_edge():
    rng = random.Random(11)
    trials = noise_trials(10, 1000, seed=2)
    trials["edge"] = [rng.gauss(0.003, 0.01) for _ in range(1000)]
    r = build_report(trials)
    assert r.best_trial == "edge"
    assert r.dsr > 0.95
    assert r.verdict == "LIKELY GENUINE"


def test_report_single_column_needs_variance_when_trials_given():
    trials = {"only": noise_trials(1, 300)["trial_000"]}
    assert build_report(trials).pbo is None
    with pytest.raises(ValueError):
        build_report(trials, n_trials=20)
    r = build_report(trials, n_trials=20, sharpe_variance=0.003)
    assert r.dsr < r.psr


def test_load_trials_skips_non_numeric_columns(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("date,a,b\n2025-01-01,0.01,-0.02\n2025-01-02,0.0,0.01\n\n", encoding="utf-8")
    assert load_trials(p) == {"a": [0.01, 0.0], "b": [-0.02, 0.01]}
    bad = tmp_path / "bad.csv"
    bad.write_text("date\n2025-01-01\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_trials(bad)


def test_cli_report_and_demo(tmp_path, capsys):
    out = tmp_path / "noise.csv"
    assert main(["demo", "--trials", "8", "--periods", "200", "--out", str(out)]) == 0
    assert "Deflated Sharpe" in capsys.readouterr().out
    write_trials(tmp_path / "copy.csv", load_trials(out))
    assert main(["report", str(tmp_path / "copy.csv"), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["n_trials"] == 8 and 0.0 <= data["dsr"] <= 1.0


def test_cli_reports_errors(tmp_path, capsys):
    assert main(["report", str(tmp_path / "missing.csv")]) == 2
    assert "dsr: error" in capsys.readouterr().err
