<div align="center">

<img src="docs/assets/logo.png" alt="Viprasol Tech logo" width="120" />

# deflated-sharpe

**Is your best backtest real, or just the luckiest of everything you tried?**

_Built and maintained by [Viprasol Tech](https://viprasol.com) — Fintech Experts. Full-Stack Builders._

[![CI](https://img.shields.io/github/actions/workflow/status/Viprasol-Tech/deflated-sharpe/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/Viprasol-Tech/deflated-sharpe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-black?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue?style=flat-square)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0-brightgreen?style=flat-square)](pyproject.toml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-261230?style=flat-square)](https://github.com/astral-sh/ruff)
[![Telegram](https://img.shields.io/badge/Telegram-help-26A5E4?style=flat-square)](https://t.me/viprasol_help)

</div>

---

Run 50 parameter sets on pure noise and the best one will show a Sharpe of about 1.5. A plain t-test calls that significant. `deflated-sharpe` checks it properly: it corrects the Sharpe ratio for **how many trials you ran**, for **fat tails and skew**, and for **out-of-sample rank decay**. You get a clear verdict in one command.

It is a small, typed Python library and a `dsr` CLI with **zero dependencies**. Everything is built on the standard library (`statistics.NormalDist`, `math.fsum`).

> ## ⚠️ Disclaimer
> This software is for **educational and research purposes only** and is **not financial advice**. A passing score does not mean a strategy will make money; markets change, and these statistics rest on assumptions (for example, roughly iid returns). Use at your own risk. Viprasol Tech is not affiliated with any third party referenced here.

## ✨ Features

- 📉 **Deflated Sharpe Ratio (DSR):** the probability that the true Sharpe beats the best Sharpe you would expect from N skill-less trials ([Bailey & López de Prado, 2014](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551)).
- 🎯 **Probabilistic Sharpe Ratio (PSR):** P(true SR > benchmark), adjusted for skewness and kurtosis.
- 🎲 **Expected maximum Sharpe** of N noise strategies, using the Gumbel approximation and checked against Monte Carlo in the tests.
- ⏳ **Minimum Track Record Length:** how many periods you need before a Sharpe is credible at 95%.
- 🔀 **Probability of Backtest Overfitting (PBO)** via Combinatorially Symmetric Cross-Validation ([Bailey et al., 2015](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)). It tries every way of splitting your history in half.
- ⚡ **Fast CSCV:** per-block sums keep each split at O(S·N), so 12,870 splits finish in seconds in pure Python.
- 📄 **CSV in, verdict out:** one column per trial; non-numeric columns such as dates are skipped automatically.
- 🧾 **`--json` output** for CI gates and notebooks.
- 🧮 **`--n-trials` override** for when you tried more configurations than you kept.
- 🧪 **Offline demo** on synthetic noise strategies that shows the trap live.
- 🔒 Typed (`mypy --strict`), tested, with zero runtime dependencies.

## 🚀 Quickstart

```bash
git clone https://github.com/Viprasol-Tech/deflated-sharpe.git
cd deflated-sharpe
pip install -e .

dsr demo                                   # 50 noise strategies: watch the "winner" fail
dsr report examples/edge_vs_noise.csv      # 9 noise trials + 1 real edge
```

`dsr demo` (every trial is zero-mean noise):

```text
Best trial          trial_017  (756 periods, 50 trials)
Sharpe              0.0989 per period  |  1.57 annualized
Skew / kurtosis     -0.032 / 2.874
PSR (SR > 0)        99.7%
E[max SR] of noise  0.0805 per period
Deflated Sharpe     69.3%
Min track record    280 periods for 95% PSR
PBO (CSCV)          46.8%  (252 splits, median logit +1.68)
Verdict             LIKELY OVERFIT / NOT SIGNIFICANT
```

A naive PSR says 99.7%. After deflating for 50 trials it drops to 69%, and almost half of the out-of-sample splits rank the "winner" below the median.

`dsr report examples/edge_vs_noise.csv` (one trial has a real drift):

```text
Best trial          momentum_v3  (756 periods, 10 trials)
Sharpe              0.1919 per period  |  3.05 annualized
...
Deflated Sharpe     97.5%
PBO (CSCV)          0.0%  (252 splits, median logit +2.30)
Verdict             LIKELY GENUINE
```

## 📥 Input format

A CSV with a header row and **one column of per-period returns per trial**: every parameter set or variant you backtested, not just the winner.

```csv
date,ema_10_30,ema_20_50,ema_50_200
2025-01-02,0.0041,-0.0012,0.0008
2025-01-03,-0.0020,0.0035,0.0011
```

| Flag | Default | Meaning |
|---|---|---|
| `--periods-per-year` | `252` | Used only to show the annualized Sharpe |
| `--splits` | `10` | CSCV blocks S (even, 4–16); runs C(S, S/2) splits |
| `--n-trials` | number of columns | Total configurations tried, if you discarded some |
| `--sharpe-variance` | variance of column Sharpes | Override; required if `--n-trials` > 1 with a single column |
| `--json` | off | Machine-readable output |

The exit code is `0` on success and `2` on bad input.

## 🐍 Library usage

```python
from deflated_sharpe import (
    moments, deflated_sharpe_ratio, probabilistic_sharpe_ratio,
    min_track_record_length, probability_of_backtest_overfitting,
)

m = moments(daily_returns)                      # Sharpe (per period), skew, raw kurtosis
psr = probabilistic_sharpe_ratio(m.sharpe, m.n, m.skew, m.kurtosis)
dsr = deflated_sharpe_ratio(m.sharpe, m.n, n_trials=120, sharpe_variance=0.0015,
                            skew=m.skew, kurtosis=m.kurtosis)
trl = min_track_record_length(m.sharpe, m.skew, m.kurtosis)

res = probability_of_backtest_overfitting(list_of_trial_return_series, n_splits=16)
print(res.pbo, res.n_combinations)
```

## 🧠 How it works

| Metric | Formula (per-period SR) |
|---|---|
| PSR(SR*) | Φ( (SR − SR*) · √(T−1) / √(1 − γ₃·SR + (γ₄−1)/4 · SR²) ) |
| E[max SR] | √V[SRₙ] · ( (1−γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ), γ ≈ 0.5772 |
| DSR | PSR(E[max SR]) |
| MinTRL | 1 + (1 − γ₃·SR + (γ₄−1)/4 · SR²) · (Φ⁻¹(p) / (SR − SR*))² |
| PBO | share of CSCV splits where the in-sample best trial ranks at or below the out-of-sample median |

Here γ₃ is skewness and γ₄ is raw kurtosis (3 for a normal distribution). The verdict is **LIKELY GENUINE** only when DSR ≥ 95% and PBO ≤ 50%.

## 🗺️ Roadmap

- [x] PSR, DSR, MinTRL, PBO/CSCV, CLI, JSON output
- [ ] Effective number of trials from clustering correlated trials
- [ ] Autocorrelation-adjusted Sharpe (Lo, 2002)
- [ ] Optional numpy fast path for very large trial matrices
- [ ] GitHub Action wrapper to fail CI on overfit strategies

## 🤝 Contributing

PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). Run `ruff check . && ruff format --check . && mypy src && pytest` before you open one.

## 📬 Contact — Viprasol Tech Private Limited

- 🌐 Website: [viprasol.com](https://viprasol.com)
- ✉️ Email: [support@viprasol.com](mailto:support@viprasol.com)
- 💬 Telegram: [t.me/viprasol_help](https://t.me/viprasol_help) · 📱 WhatsApp: +91 96336 52112
- 🐙 GitHub: [@Viprasol-Tech](https://github.com/Viprasol-Tech) · 💼 [LinkedIn](https://www.linkedin.com/in/viprasol/) · 𝕏 [@viprasol](https://twitter.com/viprasol)

> *Viprasol Tech builds fintech software, algorithmic trading systems, MT4/MT5 bots, AI voice agents, and B2B SaaS. Need a custom build? [Get in touch](mailto:support@viprasol.com).*

## License

[MIT](LICENSE) © 2025 Viprasol Tech Private Limited
