"""Command-line interface: ``dsr report`` and ``dsr demo``.

Part of deflated-sharpe by Viprasol Tech Private Limited (https://viprasol.com).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from .report import Report, build_report, load_trials, noise_trials, write_trials


def _fmt_trl(periods: float) -> str:
    return "never (Sharpe <= 0)" if periods == float("inf") else f"{periods:,.0f} periods"


def render(r: Report) -> str:
    lines = [
        f"Best trial          {r.best_trial}  ({r.n_periods:,} periods, {r.n_trials} trials)",
        f"Sharpe              {r.sharpe:.4f} per period  |  {r.sharpe_annual:.2f} annualized",
        f"Skew / kurtosis     {r.skew:.3f} / {r.kurtosis:.3f}",
        f"PSR (SR > 0)        {r.psr:.1%}",
        f"E[max SR] of noise  {r.expected_max_sharpe:.4f} per period",
        f"Deflated Sharpe     {r.dsr:.1%}",
        f"Min track record    {_fmt_trl(r.min_track_record)} for 95% PSR",
    ]
    if r.pbo is not None:
        lines.append(
            f"PBO (CSCV)          {r.pbo.pbo:.1%}  "
            f"({r.pbo.n_combinations:,} splits, median logit {r.pbo.median_logit:+.2f})"
        )
    else:
        lines.append("PBO (CSCV)          n/a (needs >= 2 trials and enough rows)")
    lines.append(f"Verdict             {r.verdict}")
    return "\n".join(lines)


def _emit(r: Report, as_json: bool) -> None:
    print(json.dumps(r.to_dict(), indent=2) if as_json else render(r))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="dsr",
        description="Is your best backtest real? Deflated Sharpe Ratio + PBO, no dependencies.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    rep = sub.add_parser("report", help="analyze a CSV of trial returns (one column per trial)")
    rep.add_argument("csv", type=Path)
    rep.add_argument("--periods-per-year", type=float, default=252.0)
    rep.add_argument("--splits", type=int, default=10, help="CSCV blocks (even, 4-16)")
    rep.add_argument("--n-trials", type=int, help="total configs tried, if more than columns")
    rep.add_argument("--sharpe-variance", type=float, help="variance of trial Sharpes")
    rep.add_argument("--json", action="store_true")

    demo = sub.add_parser("demo", help="run the report on pure-noise strategies")
    demo.add_argument("--trials", type=int, default=50)
    demo.add_argument("--periods", type=int, default=756)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--out", type=Path, help="also write the generated CSV here")
    demo.add_argument("--json", action="store_true")

    args = p.parse_args(argv)
    try:
        if args.cmd == "report":
            r = build_report(
                load_trials(args.csv),
                periods_per_year=args.periods_per_year,
                n_splits=args.splits,
                n_trials=args.n_trials,
                sharpe_variance=args.sharpe_variance,
            )
        else:
            trials = noise_trials(args.trials, args.periods, args.seed)
            if args.out:
                write_trials(args.out, trials)
            r = build_report(trials)
    except (OSError, ValueError) as exc:
        print(f"dsr: error: {exc}", file=sys.stderr)
        return 2
    _emit(r, args.json)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
