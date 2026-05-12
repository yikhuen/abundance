#!/usr/bin/env python3
"""Sprint 3 · Stages 3.3+3.4: Funding carry backtest + parameter sweep.

Backtests the funding carry strategy on BTC, ETH, SOL and compares
against buy-and-hold benchmarks. Runs a parameter sweep across
thresholds to find optimal settings.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.backtesting.carry_strategy import CarryStrategy, run_carry_backtest
from abundance.backtesting.metrics import MetricsCalculator
from abundance.config.settings import settings


PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def load_pair_data(pair: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load kline and funding data for a pair."""
    kline_path = settings.raw_dir / "klines" / f"{pair.lower()}_1h"
    funding_path = settings.raw_dir / "funding" / pair.lower()

    kline = (
        pl.scan_parquet(str(kline_path / "**" / "*.parquet"))
        .sort("timestamp_ms")
        .select(["timestamp_ms", "close"])
        .collect()
    )
    funding = (
        pl.scan_parquet(str(funding_path / "**" / "*.parquet"))
        .sort("timestamp_ms")
        .collect()
    )

    return kline, funding


def run_single(pair: str) -> dict:
    """Run carry backtest for a single pair."""
    kline, funding = load_pair_data(pair)
    strategy = CarryStrategy.from_pair(pair)

    equity_curve, report, trades = run_carry_backtest(kline, funding, strategy)

    # Buy & hold benchmark (same period)
    close = kline["close"]
    initial = 10_000.0
    bh_equity = initial * (close / close[0])
    bh_curve = kline.select("timestamp_ms").with_columns(bh_equity.alias("equity"))
    bh_report = MetricsCalculator.from_equity_curve(bh_curve)

    return {
        "pair": pair,
        "threshold": strategy.entry_threshold_pct,
        "carry_sharpe": report.sharpe_ratio,
        "carry_return": report.total_return_pct,
        "carry_maxdd": report.max_drawdown_pct,
        "bh_sharpe": bh_report.sharpe_ratio,
        "bh_return": bh_report.total_return_pct,
        "bh_maxdd": bh_report.max_drawdown_pct,
        "num_trades": report.extra.get("num_trades", 0),
        "avg_trade_duration_hours": report.extra.get("avg_trade_duration_hours", 0),
        "avg_trade_pnl_pct": report.extra.get("avg_trade_pnl_pct", 0),
    }


def run_threshold_sweep(pair: str) -> list[dict]:
    """Sweep entry threshold to find optimal setting."""
    kline, funding = load_pair_data(pair)
    results = []

    thresholds = [0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050]
    for entry in thresholds:
        strategy = CarryStrategy(
            entry_threshold_pct=entry,
            exit_threshold_pct=entry / 2,
        )
        _, report, trades = run_carry_backtest(kline, funding, strategy)

        results.append(
            {
                "pair": pair,
                "threshold": entry,
                "sharpe": report.sharpe_ratio,
                "return": report.total_return_pct,
                "max_dd": report.max_drawdown_pct,
                "calmar": report.calmar_ratio,
                "trades": report.extra.get("num_trades", 0),
                "avg_duration_h": report.extra.get("avg_trade_duration_hours", 0),
            }
        )

    return results


def main() -> None:
    """Run full carry strategy backtest suite."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 3 · Stages 3.3+3.4")
    logger.info("Funding Carry Strategy — Backtest + Parameter Sweep")
    logger.info("=" * 60)

    # ── Single-pair backtests ──────────────────────────────
    logger.info("\n--- Individual Pair Results ---\n")
    logger.info(
        f"{'Pair':<10} {'Strategy':<12} {'Sharpe':>7} {'Return%':>9} "
        f"{'MaxDD%':>7} {'Trades':>7}"
    )
    logger.info(f"{'-'*10} {'-'*12} {'-'*7} {'-'*9} {'-'*7} {'-'*7}")

    all_results = []
    for pair in PAIRS:
        result = run_single(pair)
        all_results.append(result)

        carry = result
        logger.info(
            f"{carry['pair']:<10} {'Carry':<12} "
            f"{carry['carry_sharpe']:>7.3f} {carry['carry_return']:>8.1f}% "
            f"{carry['carry_maxdd']:>6.1f}% {carry['num_trades']:>7.0f}"
        )
        logger.info(
            f"{'':<10} {'Buy&Hold':<12} "
            f"{carry['bh_sharpe']:>7.3f} {carry['bh_return']:>8.1f}% "
            f"{carry['bh_maxdd']:>6.1f}%"
        )
        if carry["num_trades"] > 0:
            logger.info(
                f"{'':<10} {'':<12} "
                f"Avg trade: {carry['avg_trade_pnl_pct']:+.2f}%, "
                f"{carry['avg_trade_duration_hours']:.0f}h duration"
            )

    # ── Parameter sweep ────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("Parameter Sweep — BTCUSDT Threshold Optimization")
    logger.info(f"{'='*60}")
    logger.info(
        f"{'Entry%':>7} {'Sharpe':>7} {'Return%':>9} {'MaxDD%':>7} "
        f"{'Calmar':>7} {'Trades':>7}"
    )
    logger.info(f"{'-'*7} {'-'*7} {'-'*9} {'-'*7} {'-'*7} {'-'*7}")

    sweep = run_threshold_sweep("BTCUSDT")
    best_sharpe = max(sweep, key=lambda x: x["sharpe"])
    best_calmar = max(sweep, key=lambda x: x["calmar"])

    for r in sweep:
        marker = ""
        if r == best_sharpe:
            marker = " ← best Sharpe"
        elif r == best_calmar:
            marker = " ← best Calmar"
        logger.info(
            f"{r['threshold']:>6.3f}% {r['sharpe']:>7.3f} "
            f"{r['return']:>8.1f}% {r['max_dd']:>6.1f}% "
            f"{r['calmar']:>7.3f} {r['trades']:>7.0f}{marker}"
        )

    logger.info(f"\n{'='*60}")
    logger.info("Sprint 3 — Funding Carry Strategy COMPLETE")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
