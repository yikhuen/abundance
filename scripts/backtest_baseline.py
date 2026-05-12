#!/usr/bin/env python3
"""Sprint 2: Evaluation harness — backtest runner with NautilusTrader.

Runs a baseline EMA crossover strategy and buy-and-hold benchmark
on BTCUSDT 1h data, computes full metrics suite.

Uses NautilusTrader's high-level BacktestNode API with a Parquet
data catalog fed from our kline pipeline.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.backtesting.catalog import CatalogWriter
from abundance.backtesting.metrics import MetricsCalculator, MetricsReport
from abundance.config.settings import settings


def run_buy_and_hold(df: pl.DataFrame) -> MetricsReport:
    """Run buy-and-hold benchmark on the data."""
    initial_capital = 10_000.0

    close = df["close"]
    equity = initial_capital * (close / close[0])

    equity_curve = df.select("timestamp_ms").with_columns(
        equity.alias("equity")
    )

    return MetricsCalculator.from_equity_curve(equity_curve)


def run_ema_crossover(
    df: pl.DataFrame, fast_period: int = 20, slow_period: int = 50
) -> tuple[pl.DataFrame, MetricsReport]:
    """Run EMA crossover strategy (Polars-native, no NautilusTrader needed for basic).

    Strategy logic:
      - Bullish: fast EMA > slow EMA → go long
      - Bearish: fast EMA < slow EMA → go flat (close position)
      - Trades at next bar's close price (no lookahead)

    Returns (equity_curve, metrics_report).
    """
    initial_capital = 10_000.0
    commission_pct = 0.001  # 0.1% per trade

    # Compute EMAs
    fast_ema = df["close"].ewm_mean(span=fast_period, min_samples=fast_period)
    slow_ema = df["close"].ewm_mean(span=slow_period, min_samples=slow_period)

    # Signals: +1 = long, 0 = flat (shifted to avoid lookahead)
    signal = (fast_ema > slow_ema).cast(pl.Int8)
    signal = signal.shift(1).fill_null(0)  # no trade on first bar

    # Daily returns (buy-and-hold per period)
    period_returns = df["close"].pct_change().fill_null(0.0)
    strategy_returns = period_returns * signal.cast(pl.Float64)

    # Deduct commission on trades (when signal changes)
    signal_change = signal.diff().abs().fill_null(0).cast(pl.Float64)
    strategy_returns = strategy_returns - signal_change * commission_pct

    # Equity curve
    equity = [initial_capital]
    for ret in strategy_returns.to_list():
        equity.append(equity[-1] * (1 + ret))

    equity_curve = df.select("timestamp_ms").with_columns(
        pl.Series("equity", equity[1:])  # Skip initial capital row
    )

    # Trade log
    trades = []
    in_position = False
    entry_price = 0.0
    for i in range(1, len(df)):
        if signal[i] == 1 and signal[i - 1] == 0:
            # Enter long
            entry_price = df["close"][i]
            in_position = True
        elif signal[i] == 0 and signal[i - 1] == 1:
            # Exit long
            if in_position:
                exit_price = df["close"][i]
                pnl = (exit_price - entry_price) / entry_price
                pnl_after_comm = pnl - 2 * commission_pct
                trades.append(
                    {
                        "entry_ts": df["timestamp_ms"][i],
                        "pnl": pnl_after_comm * equity[i],
                        "return_pct": pnl_after_comm * 100,
                    }
                )
                in_position = False

    # Close any open position at end
    if in_position:
        exit_price = df["close"][-1]
        pnl = (exit_price - entry_price) / entry_price
        pnl_after_comm = pnl - commission_pct  # only exit commission
        trades.append(
            {
                "entry_ts": df["timestamp_ms"][-1],
                "pnl": pnl_after_comm * equity[-1],
                "return_pct": pnl_after_comm * 100,
            }
        )

    trades_df = pl.DataFrame(trades) if trades else pl.DataFrame(
        {"entry_ts": [], "pnl": [], "return_pct": []}
    )

    report = MetricsCalculator.from_equity_curve(equity_curve, trades_df)
    return equity_curve, report


def write_nautilus_catalog(df: pl.DataFrame, catalog_dir: Path) -> None:
    """Write BTCUSDT 1h data to NautilusTrader-compatible catalog."""
    writer = CatalogWriter(catalog_dir)
    writer.write_bars(
        df,
        instrument_id="BTCUSDT.BINANCE",
        bar_type="BTCUSDT.BINANCE-1-HOUR-LAST-EXTERNAL",
    )

    # Write minimal instrument definition
    instruments = [
        {
            "id": "BTCUSDT.BINANCE",
            "raw_symbol": "BTCUSDT",
            "asset_class": "CRYPTO",
            "instrument_class": "SPOT",
            "price_precision": 2,
            "size_precision": 6,
            "price_increment": 0.01,
            "size_increment": 0.000001,
            "maker_fee": 0.001,
            "taker_fee": 0.001,
            "margin_init": 1.0,
            "margin_maint": 1.0,
            "multiplier": 1.0,
        }
    ]
    writer.write_instruments(instruments)


def main() -> None:
    """Run backtest suite."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 2 · Evaluation Harness")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────
    data_path = settings.raw_dir / "klines" / "btcusdt_1h"
    if not data_path.exists():
        logger.error(f"Data not found: {data_path}. Run fetch_btc_data.py first.")
        sys.exit(1)

    df = (
        pl.scan_parquet(str(data_path / "**" / "*.parquet"))
        .sort("timestamp_ms")
        .collect()
    )

    logger.info(f"Loaded {len(df):,} BTCUSDT 1h candles")
    logger.info(
        f"  Range: {datetime.fromtimestamp(df['timestamp_ms'].min()/1000, tz=timezone.utc).date()} → "
        f"{datetime.fromtimestamp(df['timestamp_ms'].max()/1000, tz=timezone.utc).date()}"
    )

    # ── Write NautilusTrader catalog ─────────────────────────
    catalog_dir = settings.processed_dir / "nautilus_catalog"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    write_nautilus_catalog(df, catalog_dir)
    logger.info(f"NautilusTrader catalog written to: {catalog_dir}")

    # ── Benchmark: Buy & Hold ────────────────────────────────
    logger.info("\n--- Benchmark: Buy & Hold ---")
    bh_report = run_buy_and_hold(df)
    bh_report.print()

    # ── Strategy: EMA Crossover ──────────────────────────────
    logger.info("\n--- Strategy: EMA Crossover (20/50) ---")
    equity_curve, ema_report = run_ema_crossover(df)
    ema_report.print()

    # ── Comparison ───────────────────────────────────────────
    logger.info(f"\n{'='*50}")
    logger.info("Strategy Comparison")
    logger.info(f"{'='*50}")
    logger.info(f"{'Metric':<25} {'Buy&Hold':>10} {'EMA 20/50':>10}")
    logger.info(f"{'-'*25} {'-'*10} {'-'*10}")
    bh = bh_report.to_dict()
    em = ema_report.to_dict()
    for key in ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "calmar_ratio"]:
        bh_val = bh.get(key, "N/A")
        em_val = em.get(key, "N/A")
        logger.info(f"{key:<25} {bh_val:>10} {em_val:>10}")

    logger.info("=" * 60)
    logger.info("Sprint 2 — Evaluation Harness COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
