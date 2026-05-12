"""Funding carry strategy.

Takes the short side of perpetual futures when funding rates are
elevated, earning the funding premium while hedging delta with spot.

Strategy logic:
  1. Monitor funding rate every 8h
  2. Entry: funding_rate > entry_threshold → short perp, long spot
  3. Exit: funding_rate < exit_threshold → close both legs
  4. Position size: fixed percentage of capital
  5. Risk limit: force-close if funding flips negative (extreme)

Reference: Hayes, A. (2021). "What Is Crypto Funding Rate Arbitrage?"
"""

from dataclasses import dataclass, field
from typing import Optional

import polars as pl
from loguru import logger

from abundance.backtesting.metrics import MetricsCalculator, MetricsReport


@dataclass
class Trade:
    """Single carry trade record."""

    entry_ts: int  # epoch ms
    exit_ts: int  # epoch ms (0 if still open)
    entry_rate: float  # funding_rate_pct at entry
    exit_rate: float  # funding_rate_pct at exit
    pnl: float  # profit/loss in quote currency
    pnl_pct: float  # PnL as % of capital
    duration_hours: float


@dataclass
class CarryStrategy:
    """Configuration for the funding carry strategy."""

    entry_threshold_pct: float  # enter short when funding > this
    exit_threshold_pct: float  # exit when funding < this
    position_size_pct: float = 0.01  # 1% of capital per trade
    max_positions: int = 1  # max concurrent positions
    force_close_threshold_pct: float = -0.1  # emergency exit

    @classmethod
    def from_pair(cls, pair: str) -> "CarryStrategy":
        """Factory with pair-specific thresholds from Stage 3.2 analysis."""
        thresholds = {
            "BTCUSDT": (0.010, 0.005),
            "ETHUSDT": (0.017, 0.008),
            "SOLUSDT": (0.010, 0.005),
        }
        entry, exit = thresholds.get(pair, (0.010, 0.005))
        return cls(entry_threshold_pct=entry, exit_threshold_pct=exit)


def run_carry_backtest(
    df_kline: pl.DataFrame,
    df_funding: pl.DataFrame,
    strategy: CarryStrategy,
    initial_capital: float = 10_000.0,
) -> tuple[pl.DataFrame, MetricsReport, list[Trade]]:
    """Run funding carry backtest.

    Args:
        df_kline: OHLCV data (used for spot PnL tracking).
        df_funding: Funding rate data with timestamp_ms and funding_rate_pct.
        strategy: Strategy parameters.
        initial_capital: Starting capital in quote currency.

    Returns:
        (equity_curve, metrics_report, trades)

    Strategy: Each funding period, check the rate:
      - If above entry threshold and no position → open short (earn funding)
      - If below exit threshold and in position → close
      - Profit = accumulated funding payments − spot price change
    """
    # Align funding data
    funding = df_funding.sort("timestamp_ms")
    kline = df_kline.sort("timestamp_ms")

    # Build equity curve at funding intervals
    capital = initial_capital
    equity_points: list[tuple[int, float]] = []
    trades: list[Trade] = []
    in_position = False
    position: Optional[dict] = None
    position_capital = 0.0

    funding_rate_series = funding["funding_rate_pct"].to_list()
    funding_ts_series = funding["timestamp_ms"].to_list()

    # Get spot close prices aligned with funding times
    close_lookup = dict(zip(kline["timestamp_ms"].to_list(), kline["close"].to_list()))

    for i, (ts, rate) in enumerate(zip(funding_ts_series, funding_rate_series)):
        # ── Portfolio: update position value ────────────────
        if in_position and position is not None:
            # Check for force close (funding flipped negative enough)
            if rate < strategy.force_close_threshold_pct:
                # Emergency exit — funding flipped, close immediately
                close_price = close_lookup.get(ts, position["spot_entry_price"])
                spot_pnl_pct = (
                    (position["spot_entry_price"] - close_price)
                    / position["spot_entry_price"]
                )
                total_pnl = position["accumulated_funding"] - spot_pnl_pct * position_capital

                capital += total_pnl
                trades.append(
                    Trade(
                        entry_ts=position["entry_ts"],
                        exit_ts=ts,
                        entry_rate=position["entry_rate"],
                        exit_rate=rate,
                        pnl=total_pnl,
                        pnl_pct=total_pnl / position_capital * 100,
                        duration_hours=(ts - position["entry_ts"]) / 3600000,
                    )
                )
                in_position = False
                position = None

        # ── Entry logic ────────────────────────────────────
        if (
            not in_position
            and rate > strategy.entry_threshold_pct
            and len(trades) < 1000  # sanity cap
        ):
            position_capital = capital * strategy.position_size_pct
            spot_entry_price = close_lookup.get(ts, kline["close"].last())

            position = {
                "entry_ts": ts,
                "entry_rate": rate,
                "spot_entry_price": spot_entry_price,
                "accumulated_funding": 0.0,
            }
            in_position = True
            continue

        # ── Accumulate funding ─────────────────────────────
        if in_position and position is not None:
            # Short side earns funding when rate > 0
            # Funding payment = rate% * position_size * 8h-period-weight
            funding_payment = (rate / 100) * position_capital
            position["accumulated_funding"] += funding_payment

        # ── Exit logic ─────────────────────────────────────
        if in_position and position is not None and rate < strategy.exit_threshold_pct:
            # Close position
            exit_price = close_lookup.get(ts, position["spot_entry_price"])
            # Short perp: profit = spot_entry - exit (we shorted perp long spot)
            spot_pnl_pct = (
                (position["spot_entry_price"] - exit_price)
                / position["spot_entry_price"]
            )
            spot_pnl = spot_pnl_pct * position_capital
            total_pnl = position["accumulated_funding"] - spot_pnl

            capital += total_pnl
            trades.append(
                Trade(
                    entry_ts=position["entry_ts"],
                    exit_ts=ts,
                    entry_rate=position["entry_rate"],
                    exit_rate=rate,
                    pnl=total_pnl,
                    pnl_pct=total_pnl / position_capital * 100,
                    duration_hours=(ts - position["entry_ts"]) / 3600000,
                )
            )
            in_position = False
            position = None

        # ── Record equity ────────────────────────────────
        equity_points.append((ts, capital))

        # Update equity with spot PnL on remaining capital when in position
        if in_position and position is not None:
            spot_price = close_lookup.get(ts, position["spot_entry_price"])
            spot_delta = (spot_price / position["spot_entry_price"] - 1)
            spot_value_change = spot_delta * position_capital
            equity_points[-1] = (
                ts,
                capital + position["accumulated_funding"] - spot_value_change,
            )

    # Build equity curve DataFrame
    if not equity_points:
        equity_curve = pl.DataFrame(
            {"timestamp_ms": [], "equity": []},
            schema={"timestamp_ms": pl.Int64, "equity": pl.Float64},
        )
    else:
        equity_curve = pl.DataFrame(
            equity_points,
            schema=["timestamp_ms", "equity"],
            orient="row",
        )

    # Build trades DataFrame for metrics
    if trades:
        trades_df = pl.DataFrame(
            [
                {
                    "pnl": t.pnl,
                    "return_pct": t.pnl_pct,
                }
                for t in trades
            ]
        )
    else:
        trades_df = None

    report = MetricsCalculator.from_equity_curve(equity_curve, trades_df)

    # Attach strategy-specific metrics
    report.extra["num_trades"] = float(len(trades))
    if trades:
        avg_duration = sum(t.duration_hours for t in trades) / len(trades)
        report.extra["avg_trade_duration_hours"] = avg_duration
        report.extra["avg_trade_pnl_pct"] = sum(t.pnl_pct for t in trades) / len(trades)

    return equity_curve, report, trades
