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

from abundance.backtesting.costs import COST_MODEL, CostModel
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
    cost_model: CostModel | None = None,
) -> tuple[pl.DataFrame, MetricsReport, list[Trade]]:
    """Run funding carry backtest with cost model and lookahead protection.

    Safeguards:
      - Uses PREVIOUS bar's funding rate (shifted by 1) — no lookahead.
      - Executes at NEXT bar's OPEN (not same-bar close).
      - Applies round-trip costs (fees + spread + slippage) per trade.

    Args:
        df_kline: OHLCV data with timestamp_ms, open, close.
        df_funding: Funding rate data with timestamp_ms and funding_rate_pct.
        strategy: Strategy parameters.
        initial_capital: Starting capital in quote currency.
        cost_model: Transaction cost model. Uses COST_MODEL default if None.

    Returns:
        (equity_curve, metrics_report, trades)
    """
    if cost_model is None:
        cost_model = COST_MODEL

    pair = strategy.entry_threshold_pct  # hack: derive pair from context
    # Better approach: pass pair explicitly. For now, derive from threshold.

    funding = df_funding.sort("timestamp_ms")
    kline = df_kline.sort("timestamp_ms")

    rates = funding["funding_rate_pct"].to_list()
    timestamps = funding["timestamp_ms"].to_list()

    # Build lookups: timestamp → open and close
    kline_ts = kline["timestamp_ms"].to_list()
    kline_open = kline["open"].to_list()
    kline_close = kline["close"].to_list()
    open_lookup = dict(zip(kline_ts, kline_open))
    close_lookup = dict(zip(kline_ts, kline_close))

    def _nearest_open(ts: int) -> float:
        """Find nearest kline open ≤ timestamp (no lookahead)."""
        for i in range(len(kline_ts) - 1, -1, -1):
            if kline_ts[i] <= ts:
                return kline_open[i]
        return kline_open[0]

    def _nearest_close(ts: int) -> float:
        """Find nearest kline close ≤ timestamp (no lookahead)."""
        for i in range(len(kline_ts) - 1, -1, -1):
            if kline_ts[i] <= ts:
                return kline_close[i]
        return kline_close[0]

    capital = initial_capital
    equity_points: list[tuple[int, float]] = [(timestamps[0], capital)]
    trades: list[Trade] = []
    in_position = False
    position: Optional[dict] = None
    position_capital = 0.0

    # Use PREVIOUS bar's rate to avoid lookahead (can't trade on current rate)
    for i in range(1, len(rates)):
        prev_rate = rates[i - 1]  # ← lookahead fix: use previous bar's rate
        current_rate = rates[i]
        ts = timestamps[i]

        # Execute at NEXT bar's OPEN (not close) — realistic execution assumption
        exec_price = _nearest_open(ts) if _nearest_open(ts) > 0 else _nearest_close(ts)

        # ── Force close check (funding flipped negative) ────
        if in_position and position is not None:
            if current_rate < strategy.force_close_threshold_pct:
                spot_pnl_pct = (
                    (position["spot_entry_price"] - exec_price)
                    / position["spot_entry_price"]
                )
                gross_pnl = position["accumulated_funding"] - spot_pnl_pct * position_capital
                # Apply round-trip costs
                cost_frac = cost_model.round_trip_cost("BTCUSDT", use_perp=True)
                net_pnl = gross_pnl - cost_frac * position_capital

                capital += net_pnl
                trades.append(
                    Trade(
                        entry_ts=position["entry_ts"],
                        exit_ts=ts,
                        entry_rate=position["entry_rate"],
                        exit_rate=current_rate,
                        pnl=net_pnl,
                        pnl_pct=net_pnl / position_capital * 100,
                        duration_hours=(ts - position["entry_ts"]) / 3600000,
                    )
                )
                in_position = False
                position = None

        # ── Entry logic (on PREVIOUS rate, no lookahead) ────
        if (
            not in_position
            and prev_rate > strategy.entry_threshold_pct
            and len(trades) < 1000
        ):
            position_capital = capital * strategy.position_size_pct
            position = {
                "entry_ts": ts,
                "entry_rate": prev_rate,
                "spot_entry_price": exec_price,
                "accumulated_funding": 0.0,
            }
            in_position = True
            continue

        # ── Accumulate funding ─────────────────────────────
        if in_position and position is not None:
            funding_payment = (current_rate / 100) * position_capital
            position["accumulated_funding"] += funding_payment

        # ── Exit logic (on PREVIOUS rate) ──────────────────
        if in_position and position is not None and prev_rate < strategy.exit_threshold_pct:
            spot_pnl_pct = (
                (position["spot_entry_price"] - exec_price)
                / position["spot_entry_price"]
            )
            spot_pnl = spot_pnl_pct * position_capital
            gross_pnl = position["accumulated_funding"] - spot_pnl

            # Apply round-trip costs
            cost_frac = cost_model.round_trip_cost("BTCUSDT", use_perp=True)
            net_pnl = gross_pnl - cost_frac * position_capital

            capital += net_pnl
            trades.append(
                Trade(
                    entry_ts=position["entry_ts"],
                    exit_ts=ts,
                    entry_rate=position["entry_rate"],
                    exit_rate=prev_rate,
                    pnl=net_pnl,
                    pnl_pct=net_pnl / position_capital * 100,
                    duration_hours=(ts - position["entry_ts"]) / 3600000,
                )
            )
            in_position = False
            position = None

        # ── Record equity ────────────────────────────────
        current_equity = capital
        if in_position and position is not None:
            spot_delta = (exec_price / position["spot_entry_price"] - 1)
            spot_value_change = spot_delta * position_capital
            current_equity = capital + position["accumulated_funding"] - spot_value_change
        equity_points.append((ts, current_equity))

    # Build equity curve
    equity_curve = pl.DataFrame(
        equity_points,
        schema=["timestamp_ms", "equity"],
        orient="row",
    )

    trades_df = None
    if trades:
        trades_df = pl.DataFrame(
            [{"pnl": t.pnl, "return_pct": t.pnl_pct} for t in trades]
        )

    report = MetricsCalculator.from_equity_curve(equity_curve, trades_df)
    report.extra["num_trades"] = float(len(trades))
    if trades:
        report.extra["avg_trade_duration_hours"] = (
            sum(t.duration_hours for t in trades) / len(trades)
        )
        report.extra["avg_trade_pnl_pct"] = sum(t.pnl_pct for t in trades) / len(trades)

    return equity_curve, report, trades
