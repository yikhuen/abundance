"""He, Manela, Ross & von Wachter (2022/2024) — No-Arbitrage Perp Strategy.

Paper: "Fundamentals of Perpetual Futures" (arXiv:2212.06888)
Reported Sharpe: 1.8 (retail) to 3.5 (market maker) on BTC.

Strategy: Compute theoretical no-arbitrage perpetual price from spot price
and funding rate. When actual perp price deviates from this theoretical price
beyond a threshold, enter a delta-neutral position to capture the convergence.

Formula: F_theoretical = S × (1 + r×t) / (1 + f×t)
  - S = spot price
  - r = risk-free rate (annualized)
  - f = funding rate (annualized, from 8h periodic rate)
  - t = time fraction (8h/8760h ≈ 0.000913)

Position: delta-neutral (short perp + long spot when F > theoretical)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from abundance.backtesting.costs import COST_MODEL
from abundance.backtesting.metrics import MetricsCalculator, MetricsReport
from abundance.config.settings import settings


def annualize_funding(rate_8h_pct: float) -> float:
    """Convert 8h funding rate percentage to annualised.

    Binance funding interval: 8 hours = 3×/day = 1095×/year.
    rate_8h_pct is e.g. 0.01 (meaning 0.01% per 8h period).
    """
    periods_per_year = 365.25 * 3  # 3 funding periods per day
    rate_per_period = rate_8h_pct / 100  # convert pct → decimal
    return (1 + rate_per_period) ** periods_per_year - 1


def theoretical_perp_price(
    spot: float,
    funding_rate_pct: float,
    risk_free: float = 0.04,
    hours_to_funding: float = 8.0,
) -> float:
    """Compute no-arbitrage perpetual futures price.

    He et al. (2022) formula:
      F = S × (1 + r×T) / (1 + f×T)

    where T = hours_to_funding / (365.25 × 24)
    """
    T = hours_to_funding / (365.25 * 24)
    f = funding_rate_pct / 100  # decimal
    r = risk_free

    if 1 + f * T <= 0:
        return spot
    return spot * (1 + r * T) / (1 + f * T)


def run_strategy(
    pair: str = "BTCUSDT",
    entry_threshold_pct: float = 0.05,  # 0.05% deviation to enter
    exit_threshold_pct: float = 0.01,   # 0.01% to exit
    position_size_pct: float = 0.10,
    risk_free: float = 0.04,
) -> tuple[pl.DataFrame, MetricsReport]:
    """He et al. (2022) no-arbitrage perpetual strategy.

    Args:
        pair: Trading pair.
        entry_threshold_pct: Minimum deviation (%) from theoretical to enter.
        exit_threshold_pct: Deviation below which we exit.
        position_size_pct: Fraction of capital per trade.
        risk_free: Annual risk-free rate (4% default).
    """
    pair_lower = pair.lower()
    cost = COST_MODEL

    # Load spot klines
    spot = (
        pl.scan_parquet(
            str(settings.raw_dir / "klines" / f"{pair_lower}_1h" / "**" / "*.parquet")
        )
        .sort("timestamp_ms")
        .select(["timestamp_ms", "close"])
        .collect()
    )

    # Load perp klines
    perp = (
        pl.scan_parquet(
            str(settings.raw_dir / "perp_klines" / f"{pair_lower}_1h" / "**" / "*.parquet")
        )
        .sort("timestamp_ms")
        .select(["timestamp_ms", "close"])
        .collect()
    )

    # Load funding rates
    funding = (
        pl.scan_parquet(
            str(settings.raw_dir / "funding" / pair_lower / "**" / "*.parquet")
        )
        .sort("timestamp_ms")
        .collect()
    )

    # Align data: build sorted lookup arrays, filter nulls
    spot_raw = [(t, c) for t, c in zip(spot["timestamp_ms"].to_list(), spot["close"].to_list()) if t is not None]
    spot_ts_arr = [x[0] for x in spot_raw]
    spot_close_arr = [x[1] for x in spot_raw]
    perp_raw = [(t, c) for t, c in zip(perp["timestamp_ms"].to_list(), perp["close"].to_list()) if t is not None]
    perp_ts_arr = [x[0] for x in perp_raw]
    perp_close_arr = [x[1] for x in perp_raw]

    def nearest(arr_ts, arr_val, target):
        """Binary-search nearest value with ts ≤ target."""
        lo, hi = 0, len(arr_ts) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if arr_ts[mid] is not None and arr_ts[mid] <= target:
                best = arr_val[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best if best is not None else 0.0

    funding_list = funding["funding_rate_pct"].to_list()
    funding_ts_list = funding["timestamp_ms"].to_list()

    # Build aligned time series at funding intervals (most informative)
    capital = 10_000.0
    equity = [(funding_ts_list[0], capital)]
    trades_list = []
    in_position = False
    pos_capital = 0.0
    pos_entry_perp = 0.0
    pos_entry_spot = 0.0
    pos_type = ""  # "long_perp_short_spot" or "short_perp_long_spot"

    for i in range(1, len(funding_ts_list)):
        ts = funding_ts_list[i]
        rate = funding_list[i - 1]  # previous period's rate (no lookahead)

        # Get spot and perp prices nearest to this funding timestamp
        spot_price = nearest(spot_ts_arr, spot_close_arr, ts)
        perp_price = nearest(perp_ts_arr, perp_close_arr, ts)
        if spot_price is None or perp_price is None or spot_price <= 0 or perp_price <= 0:
            continue

        # Compute theoretical price
        theory = theoretical_perp_price(spot_price, rate, risk_free)

        # Deviation: actual vs theoretical (as percentage)
        deviation_pct = (perp_price - theory) / theory * 100

        # ── Exit logic ──────────────────────────────────
        if in_position:
            if abs(deviation_pct) < exit_threshold_pct:
                # Close: convergence achieved
                spot_pnl = 0.0
                if pos_type == "short_perp_long_spot":
                    # Short perp: profit when perp drops relative to spot
                    perp_pnl = (pos_entry_perp - perp_price) / pos_entry_perp * pos_capital
                    spot_pnl = (spot_price / pos_entry_spot - 1) * pos_capital
                    gross = perp_pnl + spot_pnl
                else:
                    # Long perp, short spot
                    perp_pnl = (perp_price / pos_entry_perp - 1) * pos_capital
                    spot_pnl = (1 - spot_price / pos_entry_spot) * pos_capital
                    gross = perp_pnl + spot_pnl

                rt_cost = cost.round_trip_cost(pair, use_perp=True) * pos_capital
                net_pnl = gross - rt_cost
                capital += net_pnl
                trades_list.append({"pnl": net_pnl, "return_pct": net_pnl / pos_capital * 100})
                in_position = False

        # ── Entry logic ──────────────────────────────────
        if not in_position and abs(deviation_pct) > entry_threshold_pct:
            pos_capital = capital * position_size_pct
            pos_entry_perp = perp_price
            pos_entry_spot = spot_price
            if deviation_pct > 0:
                pos_type = "short_perp_long_spot"
            else:
                pos_type = "long_perp_short_spot"
            in_position = True

        # ── Record equity ────────────────────────────────
        eq_val = capital
        if in_position and pos_entry_perp > 0:
            perp_delta = (perp_price / pos_entry_perp - 1) * pos_capital
            if pos_type == "short_perp_long_spot":
                perp_delta = -perp_delta
            eq_val = capital + perp_delta
        equity.append((ts, eq_val))

    equity_df = pl.DataFrame(equity, schema=["timestamp_ms", "equity"], orient="row")
    trades_df = pl.DataFrame(trades_list) if trades_list else None
    report = MetricsCalculator.from_equity_curve(equity_df, trades_df)

    return equity_df, report


def _nearest(arr_ts, arr_val, ts: int) -> float:
    """Binary-search nearest value with key ≤ ts."""
    lo, hi = 0, len(arr_ts) - 1
    best = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr_ts[mid] <= ts:
            best = arr_val[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best
