"""Dynamic Grid Trading (DGT) — arXiv 2506.11921 (June 2025).

"Dynamic Grid Trading Strategy: From Zero Expectation to Market Outperformance"

Core algorithm:
  1. Initial grid: N levels spaced by volatility-based interval above/below price
  2. When price crosses a grid level, wait for REBOUND confirmation (2-step trigger)
  3. Execute trade, dynamically reset grid around new reference price
  4. Continuously adapt — no static boundaries that break on large moves

Reported results (BTC/ETH, 2021-2024):
  - DGT drawdown ~50% vs B&H 80% on BTC
  - Frequently matches or exceeds B&H returns in bull markets
  - 30% smaller drawdowns in bear markets
  - Better risk-adjusted returns across regimes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from abundance.backtesting.costs import COST_MODEL
from abundance.backtesting.metrics import MetricsCalculator, MetricsReport
from abundance.config.settings import settings


def run_strategy(
    pair: str = "BTCUSDT",
    grid_levels: int = 10,            # number of grid levels
    grid_spacing_atr_mult: float = 1.5,  # spacing = ATR × this
    atr_period: int = 14,             # ATR lookback
    rebound_pct: float = 0.5,         # rebound confirmation (% of grid spacing)
    base_position_pct: float = 0.10,  # base allocation per grid level
    initial_capital: float = 10_000.0,
) -> tuple[pl.DataFrame, MetricsReport, dict]:
    """Dynamic Grid Trading strategy.

    The grid dynamically resets after every trade — unlike static grids
    that break when price leaves the initial range.

    Returns: (equity_curve, full_report, regime_metrics)
    """
    pair_lower = pair.lower()
    cost = COST_MODEL

    # Load daily data
    df = (
        pl.scan_parquet(
            str(settings.raw_dir / "klines" / f"{pair_lower}_1d" / "**" / "*.parquet")
        )
        .sort("timestamp_ms")
        .collect()
    )

    close = df["close"].to_list()
    high = df["high"].to_list()
    low = df["low"].to_list()
    timestamps = df["timestamp_ms"].to_list()
    n = len(close)

    # Compute ATR (daily)
    atr_vals = [0.0] * n
    for i in range(n):
        tr = high[i] - low[i]
        if i > 0:
            tr = max(tr, abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_vals[i] = tr
    atr_smooth = [0.0] * n
    for i in range(n):
        start = max(0, i - atr_period + 1)
        window = atr_vals[start:i+1]
        atr_smooth[i] = sum(window) / len(window)

    # ── DGT Simulation ──────────────────────────────────────
    capital = initial_capital
    base_quote = capital * 0.5  # half in quote, half will be in base
    base_holdings = 0.0
    equity_curve = [(timestamps[0], capital)]
    trades_list = []

    # Grid state
    reference_price = close[0]
    grid_spacing = atr_smooth[0] * grid_spacing_atr_mult
    pending_buy = 0.0   # quote reserved for buys
    pending_sell = 0.0  # base reserved for sells

    for i in range(atr_period + 1, n):
        price = close[i]
        current_atr = atr_smooth[i]

        # Update grid spacing based on current volatility
        grid_spacing = current_atr * grid_spacing_atr_mult

        # ── Two-step trigger mechanism ────────────────────
        # Step 1: Price crosses a grid level
        # Step 2: Price rebounds by rebound_pct * grid_spacing in opposite direction

        if price <= reference_price - grid_spacing:
            # Crossed below — potential buy
            # Check rebound: price came back up from the low
            recent_low = min(low[max(0, i-3):i+1])
            rebound = price - recent_low
            if rebound >= rebound_pct * grid_spacing:
                # Confirmed buy signal
                buy_amount = base_position_pct * capital if capital > 0 else 0
                if buy_amount > 0:
                    base_bought = buy_amount / price
                    # Apply entry cost
                    entry_cost = cost.entry_cost(pair) * buy_amount
                    base_holdings += base_bought
                    base_quote -= buy_amount - entry_cost  # remaining goes to capital
                    capital = base_quote + base_holdings * price
                    trades_list.append({
                        "pnl": -entry_cost,
                        "return_pct": -entry_cost / buy_amount * 100,
                    })
                # Reset reference to current price
                reference_price = price

        elif price >= reference_price + grid_spacing:
            # Crossed above — potential sell
            recent_high = max(high[max(0, i-3):i+1])
            pullback = recent_high - price
            if pullback >= rebound_pct * grid_spacing:
                # Confirmed sell signal
                sell_amount = min(base_position_pct * base_holdings, base_holdings)
                if sell_amount > 0 and price > 0:
                    quote_received = sell_amount * price
                    exit_cost = cost.exit_cost(pair) * quote_received
                    base_holdings -= sell_amount
                    base_quote += quote_received - exit_cost
                    capital = base_quote + base_holdings * price
                    trades_list.append({
                        "pnl": -exit_cost,
                        "return_pct": 0,  # grid trades accumulate
                    })
                reference_price = price

        # ── Dynamic grid reset on large moves ──────────────
        # If price moved more than 5 grid levels away, force-reset grid
        if abs(price - reference_price) > 5 * grid_spacing:
            # Re-center: close all positions and restart grid
            if base_holdings > 0:
                # Sell all
                quote_received = base_holdings * price
                exit_cost = cost.exit_cost(pair) * quote_received
                base_quote += quote_received - exit_cost
                base_holdings = 0.0

            # Reset
            reference_price = price
            base_quote = base_quote  # preserve capital
            capital = base_quote
            trades_list.append({
                "pnl": -exit_cost if base_holdings > 0 else 0,
                "return_pct": 0,
            })

        # ── Record equity ──────────────────────────────────
        capital = base_quote + base_holdings * price
        equity_curve.append((timestamps[i], capital))

    # ── Metrics ────────────────────────────────────────────
    equity_df = pl.DataFrame(equity_curve, schema=["timestamp_ms", "equity"], orient="row")
    trades_df = pl.DataFrame([t for t in trades_list if t["return_pct"] != 0]) if trades_list else None
    full_report = MetricsCalculator.from_equity_curve(equity_df, trades_df)

    # Regime decomposition
    regimes = {
        "2021 Bull": (1609459200000, 1640995200000),
        "2022 Bear": (1640995200000, 1672531200000),
        "2023 Sideways": (1672531200000, 1704067200000),
        "2024 Bull": (1704067200000, 1735689600000),
        "2025 YTD": (1735689600000, 9999999999999),
        "Full Period": (0, 9999999999999),
    }
    regime_metrics = {}
    for name, (s, e) in regimes.items():
        mask = [(equity_curve[i][0] >= s and equity_curve[i][0] < e)
                for i in range(len(equity_curve))]
        w_ts = [equity_curve[i][0] for i in range(len(equity_curve)) if mask[i]]
        w_eq = [equity_curve[i][1] for i in range(len(equity_curve)) if mask[i]]
        if not w_eq:
            continue
        norm = [w_eq[i] / w_eq[0] * 10000 for i in range(len(w_eq))]
        w_df = pl.DataFrame({"timestamp_ms": w_ts, "equity": pl.Series(norm)})
        r = MetricsCalculator.from_equity_curve(w_df)
        regime_metrics[name] = {
            "sharpe": round(r.sharpe_ratio, 3),
            "return_pct": round(r.total_return_pct, 1),
            "max_dd": round(r.max_drawdown_pct, 1),
        }

    return equity_df, full_report, regime_metrics
