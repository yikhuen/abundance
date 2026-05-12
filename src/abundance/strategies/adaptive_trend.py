"""AdaptiveTrend-inspired multi-asset trend-following strategy.

Based on: "Systematic Trend-Following with Adaptive Portfolio Construction"
(arXiv:2602.11708, Feb 2026). Paper reports Sharpe 2.41, MaxDD -12.7%.

Adapted for our 3-asset universe (BTC/ETH/SOL):
  1. Trend signal: 4h EMA crossovers (50/200) — bullish if fast > slow
  2. Dynamic trailing stop: 3× ATR below peak
  3. Adaptive allocation: higher weight to assets with higher rolling Sharpe
  4. Monthly rebalancing
  5. Cross-regime testing: bull (2020-21, 2024), bear (2022), sideways (2023)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from abundance.backtesting.costs import COST_MODEL
from abundance.backtesting.metrics import MetricsCalculator, MetricsReport
from abundance.config.settings import settings

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
TIMEFRAME = "4h"  # closest to paper's 6h
ATR_PERIOD = 20
EMA_FAST = 50
EMA_SLOW = 200
STOP_MULT = 3.0
REBALANCE_DAYS = 30  # monthly
SHARPE_WINDOW = 90  # periods for rolling Sharpe


def compute_regime_metrics(
    df: pl.DataFrame, regime_name: str
) -> dict:
    """Compute strategy metrics for a specific market regime."""
    if df.is_empty():
        return {"sharpe": 0, "return_pct": 0, "max_dd": 0}

    report = MetricsCalculator.from_equity_curve(
        df.select(["timestamp_ms", "equity"])
    )
    return {
        "sharpe": round(report.sharpe_ratio, 3),
        "return_pct": round(report.total_return_pct, 1),
        "max_dd": round(report.max_drawdown_pct, 1),
        "calmar": round(report.calmar_ratio, 3),
    }


def run_strategy(
    initial_capital: float = 10_000.0,
    rebalance_days: int = REBALANCE_DAYS,
) -> tuple[pl.DataFrame, MetricsReport, dict]:
    """Multi-asset adaptive trend-following with cross-regime testing.

    Returns:
        (equity_curve, full_report, regime_metrics)
    """
    cost = COST_MODEL
    all_equity_points = []

    # Load and prepare data for each pair
    pair_data = {}
    for pair in PAIRS:
        plower = pair.lower()
        df = (
            pl.scan_parquet(
                str(settings.raw_dir / "klines" / f"{plower}_{TIMEFRAME}" / "**" / "*.parquet")
            )
            .sort("timestamp_ms")
            .collect()
        )
        pair_data[pair] = {
            "timestamps": df["timestamp_ms"].to_list(),
            "close": df["close"].to_list(),
            "high": df["high"].to_list(),
            "low": df["low"].to_list(),
        }

    # Find common time range (SOL listed ~2020-08)
    min_ts = max(
        min(pair_data["SOLUSDT"]["timestamps"]),
        min(pair_data["BTCUSDT"]["timestamps"]),
        min(pair_data["ETHUSDT"]["timestamps"]),
    )
    max_ts = min(
        max(pair_data["SOLUSDT"]["timestamps"]),
        max(pair_data["BTCUSDT"]["timestamps"]),
        max(pair_data["ETHUSDT"]["timestamps"]),
    )

    # Align all pairs to common timestamps (daily for rebalancing)
    # Use 1d data for rebalancing decisions
    daily_data = {}
    for pair in PAIRS:
        plower = pair.lower()
        df = (
            pl.scan_parquet(
                str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
            )
            .sort("timestamp_ms")
            .collect()
        )
        daily_data[pair] = {
            "ts": df["timestamp_ms"].to_list(),
            "close": df["close"].to_list(),
        }

    # ── Compute signals per pair ──────────────────────────
    signals = {}
    for pair in PAIRS:
        close = pair_data[pair]["close"]
        high = pair_data[pair]["high"]
        low = pair_data[pair]["low"]
        ts = pair_data[pair]["timestamps"]
        n = len(close)

        # EMA trend signal
        fe = [close[0]] * n
        se = [close[0]] * n
        alpha_f = 2 / (EMA_FAST + 1)
        alpha_s = 2 / (EMA_SLOW + 1)
        for i in range(1, n):
            fe[i] = close[i] * alpha_f + fe[i-1] * (1 - alpha_f)
            se[i] = close[i] * alpha_s + se[i-1] * (1 - alpha_s)

        trend_sig = [0.0] * n
        for i in range(EMA_SLOW, n):
            if fe[i] > se[i]:
                trend_sig[i] = 1.0

        # ATR
        atr_vals = [0.0] * n
        for i in range(n):
            tr = high[i] - low[i]
            if i > 0:
                tr = max(tr, abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
            atr_vals[i] = tr
        atr_smooth = [0.0] * n
        for i in range(n):
            start = max(0, i - ATR_PERIOD + 1)
            window = atr_vals[start:i+1]
            atr_smooth[i] = sum(window) / len(window)

        # Trailing stop
        sig = [0.0] * n
        in_pos = False
        peak = 0.0
        for i in range(max(EMA_SLOW, ATR_PERIOD), n):
            if not in_pos and trend_sig[i] > 0:
                sig[i] = 0.33  # equal weight at entry
                in_pos = True
                peak = close[i]
            elif in_pos:
                if close[i] > peak:
                    peak = close[i]
                if close[i] < peak - STOP_MULT * atr_smooth[i]:
                    sig[i] = 0.0
                    in_pos = False
                else:
                    sig[i] = 0.33  # maintain

        signals[pair] = {
            "sig": sig,
            "ts": ts,
            "close": close,
        }

    # ── Build daily returns for adaptive allocation ───────
    pair_returns = {}
    for pair in PAIRS:
        d_close = daily_data[pair]["close"]
        d_ts = daily_data[pair]["ts"]
        rets = [0.0]
        for i in range(1, len(d_close)):
            rets.append(d_close[i] / d_close[i-1] - 1)
        pair_returns[pair] = {"ts": d_ts, "close": d_close, "ret": rets}

    # ── Simulate portfolio ────────────────────────────────
    capital = initial_capital
    allocations = {p: 0.0 for p in PAIRS}
    equity = []
    positions = {p: {"in": False, "capital": 0.0, "entry": 0.0, "peak": 0.0} for p in PAIRS}
    trades_list = []

    # Walk through aligned timestamps
    common_ts = sorted(set(
        t for pair in PAIRS
        for t in daily_data[pair]["ts"]
        if min_ts <= t <= max_ts
    ))

    # Rebalance monthly
    last_rebalance = 0
    rebalance_ms = rebalance_days * 24 * 3600 * 1000

    for ts_idx, ts in enumerate(common_ts):
        day_from_start = ts - common_ts[0]

        # ── Monthly rebalance ──────────────────────────
        if ts_idx == 0 or day_from_start - last_rebalance >= rebalance_ms:
            last_rebalance = day_from_start
            # Adaptive allocation: weight by rolling Sharpe
            sharpe_weights = {}
            total_weight = 0.0
            for pair in PAIRS:
                rets = pair_returns[pair]["ret"]
                # Find returns up to this timestamp
                start_idx = max(0, ts_idx - SHARPE_WINDOW)
                window_rets = rets[max(0, start_idx):ts_idx+1]
                if len(window_rets) > 2:
                    mean_r = sum(window_rets) / len(window_rets)
                    std_r = (sum((r - mean_r)**2 for r in window_rets) / len(window_rets))**0.5
                    if std_r > 0:
                        sharpe_weights[pair] = max(0, mean_r / std_r)
                    else:
                        sharpe_weights[pair] = 0.0
                else:
                    sharpe_weights[pair] = 0.0
                total_weight += sharpe_weights[pair]

            if total_weight > 0:
                for pair in PAIRS:
                    allocations[pair] = sharpe_weights[pair] / total_weight
            else:
                for pair in PAIRS:
                    allocations[pair] = 1.0 / len(PAIRS)

        # ── Update positions ──────────────────────────────
        for pair in PAIRS:
            pos = positions[pair]
            # Get 4h signal nearest to this daily timestamp
            sig_list = signals[pair]["sig"]
            sig_ts = signals[pair]["ts"]
            sig_val = 0.0
            for i in range(len(sig_ts) - 1, -1, -1):
                if sig_ts[i] <= ts:
                    sig_val = sig_list[i]
                    break

            close_list = signals[pair]["close"]
            price = 0.0
            for i in range(len(sig_ts) - 1, -1, -1):
                if sig_ts[i] <= ts:
                    price = close_list[i]
                    break

            if not pos["in"] and sig_val > 0:
                pos["in"] = True
                alloc_capital = capital * allocations[pair] * 0.5
                if alloc_capital <= 0:
                    continue
                pos["capital"] = alloc_capital
                pos["entry"] = price
                pos["peak"] = price
            elif pos["in"] and sig_val == 0:
                # Exit
                if pos["capital"] > 0:
                    pnl = (price / pos["entry"] - 1) * pos["capital"]
                    rt_cost = cost.round_trip_cost(pair) * pos["capital"]
                    net_pnl = pnl - rt_cost
                    capital += net_pnl
                    trades_list.append({"pnl": net_pnl, "return_pct": net_pnl / pos["capital"] * 100})
                pos["in"] = False
                pos["capital"] = 0.0
            elif pos["in"]:
                # Update peak
                if price > pos["peak"]:
                    pos["peak"] = price

        # ── Compute equity ────────────────────────────────
        total_eq = capital
        for pair in PAIRS:
            pos = positions[pair]
            if pos["in"] and pos["entry"] > 0 and pos["capital"] > 0:
                # Get current price
                close_list = signals[pair]["close"]
                sig_ts = signals[pair]["ts"]
                price = 0.0
                for i in range(len(sig_ts) - 1, -1, -1):
                    if sig_ts[i] <= ts:
                        price = close_list[i]
                        break
                if price > 0:
                    total_eq += (price / pos["entry"] - 1) * pos["capital"]
        equity.append((ts, total_eq))

    equity_df = pl.DataFrame(equity, schema=["timestamp_ms", "equity"], orient="row")
    trades_df = pl.DataFrame(trades_list) if trades_list else None
    full_report = MetricsCalculator.from_equity_curve(equity_df, trades_df)

    # ── Cross-regime testing ──────────────────────────────
    regimes = {
        "2018 Bear": (1514764800000, 1546300800000),
        "2020-21 Bull": (1577836800000, 1640995200000),
        "2022 Bear": (1640995200000, 1672531200000),
        "2023 Sideways": (1672531200000, 1704067200000),
        "2024 Bull": (1704067200000, 1735689600000),
        "Full Period": (0, 9999999999999),
    }

    regime_metrics = {}
    for name, (start, end) in regimes.items():
        regime_df = equity_df.filter(
            (pl.col("timestamp_ms") >= start) & (pl.col("timestamp_ms") <= end)
        )
        regime_metrics[name] = compute_regime_metrics(regime_df, name)

    return equity_df, full_report, regime_metrics
