"""Cross-Sectional Momentum with Adaptive Portfolio Construction.

Based on: "Systematic Trend-Following with Adaptive Portfolio Construction"
(Nguyen-Van, arXiv:2602.11708, Feb 2026)

Key algorithm components (from paper):
  1. Cross-sectional ranking: compute N-day momentum for ALL assets,
     go long top 30%, go short bottom 30% (not time-series per asset)
  2. Asymmetric 70/30 long-short: 70% capital longs, 30% shorts
  3. Monthly rebalancing with rolling-Sharpe filtering
  4. Dynamic trailing stop: 2× ATR, calibrated per asset
  5. Volume/market-cap filtering: skip illiquid pairs

Why cross-sectional beats time-series:
  - Time-series: "is BTC trending up?" (directional bet)
  - Cross-sectional: "which of 13 assets has strongest trend right now?"
    → market-neutral, captures relative strength
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl

from abundance.backtesting.costs import COST_MODEL
from abundance.backtesting.metrics import MetricsCalculator, MetricsReport
from abundance.config.settings import settings


def run_strategy(
    momentum_days: int = 60,       # N-day momentum for ranking
    top_fraction: float = 0.30,    # top 30% go long
    bottom_fraction: float = 0.30, # bottom 30% go short
    long_weight: float = 0.70,     # 70% capital to longs
    short_weight: float = 0.30,    # 30% capital to shorts
    rebalance_days: int = 30,      # monthly rebalancing
    atr_period: int = 14,          # ATR for trailing stop
    stop_mult: float = 2.0,        # stop = peak - 2×ATR
    vol_filter_pct: float = 2.0,  # skip assets with vol > 200% annualized (crypto-appropriate)
) -> tuple[pl.DataFrame, MetricsReport, dict]:
    """Cross-sectional momentum strategy per AdaptiveTrend paper.

    Returns: (equity_curve, full_report, regime_metrics)
    """
    cost = COST_MODEL

    # Auto-discover pairs
    klines_dir = settings.raw_dir / "klines"
    pairs = sorted([
        d.name.replace("_1d", "").upper()
        for d in klines_dir.iterdir()
        if d.is_dir() and d.name.endswith("_1d")
    ])

    # Load daily data
    daily_data = {}
    for pair in pairs:
        plower = pair.lower()
        try:
            df = (
                pl.scan_parquet(
                    str(klines_dir / f"{plower}_1d" / "**" / "*.parquet")
                )
                .sort("timestamp_ms")
                .collect()
            )
            if len(df) > 100:  # minimum 100 days
                daily_data[pair] = {
                    "ts": df["timestamp_ms"].to_list(),
                    "close": df["close"].to_list(),
                    "high": df["high"].to_list(),
                    "low": df["low"].to_list(),
                    "volume": df["volume"].to_list() if "volume" in df.columns else [1]*len(df),
                }
        except Exception:
            continue

    pairs = sorted(daily_data.keys())
    n_pairs = len(pairs)
    n_long = max(1, int(n_pairs * top_fraction))
    n_short = max(1, int(n_pairs * bottom_fraction))

    # Find common start. End handled dynamically (pairs drop out).
    min_ts = max(min(daily_data[p]["ts"]) for p in pairs)

    # Build common timestamp grid (all unique days, handle pair expiry)
    all_ts = set()
    for p in pairs:
        for t in daily_data[p]["ts"]:
            if t >= min_ts:
                all_ts.add(t)
    common_ts = sorted(all_ts)

    # ── Main simulation ──────────────────────────────────────
    capital = 10_000.0
    equity_curve = [(common_ts[0], capital)]
    trades_list = []
    positions = {}  # pair → {"direction": "long"/"short", "entry": price, "peak": price, "capital": $}
    last_rebalance = common_ts[0]
    rebalance_ms = rebalance_days * 24 * 3600 * 1000

    for ts_idx, ts in enumerate(common_ts):
        # ── Monthly Rebalance ─────────────────────────────
        if ts_idx == 0 or ts - last_rebalance >= rebalance_ms:
            last_rebalance = ts
            rebalance = True
        else:
            rebalance = False

        if rebalance:
            # Step 1: Compute N-day momentum for each asset (skip expired)
            momentum_scores = {}
            for pair in pairs:
                if pair not in daily_data:
                    continue
                close = daily_data[pair]["close"]
                ts_arr = daily_data[pair]["ts"]
                if ts_arr[-1] < ts:  # pair data ended
                    continue
                vol = daily_data[pair]["volume"]
                # Find current price and price N days ago
                idx_now = next(
                    (i for i in range(len(ts_arr) - 1, -1, -1) if ts_arr[i] <= ts),
                    -1,
                )
                idx_past = next(
                    (i for i in range(len(ts_arr) - 1, -1, -1)
                     if ts_arr[i] <= ts - momentum_days * 24 * 3600 * 1000),
                    -1,
                )
                if idx_now < 0 or idx_past < 0:
                    continue
                mom = (close[idx_now] / close[idx_past] - 1) * 100  # percentage

                # Volume filter: skip if avg volume too low
                vol_start = max(0, idx_now - 30)
                avg_vol = sum(vol[vol_start:idx_now+1]) / max(1, idx_now - vol_start + 1)
                if avg_vol < 1:  # essentially no volume data
                    continue

                # Volatility filter: skip extreme vol
                ret_window = [
                    close[i] / close[i-1] - 1
                    for i in range(max(1, idx_now - 30), idx_now + 1)
                    if close[i-1] > 0
                ]
                if len(ret_window) > 5:
                    daily_vol = (
                        sum((r - sum(ret_window)/len(ret_window))**2 for r in ret_window)
                        / len(ret_window)
                    ) ** 0.5
                    ann_vol = daily_vol * (365 ** 0.5) * 100
                    if ann_vol > vol_filter_pct * 100:
                        continue

                momentum_scores[pair] = mom

            # Step 2: Rank and select
            ranked = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
            longs = [p for p, _ in ranked[:n_long]]
            shorts = [p for p, _ in ranked[-n_short:]]

            # Step 3: Allocate capital
            # Close positions not in new selection
            for pair in list(positions.keys()):
                if pair not in longs and pair not in shorts:
                    pos = positions[pair]
                    if pos["capital"] > 0 and pos["entry"] > 0:
                        # Get current price
                        close_arr = daily_data[pair]["close"]
                        ts_arr = daily_data[pair]["ts"]
                        price = 0.0
                        for i in range(len(ts_arr) - 1, -1, -1):
                            if ts_arr[i] <= ts:
                                price = close_arr[i]
                                break
                        if price > 0 and pos["entry"] > 0:
                            if pos["direction"] == "long":
                                pnl = (price / pos["entry"] - 1) * pos["capital"]
                            else:
                                pnl = (pos["entry"] / price - 1) * pos["capital"]
                            rt_cost = cost.round_trip_cost(pair) * pos["capital"]
                            net = pnl - rt_cost
                            capital += net
                            trades_list.append({
                                "pnl": net,
                                "return_pct": net / pos["capital"] * 100,
                            })
                    del positions[pair]

            # Open new positions
            per_long_cap = capital * long_weight / max(1, len(longs))
            per_short_cap = capital * short_weight / max(1, len(shorts))

            for pair in longs:
                if pair not in positions and per_long_cap > 0:
                    close_arr = daily_data[pair]["close"]
                    ts_arr = daily_data[pair]["ts"]
                    price = 0.0
                    for i in range(len(ts_arr) - 1, -1, -1):
                        if ts_arr[i] <= ts:
                            price = close_arr[i]; break
                    if price > 0:
                        positions[pair] = {
                            "direction": "long",
                            "entry": price,
                            "peak": price,
                            "capital": per_long_cap,
                        }

            for pair in shorts:
                if pair not in positions and per_short_cap > 0:
                    close_arr = daily_data[pair]["close"]
                    ts_arr = daily_data[pair]["ts"]
                    price = 0.0
                    for i in range(len(ts_arr) - 1, -1, -1):
                        if ts_arr[i] <= ts:
                            price = close_arr[i]; break
                    if price > 0:
                        positions[pair] = {
                            "direction": "short",
                            "entry": price,
                            "peak": price,
                            "capital": per_short_cap,
                        }

        # ── Daily: update stops and check exits ─────────────
        for pair in list(positions.keys()):
            pos = positions[pair]
            close_arr = daily_data[pair]["close"]
            ts_arr = daily_data[pair]["ts"]
            price = 0.0
            for i in range(len(ts_arr) - 1, -1, -1):
                if ts_arr[i] <= ts:
                    price = close_arr[i]; break
            if price <= 0:
                continue

            # Update peak
            if pos["direction"] == "long" and price > pos["peak"]:
                pos["peak"] = price
            elif pos["direction"] == "short" and price < pos["peak"]:
                pos["peak"] = price

            # Compute ATR for trailing stop
            high_arr = daily_data[pair]["high"]
            low_arr = daily_data[pair]["low"]
            idx = next(
                (i for i in range(len(ts_arr) - 1, -1, -1) if ts_arr[i] <= ts), 0
            )
            tr_vals = []
            for i in range(max(0, idx - atr_period + 1), idx + 1):
                tr = high_arr[i] - low_arr[i]
                if i > 0:
                    tr = max(tr, abs(high_arr[i] - close_arr[i-1]),
                             abs(low_arr[i] - close_arr[i-1]))
                tr_vals.append(tr)
            atr = sum(tr_vals) / len(tr_vals) if tr_vals else 0

            if atr <= 0:
                continue

            # Trailing stop check
            if pos["direction"] == "long":
                stop = pos["peak"] - stop_mult * atr
                if price < stop:
                    pnl = (price / pos["entry"] - 1) * pos["capital"]
                    rt_cost = cost.round_trip_cost(pair) * pos["capital"]
                    net = pnl - rt_cost
                    capital += net
                    trades_list.append({
                        "pnl": net,
                        "return_pct": net / pos["capital"] * 100,
                    })
                    del positions[pair]
            else:  # short
                stop = pos["peak"] + stop_mult * atr
                if price > stop:
                    pnl = (pos["entry"] / price - 1) * pos["capital"]
                    rt_cost = cost.round_trip_cost(pair) * pos["capital"]
                    net = pnl - rt_cost
                    capital += net
                    trades_list.append({
                        "pnl": net,
                        "return_pct": net / pos["capital"] * 100,
                    })
                    del positions[pair]

        # ── Record equity ──────────────────────────────────
        total_eq = capital
        for pair, pos in positions.items():
            if pos["capital"] > 0 and pos["entry"] > 0:
                close_arr = daily_data[pair]["close"]
                ts_arr = daily_data[pair]["ts"]
                price = 0.0
                for i in range(len(ts_arr) - 1, -1, -1):
                    if ts_arr[i] <= ts:
                        price = close_arr[i]; break
                if price > 0:
                    if pos["direction"] == "long":
                        total_eq += (price / pos["entry"] - 1) * pos["capital"]
                    else:
                        total_eq += (pos["entry"] / price - 1) * pos["capital"]
        equity_curve.append((ts, total_eq))

    # ── Metrics ────────────────────────────────────────────
    equity_df = pl.DataFrame(equity_curve, schema=["timestamp_ms", "equity"], orient="row")
    trades_df = pl.DataFrame(trades_list) if trades_list else None
    full_report = MetricsCalculator.from_equity_curve(equity_df, trades_df)

    # Regime decomposition
    regimes = {
        "Train 20-22": (1577836800000, 1672531200000),
        "Test 23-24": (1672531200000, 1735689600000),
        "2025 YTD": (1735689600000, 9999999999999),
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
        regime_metrics[name] = {
            "sharpe": round(MetricsCalculator.from_equity_curve(w_df).sharpe_ratio, 3),
            "return_pct": round(MetricsCalculator.from_equity_curve(w_df).total_return_pct, 1),
            "max_dd": round(MetricsCalculator.from_equity_curve(w_df).max_drawdown_pct, 1),
        }

    return equity_df, full_report, regime_metrics
