"""Grayscale Bitcoin Momentum Strategy (50-day MA crossover).

Based on: Grayscale Research "The Trend Is Your Friend" (2025)
          50-day MA crossover on BTC spot.

Signal: close[t-1] > MA[t-2] → go long at open of day t.
No lookahead: decision made on yesterday's data, earns today's return.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.backtesting.costs import COST_MODEL
from abundance.config.settings import settings

def run_strategy(pair: str = "BTCUSDT", ma_period: int = 50):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); ts = df["timestamp_ms"].to_list(); N = len(close)

    # MA computed on all data (including today — OK because we use ma[i-2] for signal)
    ma = [sum(close[max(0,i-ma_period+1):i+1])/min(i+1,ma_period) for i in range(N)]

    # Signal: decide at end of day t-1 whether to be long for day t
    sig = [0.0] * N
    for i in range(ma_period + 1, N):
        sig[i] = 1.0 if close[i-1] > ma[i-2] else 0.0

    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,N)]
    cost_per_trade = COST_MODEL.round_trip_cost(pair, use_perp=True)
    strat_ret = [ret[i] * sig[i] for i in range(N)]
    trades = sum(1 for i in range(1,N) if sig[i] != sig[i-1])

    eq = [10000.0]
    for i in range(ma_period + 1, N):
        txn_cost = abs(sig[i] - sig[i-1]) * cost_per_trade
        eq.append(eq[-1] * (1 + strat_ret[i] - txn_cost))

    eq_df = pl.DataFrame({"timestamp_ms": ts[ma_period + 1:], "equity": eq[1:]})
    mc = MetricsCalculator.from_equity_curve(eq_df)
    mc.trades = trades
    return eq_df, mc
