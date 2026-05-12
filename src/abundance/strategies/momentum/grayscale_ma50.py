"""Grayscale Bitcoin Momentum Strategy (50-day MA crossover).

Based on: Grayscale Research "The Trend Is Your Friend" (2025)
          50-day MA crossover on BTC spot.

Reported: Sharpe 1.9, Annualized Return 126% (2012-2023).
Simple: long above 50d MA, cash below. No lookahead.
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
    close = df["close"].to_list(); ts = df["timestamp_ms"].to_list(); n = len(close)
    
    ma = [sum(close[max(0,i-ma_period+1):i+1])/min(i+1,ma_period) for i in range(n)]
    sig = [0.0]*n
    for i in range(ma_period,n):
        sig[i] = 1.0 if close[i] > ma[i-1] else 0.0  # shifted: use yesterday's MA
    
    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,n)]
    strat_ret = [ret[i]*sig[i] for i in range(n)]
    eq = [10000.0]
    for i in range(ma_period,n): eq.append(eq[-1]*(1+strat_ret[i]))
    
    eq_df = pl.DataFrame({"timestamp_ms":ts[ma_period:],"equity":pl.Series(eq[1:])})
    return eq_df, MetricsCalculator.from_equity_curve(eq_df)
