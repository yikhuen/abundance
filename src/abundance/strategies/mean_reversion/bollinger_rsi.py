"""Bollinger Band Mean Reversion with RSI confirmation.

Entry: price touches lower BB(20,2) AND RSI(14) < 30 → long
Exit: price reverts to middle band (SMA 20) OR RSI > 50 → close

Reported CAGR 49.7%, lower MaxDD than B&H, risk-adjusted return 144.7%.
Based on quantifiedstrategies.com backtest + Gate Research (2026).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.backtesting.costs import COST_MODEL
from abundance.config.settings import settings

def run_strategy(pair: str = "BTCUSDT", bb_period: int = 20, bb_std: float = 2.0, rsi_period: int = 14):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); high = df["high"].to_list(); low = df["low"].to_list()
    ts = df["timestamp_ms"].to_list(); n = len(close)

    # Bollinger Bands
    sma = [sum(close[max(0,i-bb_period+1):i+1])/min(i+1,bb_period) for i in range(n)]
    std = [(sum((close[j]-sma[i])**2 for j in range(max(0,i-bb_period+1),i+1))/min(i+1,bb_period))**0.5 for i in range(n)]
    upper = [sma[i]+bb_std*std[i] for i in range(n)]
    lower = [sma[i]-bb_std*std[i] for i in range(n)]

    # RSI
    delta = [close[i]-close[i-1] if i>0 else 0 for i in range(n)]
    gain = [max(d,0) for d in delta]; loss = [max(-d,0) for d in delta]
    avg_gain = [sum(gain[max(0,i-rsi_period+1):i+1])/min(i+1,rsi_period) for i in range(n)]
    avg_loss = [sum(loss[max(0,i-rsi_period+1):i+1])/min(i+1,rsi_period) for i in range(n)]
    rs = [avg_gain[i]/max(avg_loss[i],0.0001) for i in range(n)]
    rsi = [100-100/(1+rs[i]) for i in range(n)]

    # Signals (shifted — no lookahead)
    warmup = max(bb_period, rsi_period)
    sig = [0.0]*n
    in_pos = False; entry_price = 0.0
    cost = COST_MODEL

    for i in range(warmup,n):
        prev_lower = lower[i-1]; prev_sma = sma[i-1]; prev_rsi = rsi[i-1]
        if not in_pos and close[i] <= prev_lower and prev_rsi < 30:
            sig[i] = 1.0; in_pos = True; entry_price = close[i]
        elif in_pos and (close[i] >= prev_sma or prev_rsi > 50):
            sig[i] = 0.0; in_pos = False
        elif in_pos:
            sig[i] = 1.0

    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,n)]
    strat_ret = [ret[i]*sig[i] for i in range(n)]
    eq = [10000.0]
    for i in range(warmup,n): eq.append(eq[-1]*(1+strat_ret[i]))

    eq_df = pl.DataFrame({"timestamp_ms":ts[warmup:],"equity":pl.Series(eq[1:])})
    return eq_df, MetricsCalculator.from_equity_curve(eq_df)
