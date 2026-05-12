"""Dual Thrust Breakout Strategy.

Classic range-breakout system: HH-LC defines range, bands = open ± K*range.
Buy when close > upper band, sell when close < lower band.
Different signal logic from MA/ADX — captures volatility breakouts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.config.settings import settings

def run_strategy(pair: str = "BTCUSDT", n: int = 20, k1: float = 0.5, k2: float = 0.5):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); high = df["high"].to_list()
    low = df["low"].to_list(); opn = df["open"].to_list()
    ts = df["timestamp_ms"].to_list(); N = len(close)

    hh = [max(high[max(0,i-n+1):i+1]) for i in range(N)]
    lc = [min(close[max(0,i-n+1):i+1]) for i in range(N)]
    rng = [hh[i] - lc[i] for i in range(N)]

    sig = [0.0] * N
    for i in range(n, N):
        buy_line = opn[i] + k2 * rng[i-1]
        sell_line = opn[i] - k1 * rng[i-1]
        if close[i] > buy_line:
            sig[i] = 1.0
        elif close[i] < sell_line:
            sig[i] = -1.0
        # else stay flat

    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,N)]
    strat_ret = [ret[i]*sig[i] for i in range(N)]
    eq = [10000.0]
    for i in range(n,N):
        cost = abs(sig[i] - sig[i-1]) * 0.001  # 10bp txn cost
        eq.append(eq[-1] * (1 + strat_ret[i] - cost))

    eq_df = pl.DataFrame({"timestamp_ms": ts[n:], "equity": eq[1:]})
    return eq_df, MetricsCalculator.from_equity_curve(eq_df)
