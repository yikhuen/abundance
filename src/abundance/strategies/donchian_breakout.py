"""Donchian Channel Breakout Strategy.

Classic Turtle Trader system: buy when yesterday's close breaks above
the 20-day high from t-2. Simple, robust, no overfitting.

Signal: close[t-1] > high_max[t-2:t-22] → long at open of day t.
No lookahead: decision uses only data available at end of day t-1.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.backtesting.costs import COST_MODEL
from abundance.config.settings import settings

N_PERIOD = 20

def run_strategy(pair: str = "BTCUSDT", n: int = N_PERIOD):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); high = df["high"].to_list()
    ts = df["timestamp_ms"].to_list(); N = len(close)

    # 20-day rolling high (computed on all data — OK because we use hh[i-2])
    hh = [max(high[max(0,i-n+1):i+1]) for i in range(N)]

    # Signal: at end of day t-1, check if close[t-1] > hh[t-2]
    sig = [0.0] * N
    for i in range(n + 1, N):
        sig[i] = 1.0 if close[i-1] > hh[i-2] else 0.0

    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,N)]
    cost_per_trade = COST_MODEL.round_trip_cost(pair, use_perp=True)
    strat_ret = [ret[i] * sig[i] for i in range(N)]
    trades = sum(1 for i in range(1,N) if sig[i] != sig[i-1])

    eq = [10000.0]
    for i in range(n + 1, N):
        txn_cost = abs(sig[i] - sig[i-1]) * cost_per_trade
        eq.append(eq[-1] * (1 + strat_ret[i] - txn_cost))

    eq_df = pl.DataFrame({"timestamp_ms": ts[n + 1:], "equity": eq[1:]})
    mc = MetricsCalculator.from_equity_curve(eq_df)
    mc.trades = trades
    return eq_df, mc
