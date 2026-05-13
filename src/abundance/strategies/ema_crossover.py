"""EMA 20-Day Crossover Strategy.

Simple trend-following: long when yesterday's close > EMA at t-2.
Faster than MA50 (responds quicker), captures shorter trends.

Signal: close[t-1] > EMA[t-2] → long at open of day t.
No lookahead: decision made on yesterday's data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.backtesting.costs import COST_MODEL
from abundance.config.settings import settings

EMA_PERIOD = 20

def run_strategy(pair: str = "BTCUSDT", ema_period: int = EMA_PERIOD):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); ts = df["timestamp_ms"].to_list(); N = len(close)

    # EMA (computed on all data — OK because we use ema[i-2] for signal)
    alpha = 2 / (ema_period + 1)
    ema = [close[0]] * N
    for i in range(1, N):
        ema[i] = ema[i-1] + alpha * (close[i] - ema[i-1])

    # Signal: decide at end of day t-1, earn return of day t
    sig = [0.0] * N
    for i in range(ema_period + 1, N):
        sig[i] = 1.0 if close[i-1] > ema[i-2] else 0.0

    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,N)]
    cost_per_trade = COST_MODEL.round_trip_cost(pair, use_perp=True)
    strat_ret = [ret[i] * sig[i] for i in range(N)]
    trades = sum(1 for i in range(1,N) if sig[i] != sig[i-1])

    eq = [10000.0]
    for i in range(ema_period + 1, N):
        txn_cost = abs(sig[i] - sig[i-1]) * cost_per_trade
        eq.append(eq[-1] * (1 + strat_ret[i] - txn_cost))

    eq_df = pl.DataFrame({"timestamp_ms": ts[ema_period + 1:], "equity": eq[1:]})
    mc = MetricsCalculator.from_equity_curve(eq_df)
    mc.trades = trades
    return eq_df, mc
