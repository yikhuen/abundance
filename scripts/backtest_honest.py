"""Honest backtest: all 4 strategies with lookahead-fixed signals, consistent costs, B&H baseline."""
import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import polars as pl; from datetime import datetime
from abundance.backtesting.metrics import MetricsCalculator
from abundance.backtesting.costs import COST_MODEL
from abundance.config.settings import settings
from abundance.strategies.momentum.grayscale_ma50 import run_strategy as ma50
from abundance.strategies.ema_crossover import run_strategy as ema20
from abundance.strategies.donchian_breakout import run_strategy as donchian
from abundance.strategies.composite.adx_blend import run_strategy as adx_blend

def bh_metrics(pair: str):
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{pair.lower()}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close=df["close"].to_list(); ts=df["timestamp_ms"].to_list(); N=len(close)
    ret=[0.0]+[close[i]/close[i-1]-1 for i in range(1,N)]
    eq=[10000.0]
    for i in range(1,N): eq.append(eq[-1]*(1+ret[i]))
    eq_df=pl.DataFrame({"timestamp_ms":ts,"equity":eq})
    return MetricsCalculator.from_equity_curve(eq_df), (eq[-1]/eq[0]-1)*100

for pair, label in [("BTCUSDT","BTC"),("ETHUSDT","ETH")]:
    print(f"\n{'='*60}\n{label}USDT")
    bh_mc,bh_ret=bh_metrics(pair)
    print(f"  Buy & Hold: Return={bh_ret:,.1f}% Sharpe={bh_mc.sharpe_ratio:.2f} DD={bh_mc.max_drawdown_pct:.1f}%")

    strategies = [
        ("MA50", ma50),
        ("Donchian20", donchian),
        ("EMA20", ema20),
        ("ADX-blend", adx_blend),
    ]
    for name, fn in strategies:
        try:
            eq_df, mc = fn(pair)
            ret_pct = (eq_df["equity"][-1]/eq_df["equity"][0]-1)*100
            trades = getattr(mc, 'trades', 0)
            print(f"  {name:15s}: Return={ret_pct:>10,.1f}% Sharpe={mc.sharpe_ratio:5.2f} "
                  f"DD={mc.max_drawdown_pct:5.1f}% Trades={trades:4d} "
                  f"α={(mc.sharpe_ratio-bh_mc.sharpe_ratio):+.2f}")
        except Exception as e:
            print(f"  {name:15s}: ERROR — {e}")
