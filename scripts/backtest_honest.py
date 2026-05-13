"""Honest backtest v2: Strategy ABC, fixed costs (one-way fees), B&H baseline."""
import sys; from pathlib import Path; sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import polars as pl
from abundance.config.settings import settings
from abundance.backtesting.metrics import MetricsCalculator
from abundance.strategies.momentum.grayscale_ma50 import MA50Strategy
from abundance.strategies.ema_crossover import EMA20Strategy
from abundance.strategies.donchian_breakout import DonchianStrategy
from abundance.strategies.composite.adx_blend import ADXBlendStrategy

def bh_metrics(pair: str):
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{pair.lower()}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close=df["close"].to_list(); ts=df["timestamp_ms"].to_list(); N=len(close)
    ret=[0.0]+[close[i]/close[i-1]-1 for i in range(1,N)]
    eq=[10000.0]
    for i in range(1,N): eq.append(eq[-1]*(1+ret[i]))
    eq_df=pl.DataFrame({"timestamp_ms":ts,"equity":eq})
    return MetricsCalculator.from_equity_curve(eq_df), (eq[-1]/eq[0]-1)*100

for pair, label in [("BTCUSDT","BTC"),("ETHUSDT","ETH")]:
    print(f"\n{'='*60}\n{label}USDT  (costs: one-way entry + exit = round-trip)")
    bh_mc,bh_ret=bh_metrics(pair)
    print(f"  B&H:      Return={bh_ret:>10,.1f}% Sharpe={bh_mc.sharpe_ratio:5.2f} DD={bh_mc.max_drawdown_pct:5.1f}%")

    for name, StratClass, kwargs in [
        ("MA50",        MA50Strategy,       {}),
        ("Donchian20",  DonchianStrategy,   {"n": 20}),
        ("EMA20",       EMA20Strategy,      {}),
        ("ADX-blend",   ADXBlendStrategy,   {}),
    ]:
        s = StratClass(**kwargs)
        art = s.run(pair)
        ret_pct = (art.equity_curve["equity"][-1]/art.equity_curve["equity"][0]-1)*100
        trades = art.metrics.trades
        alpha = art.metrics.sharpe_ratio - bh_mc.sharpe_ratio
        print(f"  {name:15s}: Return={ret_pct:>10,.1f}% Sharpe={art.metrics.sharpe_ratio:5.2f} "
              f"DD={art.metrics.max_drawdown_pct:5.1f}% Trades={trades:4d} α={alpha:+.2f}")
