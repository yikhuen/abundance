"""Grayscale Bitcoin Momentum Strategy (50-day MA crossover).

Based on: Grayscale Research "The Trend Is Your Friend" (2025)
Signal: close[t-1] > MA[t-2] → long at open of day t.
No lookahead: decision made on yesterday's data.
Implements Strategy ABC with separable signals(), apply_costs(), etc.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class MA50Strategy(Strategy):
    """50-day moving average crossover — trend following."""

    def __init__(self, ma_period: int = 50):
        self.ma_period = ma_period

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        N = len(close)
        mp = self.ma_period
        ma = [sum(close[max(0, i-mp+1):i+1]) / min(i+1, mp) for i in range(N)]
        sig = [0.0] * N
        for i in range(mp + 1, N):
            sig[i] = 1.0 if close[i-1] > ma[i-2] else 0.0
        return sig

    def _get_params(self) -> dict:
        return {"ma_period": self.ma_period}

# Backward-compatible runner
def run_strategy(pair: str = "BTCUSDT", ma_period: int = 50):
    s = MA50Strategy(ma_period=ma_period)
    art = s.run(pair)
    return art.equity_curve, art.metrics
