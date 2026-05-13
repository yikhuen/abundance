"""Donchian Channel Breakout Strategy.

Signal: close[t-1] > 20-day high[t-2:t-22] → long at open of day t.
Classic Turtle Trader breakout system.
Implements Strategy ABC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class DonchianStrategy(Strategy):
    """N-day Donchian channel breakout."""

    def __init__(self, n: int = 20):
        self.n = n

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        high = df["high"].to_list()
        N = len(close)
        n = self.n
        hh = [max(high[max(0, i-n+1):i+1]) for i in range(N)]
        sig = [0.0] * N
        for i in range(n + 1, N):
            sig[i] = 1.0 if close[i-1] > hh[i-2] else 0.0
        return sig

    def _get_params(self) -> dict:
        return {"n": self.n}

    def _detect_pair(self, df: pl.DataFrame) -> str:
        return "BTCUSDT"


def run_strategy(pair: str = "BTCUSDT", n: int = 20):
    s = DonchianStrategy(n=n)
    art = s.run(pair)
    return art.equity_curve, art.metrics
