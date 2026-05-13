"""EMA 20-Day Crossover Strategy.

Signal: close[t-1] > EMA[t-2] → long at open of day t.
Faster than MA50, responds quicker to trend changes.
Implements Strategy ABC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class EMA20Strategy(Strategy):
    """20-day EMA crossover — faster trend following."""

    def __init__(self, ema_period: int = 20):
        self.ema_period = ema_period

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        N = len(close)
        ep = self.ema_period
        alpha = 2 / (ep + 1)
        ema = [close[0]] * N
        for i in range(1, N):
            ema[i] = ema[i-1] + alpha * (close[i] - ema[i-1])
        sig = [0.0] * N
        for i in range(ep + 1, N):
            sig[i] = 1.0 if close[i-1] > ema[i-2] else 0.0
        return sig

    def _get_params(self) -> dict:
        return {"ema_period": self.ema_period}

    def _detect_pair(self, df: pl.DataFrame) -> str:
        return "BTCUSDT"


def run_strategy(pair: str = "BTCUSDT", ema_period: int = 20):
    s = EMA20Strategy(ema_period=ema_period)
    art = s.run(pair)
    return art.equity_curve, art.metrics
