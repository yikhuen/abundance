"""Bollinger Band Mean Reversion Strategy.

Economic mechanism: Prices tend to revert from extremes. When price touches
lower Bollinger band (2σ below moving average), oversold condition → buy.
Exit when price returns to middle band (MA). Different signal source from
trend-following — captures mean-reversion alpha.

Signal: close[t-1] < MA[t-2] - 2*σ[t-2] → long at open of day t.
Exit:  close[t-1] > MA[t-2] → flat.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class BollingerReversionStrategy(Strategy):
    """Bollinger Band mean reversion — buy at lower band, sell at MA."""

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self.period = period
        self.num_std = num_std

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        N = len(close)
        p = self.period
        k = self.num_std

        ma = [sum(close[max(0,i-p+1):i+1])/min(i+1,p) for i in range(N)]
        # Rolling std
        std = [0.0] * N
        for i in range(p, N):
            window = close[i-p+1:i+1]
            m = ma[i]
            std[i] = (sum((x-m)**2 for x in window)/p)**0.5

        lower_band = [ma[i] - k*std[i] for i in range(N)]

        sig = [0.0] * N
        in_pos = False
        for i in range(p + 1, N):
            # Entry: yesterday's close below lower band → oversold bounce
            if not in_pos and close[i-1] < lower_band[i-2]:
                sig[i] = 1.0
                in_pos = True
            # Exit: yesterday's close back above MA
            elif in_pos and close[i-1] > ma[i-2]:
                sig[i] = 0.0
                in_pos = False
            elif in_pos:
                sig[i] = 1.0  # stay in position
        return sig

    def _get_params(self) -> dict:
        return {"period": self.period, "num_std": self.num_std}

    def _detect_pair(self, df: pl.DataFrame) -> str:
        return "BTCUSDT"


def run_strategy(pair="BTCUSDT"):
    s = BollingerReversionStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
