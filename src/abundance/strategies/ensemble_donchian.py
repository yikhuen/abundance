"""Ensemble Donchian Trend-Following Strategy.

Reference: Zarattini, Pagani & Barbon (2025)
  "Catching Crypto Trends: A Tactical Approach for Bitcoin and Altcoins"
  SSRN: 10.2139/ssrn.5209907

Published results (BTC, 2015-2025):
  Sharpe: 1.58 | CAGR: 30% | Max DD: 19% | Alpha: 14% vs BTC

Strategy:
  - Ensemble of Donchian Channel models with N lookback periods [5..360]
  - Entry: fraction of models with close > upper band > 0.5
  - Exit: trailing stop at Donchian channel midpoint
  - Volatility-based position sizing (target 25% ann vol)

All signals use data[t-1] — no lookahead.
Implements Strategy ABC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class EnsembleDonchianStrategy(Strategy):
    """Zarattini et al. (2025) Ensemble Donchian trend-following."""

    def __init__(self, lookbacks: list[int] | None = None, stop_lookback: int = 20):
        self.stop_lookback = stop_lookback
        self.lookbacks = lookbacks or [5, 10, 20, 30, 60, 90, 150, 250, 360]

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        high = df["high"].to_list()
        low = df["low"].to_list()
        N = len(close)

        # Compute signals for each lookback
        model_signals = {}
        for n in self.lookbacks:
            hh = [max(high[max(0, i-n+1):i+1]) for i in range(N)]
            ll = [min(low[max(0, i-n+1):i+1]) for i in range(N)]
            sig = [0.0] * N
            for i in range(n + 1, N):
                sig[i] = 1.0 if close[i-1] > hh[i-2] else 0.0
            model_signals[n] = sig

        # Ensemble: fraction of models that are long at each bar
        ensemble = [0.0] * N
        for i in range(max(self.lookbacks) + 1, N):
            long_count = sum(model_signals[n][i] for n in self.lookbacks)
            ensemble[i] = long_count / len(self.lookbacks)

        # Entry threshold: >50% of models signal long
        # Exit: trailing stop at Donchian midpoint
        sig = [0.0] * N
        in_pos = False
        stop_price = 0.0

        for i in range(max(self.lookbacks) + 1, N):
            if not in_pos and ensemble[i] > 0.5:
                sig[i] = 1.0
                in_pos = True
                # Initial stop: midpoint of longest-lookback Donchian channel
                ns = self.stop_lookback
                hh_s = max(high[max(0, i-ns+1):i])
                ll_s = min(low[max(0, i-ns+1):i])
                stop_price = (hh_s + ll_s) / 2
            elif in_pos:
                # Update trailing stop (never moves down)
                ns = self.stop_lookback
                hh_s = max(high[max(0, i-ns+1):i])
                ll_s = min(low[max(0, i-ns+1):i])
                new_stop = (hh_s + ll_s) / 2
                if new_stop > stop_price:
                    stop_price = new_stop

                # Exit: yesterday's close below stop
                if close[i-1] < stop_price:
                    sig[i] = 0.0
                    in_pos = False
                else:
                    sig[i] = 1.0
        return sig

    def _get_params(self) -> dict:
        return {"lookbacks": self.lookbacks, "stop_lookback": self.stop_lookback}


def run_strategy(pair="BTCUSDT"):
    s = EnsembleDonchianStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
