"""Ensemble Donchian Trend-Following — Zarattini, Pagani & Barbon (2025).

Reference: "Catching Crypto Trends" (SSRN: 10.2139/ssrn.5209907)
Published: Sharpe 1.58, CAGR 30%, MaxDD 19% on BTC (2015-2025).

Full implementation:
1. Ensemble of Donchian models with lookbacks [5,10,20,30]
2. Each model: entry on close > upper band, exit on trailing stop at its Donchian midpoint
3. Ensemble position = mean of all model positions (continuous 0-1)
4. Volatility targeting: position *= min(0.25 / realized_vol_20d, 2.0)
5. Rebalance threshold: 20% change before adjusting
6. All signals use data[t-1] — no lookahead
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class EnsembleDonchianStrategy(Strategy):
    """Zarattini et al. (2025) full implementation."""

    def __init__(self, lookbacks: list[int] | None = None,
                 vol_target: float = 0.25, vol_lookback: int = 20,
                 vol_cap: float = 2.0, rebalance_threshold: float = 0.20):
        self.lookbacks = lookbacks or [5, 10, 20, 30]
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.vol_cap = vol_cap
        self.rebalance_threshold = rebalance_threshold

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        high = df["high"].to_list()
        low = df["low"].to_list()
        N = len(close)

        max_lb = max(self.lookbacks)

        # --- Per-model signals with individual trailing stops ---
        model_positions = {}
        for n in self.lookbacks:
            hh = [max(high[max(0, i-n+1):i+1]) for i in range(N)]
            ll = [min(low[max(0, i-n+1):i+1]) for i in range(N)]
            pos = [0.0] * N
            in_pos = False
            stop = 0.0
            for i in range(n + 1, N):
                if not in_pos and close[i-1] > hh[i-2]:
                    pos[i] = 1.0
                    in_pos = True
                    stop = (hh[i-2] + ll[i-2]) / 2 if i >= 2 else 0
                elif in_pos:
                    new_stop = (hh[i-2] + ll[i-2]) / 2 if i >= 2 else 0
                    if new_stop > stop:
                        stop = new_stop
                    if close[i-1] < stop:
                        pos[i] = 0.0
                        in_pos = False
                    else:
                        pos[i] = 1.0
            model_positions[n] = pos

        # --- Ensemble: mean of all model positions ---
        ensemble = [0.0] * N
        for i in range(max_lb + 2, N):
            total = sum(model_positions[n][i] for n in self.lookbacks)
            ensemble[i] = total / len(self.lookbacks)

        # --- Volatility targeting ---
        ret = [0.0] + [
            (close[i]/close[i-1]-1) for i in range(1, N)
        ]
        vol_weight = [1.0] * N
        vl = self.vol_lookback
        for i in range(vl + 1, N):
            window = [ret[j] for j in range(max(1, i-vl), i)]
            if len(window) > 5:
                daily_std = (sum((r - sum(window)/len(window))**2
                                 for r in window) / len(window))**0.5
                ann_vol = daily_std * (365**0.5)
                if ann_vol > 0:
                    vol_weight[i] = min(self.vol_target / ann_vol, self.vol_cap)

        # --- Combine ensemble + vol targeting ---
        raw_position = [0.0] * N
        for i in range(max(vl, max_lb) + 2, N):
            raw_position[i] = ensemble[i] * vol_weight[i]

        # --- Rebalance threshold ---
        sig = [0.0] * N
        last_position = 0.0
        for i in range(max(vl, max_lb) + 2, N):
            if abs(raw_position[i] - last_position) > self.rebalance_threshold:
                sig[i] = raw_position[i]
                last_position = raw_position[i]
            else:
                sig[i] = last_position

        return sig

    def _get_params(self) -> dict:
        return {
            "lookbacks": self.lookbacks,
            "vol_target": self.vol_target,
            "vol_lookback": self.vol_lookback,
            "vol_cap": self.vol_cap,
            "rebalance_threshold": self.rebalance_threshold,
        }

    def apply_costs(self, signals, df, use_perp=True):
        """Override: continuous positions need proportional cost model."""
        close = df["close"].to_list()
        N = len(close)
        ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1, N)]
        pair = self._detect_pair(df)
        entry_cost = __import__('abundance.backtesting.costs', fromlist=['COST_MODEL']).COST_MODEL.entry_cost(pair, use_perp)
        exit_cost = __import__('abundance.backtesting.costs', fromlist=['COST_MODEL']).COST_MODEL.exit_cost(pair, use_perp)

        net_ret = [0.0] * N
        for i in range(1, N):
            gross = ret[i] * signals[i]
            delta = signals[i] - signals[i-1]
            txn = 0.0
            if delta > 0:
                txn = delta * entry_cost
            elif delta < 0:
                txn = abs(delta) * exit_cost
            net_ret[i] = gross - txn
        return net_ret


def run_strategy(pair="BTCUSDT"):
    s = EnsembleDonchianStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
