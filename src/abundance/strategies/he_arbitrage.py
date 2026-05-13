"""He, Manela, Ross & von Wachter (2022) — No-Arbitrage Perp Strategy.

Paper: "Fundamentals of Perpetual Futures" (arXiv:2212.06888)
Economic mechanism: When actual perp price deviates from theoretical
no-arbitrage price, enter delta-neutral position to capture convergence.

F_theoretical = S × (1 + rT) / (1 + fT)
Strategy: delta-neutral (short perp + long spot when F > theoretical)
Implements Strategy ABC with multi-source data (spot, perp, funding).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts
from abundance.config.settings import settings
from abundance.backtesting.costs import COST_MODEL


class HeArbitrageStrategy(Strategy):
    """He et al. no-arbitrage perpetual convergence."""

    def __init__(self, entry_threshold_pct: float = 0.05, exit_threshold_pct: float = 0.01,
                 position_size_pct: float = 0.10, risk_free: float = 0.04):
        self.entry_threshold_pct = entry_threshold_pct
        self.exit_threshold_pct = exit_threshold_pct
        self.position_size_pct = position_size_pct
        self.risk_free = risk_free

    def signals(self, df: pl.DataFrame) -> list[float]:
        return []

    def run(self, pair: str = "BTCUSDT") -> StrategyArtifacts:
        self.set_pair(pair)
        plower = pair.lower()

        spot = (pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1h"/"**"/"*.parquet"))
                .sort("timestamp_ms").select(["timestamp_ms","close"]).collect())
        perp = (pl.scan_parquet(str(settings.raw_dir/"perp_klines"/f"{plower}_1h"/"**"/"*.parquet"))
                .sort("timestamp_ms").select(["timestamp_ms","close"]).collect())
        funding = (pl.scan_parquet(str(settings.raw_dir/"funding"/plower/"**"/"*.parquet"))
                   .sort("timestamp_ms").collect())

        sts = spot["timestamp_ms"].to_list(); sc = spot["close"].to_list()
        pts = perp["timestamp_ms"].to_list(); pc = perp["close"].to_list()
        fr = funding["funding_rate_pct"].to_list(); fts = funding["timestamp_ms"].to_list()

        entry_cost = COST_MODEL.entry_cost(pair, use_perp=True)
        exit_cost = COST_MODEL.exit_cost(pair, use_perp=True)

        def spot_at(t): return _near(sts, sc, t)
        def perp_at(t): return _near(pts, pc, t)

        capital = 10000.0; equity = [(fts[0], capital)]; trades = []; sig = [0.0]*len(fts)
        in_pos = False; pos_cap = 0.0; pos_entry_p = 0.0; pos_entry_s = 0.0; pos_type = ""

        for i in range(1, len(fts)):
            ts = fts[i]; rate_prev = fr[i-1]
            sp = spot_at(ts); pp = perp_at(ts)
            if sp <= 0 or pp <= 0: continue

            T = 8.0/(365.25*24); f = rate_prev/100; r = self.risk_free
            theory = sp * (1+r*T)/(1+f*T) if 1+f*T > 0 else sp
            dev = (pp-theory)/theory*100

            # Exit
            if in_pos and abs(dev) < self.exit_threshold_pct:
                if pos_type == "short_perp_long_spot":
                    perp_pnl = (pos_entry_p-pp)/pos_entry_p*pos_cap
                    spot_pnl = (sp/pos_entry_s-1)*pos_cap
                else:
                    perp_pnl = (pp/pos_entry_p-1)*pos_cap
                    spot_pnl = (1-sp/pos_entry_s)*pos_cap
                gross = perp_pnl+spot_pnl; net = gross - exit_cost*pos_cap
                capital += net
                trades.append({"entry_bar":i,"pnl":net})
                in_pos = False; sig[i] = 0.0

            # Entry
            if not in_pos and abs(dev) > self.entry_threshold_pct:
                pos_cap = capital*self.position_size_pct
                pos_entry_p = pp; pos_entry_s = sp
                pos_type = "short_perp_long_spot" if dev > 0 else "long_perp_short_spot"
                in_pos = True; sig[i] = 1.0
                capital -= entry_cost*pos_cap

            if in_pos: sig[i] = 1.0

            eq = capital
            if in_pos and pos_entry_p > 0:
                delta = (pp/pos_entry_p-1)*pos_cap
                if pos_type == "short_perp_long_spot": delta = -delta
                eq += delta
            equity.append((ts, max(eq, 0.01)))

        eq_df = pl.DataFrame(equity, schema=["timestamp_ms","equity"], orient="row")
        from abundance.backtesting.metrics import MetricsCalculator
        mc = MetricsCalculator.from_equity_curve(eq_df); mc.trades = len(trades)
        return StrategyArtifacts(signals=sig, equity_curve=eq_df, metrics=mc, trades=trades,
                                 params=self._get_params(), pair=pair)

    def _get_params(self) -> dict:
        return {"entry_threshold_pct": self.entry_threshold_pct,
                "exit_threshold_pct": self.exit_threshold_pct,
                "position_size_pct": self.position_size_pct, "risk_free": self.risk_free}


def _near(arr_ts, arr_val, target):
    lo, hi = 0, len(arr_ts)-1; best = 0.0
    while lo <= hi:
        mid = (lo+hi)//2
        if arr_ts[mid] <= target: best = arr_val[mid]; lo = mid+1
        else: hi = mid-1
    return best


def run_strategy(pair="BTCUSDT"):
    s = HeArbitrageStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
