"""Funding Rate Carry Strategy.

Economic mechanism: When funding rate is positive, leveraged longs pay shorts.
Enter delta-neutral position (short perp + long spot) to capture the carry.
Exit when funding normalizes.

Reference: Published Sharpe 0.8-1.5 (decays as carry compresses).
Implements Strategy ABC with custom multi-source data loading.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts
from abundance.config.settings import settings
from abundance.backtesting.costs import COST_MODEL


class FundingCarryStrategy(Strategy):
    """Delta-neutral funding rate carry."""

    def __init__(self, entry_percentile: float = 0.75, exit_percentile: float = 0.25,
                 position_size_pct: float = 0.05):
        self.entry_percentile = entry_percentile
        self.exit_percentile = exit_percentile
        self.position_size_pct = position_size_pct

    def signals(self, df: pl.DataFrame) -> list[float]:
        return []

    def run(self, pair: str = "BTCUSDT") -> StrategyArtifacts:
        self.set_pair(pair)
        plower = pair.lower()

        funding = (
            pl.scan_parquet(str(settings.raw_dir / "funding" / plower / "**" / "*.parquet"))
            .sort("timestamp_ms").collect()
        )
        kline = (
            pl.scan_parquet(str(settings.raw_dir / "klines" / f"{plower}_1h" / "**" / "*.parquet"))
            .sort("timestamp_ms").select(["timestamp_ms", "open"]).collect()
        )

        rates = funding["funding_rate_pct"].to_list()
        ft = funding["timestamp_ms"].to_list()
        kt = kline["timestamp_ms"].to_list()
        ko = kline["open"].to_list()
        N = len(rates)

        entry_th = rates_quantile(rates, self.entry_percentile) if len(rates) > 10 else 0.01
        exit_th = rates_quantile(rates, self.exit_percentile) if len(rates) > 10 else 0.005

        cost_per_trade = COST_MODEL.round_trip_cost(pair, use_perp=True)
        entry_cost = COST_MODEL.entry_cost(pair, use_perp=True)
        exit_cost = COST_MODEL.exit_cost(pair, use_perp=True)

        capital = 10000.0
        equity = [(ft[0], capital)]
        trades = []
        sig = [0.0] * N
        in_pos = False
        pos_cap = 0.0; pos_entry = 0.0; total_fund = 0.0
        prev_rate = rates[0]

        for i in range(1, N):
            ts = ft[i]; rate = rates[i]
            exec_price = _nearest_le(kt, ko, ts)

            # Exit
            if in_pos and prev_rate < exit_th:
                spot_pnl = (pos_entry - exec_price) / max(pos_entry, 0.01) * pos_cap
                gross = total_fund - spot_pnl
                net = gross - exit_cost * pos_cap
                capital += net
                trades.append({"entry_bar": i, "pnl": net, "return_pct": net / max(pos_cap, 1) * 100})
                in_pos = False
                sig[i] = 0.0

            # Entry
            if not in_pos and prev_rate > entry_th:
                pos_cap = capital * self.position_size_pct
                pos_entry = exec_price
                total_fund = 0.0
                in_pos = True
                sig[i] = 1.0
                capital -= entry_cost * pos_cap

            if in_pos:
                total_fund += (rate / 100) * pos_cap
                sig[i] = 1.0

            eq = capital
            if in_pos:
                eq = capital + total_fund - (pos_entry - exec_price) / max(pos_entry, 0.01) * pos_cap
            equity.append((ts, max(eq, 0.01)))

            prev_rate = rate

        eq_df = pl.DataFrame(equity, schema=["timestamp_ms", "equity"], orient="row")
        from abundance.backtesting.metrics import MetricsCalculator
        mc = MetricsCalculator.from_equity_curve(eq_df)
        mc.trades = len(trades)

        return StrategyArtifacts(
            signals=sig, equity_curve=eq_df, metrics=mc, trades=trades,
            params=self._get_params(), pair=pair,
        )

    def _get_params(self) -> dict:
        return {
            "entry_percentile": self.entry_percentile,
            "exit_percentile": self.exit_percentile,
            "position_size_pct": self.position_size_pct,
        }


def _nearest_le(arr_ts, arr_val, target):
    lo, hi = 0, len(arr_ts) - 1
    best = arr_val[0] if arr_val else 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr_ts[mid] <= target:
            best = arr_val[mid]; lo = mid + 1
        else:
            hi = mid - 1
    return best


def rates_quantile(values, q):
    s = sorted(values)
    idx = int(len(s) * q)
    return s[min(idx, len(s) - 1)]


def run_strategy(pair="BTCUSDT"):
    s = FundingCarryStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
