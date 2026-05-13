"""Funding Rate Carry Strategy.

Economic mechanism: when funding rate is positive, leveraged longs pay shorts.
Delta-neutral position (short perp + long spot, equal $ notional) earns
funding without directional exposure. Exit when funding compresses.

Reference: published Sharpe ~0.5-1.2 net of costs (Liu Tsyvinski Wu 2022 and
follow-ups). Decays as carry compresses with institutional participation.

Implements Strategy ABC:
  signals(df) — returns 1.0 when in position, 0.0 when flat (aligned to df's timestamps)
  run(pair)   — full delta-neutral two-leg PnL with funding accrual on the perp leg
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts
from abundance.config.settings import settings
from abundance.backtesting.costs import COST_MODEL


class FundingCarryStrategy(Strategy):
    """Delta-neutral funding rate carry on Binance perpetuals."""

    def __init__(
        self,
        entry_rate_pct: float = 0.01,    # enter when 8h funding > 0.01% (~11% annualized)
        exit_rate_pct: float = 0.005,    # exit when 8h funding < 0.005% (~5.5% annualized)
        position_size_pct: float = 0.05,
    ):
        super().__init__()
        self.entry_rate_pct = entry_rate_pct
        self.exit_rate_pct = exit_rate_pct
        self.position_size_pct = position_size_pct

    def _load_funding(self):
        plower = self._pair.lower()
        try:
            return (
                pl.scan_parquet(str(settings.raw_dir / "funding" / plower / "**" / "*.parquet"))
                .sort("timestamp_ms").collect()
            )
        except Exception:
            return None

    def signals(self, df: pl.DataFrame) -> list[float]:
        """Compute in-position state aligned to df's timestamps.

        Reads funding data internally for `self._pair`. df only needs timestamp_ms.
        Decision at end of bar i-1 uses funding rate available at that timestamp,
        so signal[i] is safe to multiply with bar-i return without lookahead.

        Fixed rate thresholds (not full-history percentile) keep this lookahead-safe.
        """
        funding = self._load_funding()
        if funding is None or len(funding) < 10:
            return [0.0] * len(df)

        ft = funding["timestamp_ms"].to_list()
        rates = funding["funding_rate_pct"].to_list()
        kt = df["timestamp_ms"].to_list()
        N = len(kt)
        if N < 2:
            return [0.0] * N

        sig = [0.0] * N
        in_pos = False
        for i in range(1, N):
            rate_prev = _nearest_le(ft, rates, kt[i - 1])
            if not in_pos and rate_prev > self.entry_rate_pct:
                in_pos = True
            elif in_pos and rate_prev < self.exit_rate_pct:
                in_pos = False
            sig[i] = 1.0 if in_pos else 0.0
        return sig

    def run(self, pair: str = "BTCUSDT") -> StrategyArtifacts:
        """Full backtest with delta-neutral two-leg PnL.

        Each entry opens BOTH legs (short perp + long spot, equal notional).
        Each exit closes both. Funding accrues on the perp leg's notional.
        Transaction costs charged per leg per side (2 entry costs + 2 exit costs
        per complete round trip).
        """
        self.set_pair(pair)
        plower = pair.lower()

        funding = self._load_funding()
        if funding is None or len(funding) < 10:
            raise RuntimeError(f"No funding data for {pair}")
        kline = (
            pl.scan_parquet(str(settings.raw_dir / "klines" / f"{plower}_1h" / "**" / "*.parquet"))
            .sort("timestamp_ms").select(["timestamp_ms", "open"]).collect()
        )

        rates = funding["funding_rate_pct"].to_list()
        ft = funding["timestamp_ms"].to_list()
        kt = kline["timestamp_ms"].to_list()
        ko = kline["open"].to_list()
        N = len(rates)

        entry_cost = COST_MODEL.entry_cost(pair, use_perp=True)
        exit_cost = COST_MODEL.exit_cost(pair, use_perp=True)

        capital = 10_000.0
        equity = [(ft[0], capital)]
        trades: list[dict] = []
        sig = [0.0] * N
        in_pos = False
        pos_cap = 0.0           # $ notional per leg
        pos_entry_price = 0.0   # entry price (shared by both legs at entry instant)
        total_fund = 0.0        # accumulated funding on perp leg
        prev_rate = rates[0]

        for i in range(1, N):
            ts = ft[i]
            rate = rates[i]
            exec_price = _nearest_le(kt, ko, ts)

            # ── Exit (close both legs) ──
            if in_pos and prev_rate < self.exit_rate_pct:
                short_perp_pnl = (pos_entry_price - exec_price) / max(pos_entry_price, 0.01) * pos_cap
                long_spot_pnl = (exec_price - pos_entry_price) / max(pos_entry_price, 0.01) * pos_cap
                # short + long ≈ 0 for price moves (delta-neutral); only carry remains
                gross = total_fund + short_perp_pnl + long_spot_pnl
                net = gross - 2.0 * exit_cost * pos_cap   # two legs to close
                capital += net
                trades.append({
                    "entry_bar": i, "pnl": net,
                    "return_pct": net / max(pos_cap, 1.0) * 100,
                    "funding_earned": total_fund,
                })
                in_pos = False

            # ── Entry (open both legs) ──
            if not in_pos and prev_rate > self.entry_rate_pct:
                pos_cap = capital * self.position_size_pct
                pos_entry_price = exec_price
                total_fund = 0.0
                in_pos = True
                capital -= 2.0 * entry_cost * pos_cap     # two legs to open

            if in_pos:
                # Funding paid by longs accrues to the short-perp leg
                total_fund += (rate / 100.0) * pos_cap
                sig[i] = 1.0

            # ── Mark-to-market equity (both legs) ──
            eq = capital
            if in_pos and pos_entry_price > 0:
                short_perp_mtm = (pos_entry_price - exec_price) / pos_entry_price * pos_cap
                long_spot_mtm = (exec_price - pos_entry_price) / pos_entry_price * pos_cap
                eq += total_fund + short_perp_mtm + long_spot_mtm
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
            "entry_rate_pct": self.entry_rate_pct,
            "exit_rate_pct": self.exit_rate_pct,
            "position_size_pct": self.position_size_pct,
        }


def _nearest_le(arr_ts, arr_val, target):
    """Binary search for the largest arr_ts <= target, return corresponding arr_val."""
    lo, hi = 0, len(arr_ts) - 1
    best = arr_val[0] if arr_val else 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr_ts[mid] <= target:
            best = arr_val[mid]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def run_strategy(pair: str = "BTCUSDT"):
    s = FundingCarryStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
