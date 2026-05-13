"""He, Manela, Ross & von Wachter (2022) — No-Arbitrage Perp Strategy.

Paper: "Fundamentals of Perpetual Futures" (arXiv:2212.06888)

Economic mechanism: When actual perp price deviates from theoretical no-arbitrage
price, enter delta-neutral position to capture the convergence.

  F_theoretical = S × (1 + r_annual × T) / (1 + f_annual × T)

where:
  S = spot price
  r_annual = annualized risk-free rate (e.g., 0.04)
  f_annual = annualized funding rate (per-period rate × 3 fundings/day × 365 days)
  T = time to next funding payment, in years (8h ≈ 9.13e-4 years)

Both r and f are now annualized so r*T and f*T are dimensionally consistent.
The previous version multiplied per-period funding by years, collapsing the
funding term to ~1e-7 of its intended magnitude.

Strategy:
  short perp + long spot when F > theoretical (perp expensive)
  long perp + short spot when F < theoretical (perp cheap)
  exit when |dev| < exit_threshold_pct
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts
from abundance.config.settings import settings
from abundance.backtesting.costs import COST_MODEL


# ── Constants (Binance perp convention: 8h funding) ──
PERIOD_HOURS = 8.0
PERIODS_PER_YEAR = (365.25 * 24) / PERIOD_HOURS         # ≈ 1095.75
TIME_TO_NEXT_FUNDING_YEARS = PERIOD_HOURS / (365.25 * 24)


class HeArbitrageStrategy(Strategy):
    """He et al. no-arbitrage perpetual convergence."""

    def __init__(
        self,
        entry_threshold_pct: float = 0.05,
        exit_threshold_pct: float = 0.01,
        position_size_pct: float = 0.10,
        risk_free: float = 0.04,
    ):
        super().__init__()
        self.entry_threshold_pct = entry_threshold_pct
        self.exit_threshold_pct = exit_threshold_pct
        self.position_size_pct = position_size_pct
        self.risk_free = risk_free

    def _theoretical_price(self, spot: float, rate_per_period: float) -> float:
        """No-arbitrage perp price.

        Args:
            spot: current spot price.
            rate_per_period: funding rate as decimal per 8h period (e.g., 0.0001 = 1bp/8h).
        """
        f_annual = rate_per_period * PERIODS_PER_YEAR
        T = TIME_TO_NEXT_FUNDING_YEARS
        denom = 1.0 + f_annual * T
        if denom <= 0:
            return spot
        return spot * (1.0 + self.risk_free * T) / denom

    def _load_supplementary(self, pair: str):
        """Load perp klines and funding for `pair`. Returns (perp_df, funding_df) or (None, None).

        Filters null timestamps so downstream binary search can safely compare.
        """
        plower = pair.lower()
        try:
            perp = (
                pl.scan_parquet(str(settings.raw_dir / "perp_klines" / f"{plower}_1h" / "**" / "*.parquet"))
                .sort("timestamp_ms").select(["timestamp_ms", "close"]).collect()
            )
            perp = perp.filter(
                pl.col("timestamp_ms").is_not_null() & pl.col("close").is_not_null()
            )
        except Exception:
            return None, None
        try:
            funding = (
                pl.scan_parquet(str(settings.raw_dir / "funding" / plower / "**" / "*.parquet"))
                .sort("timestamp_ms").collect()
            )
            funding = funding.filter(
                pl.col("timestamp_ms").is_not_null() & pl.col("funding_rate_pct").is_not_null()
            )
        except Exception:
            return None, None
        return perp, funding

    def signals(self, df: pl.DataFrame) -> list[float]:
        """Compute in-position state aligned to df's timestamps.

        df: spot price DataFrame with timestamp_ms and close columns.
        Loads perp + funding internally for `self._pair`.

        Returns all-zeros if supplementary data is missing — the
        check_strategy_uses_abc adversarial check will then flag this strategy
        as un-auditable for the pair, which is the correct outcome.
        """
        if "close" not in df.columns:
            return [0.0] * len(df)

        perp, funding = self._load_supplementary(self._pair)
        if perp is None or funding is None or len(perp) < 10 or len(funding) < 10:
            return [0.0] * len(df)

        pts = perp["timestamp_ms"].to_list()
        pc = perp["close"].to_list()
        fts = funding["timestamp_ms"].to_list()
        fr = funding["funding_rate_pct"].to_list()

        sts = df["timestamp_ms"].to_list()
        sc = df["close"].to_list()
        N = len(sts)
        if N < 2:
            return [0.0] * N

        sig = [0.0] * N
        in_pos = False
        for i in range(1, N):
            prev_ts = sts[i - 1]
            # Skip bars with null spot price/timestamp — keep prior signal
            if prev_ts is None or sc[i - 1] is None:
                sig[i] = 1.0 if in_pos else 0.0
                continue
            sp_prev = sc[i - 1] if sc[i - 1] > 0 else _near(sts, sc, prev_ts)
            pp_prev = _near(pts, pc, prev_ts)
            rate_prev = _near(fts, fr, prev_ts)
            if sp_prev is None or pp_prev is None or sp_prev <= 0 or pp_prev <= 0:
                sig[i] = 1.0 if in_pos else 0.0
                continue
            theory = self._theoretical_price(sp_prev, (rate_prev or 0.0) / 100.0)
            dev = (pp_prev - theory) / theory * 100.0
            if not in_pos and abs(dev) > self.entry_threshold_pct:
                in_pos = True
            elif in_pos and abs(dev) < self.exit_threshold_pct:
                in_pos = False
            sig[i] = 1.0 if in_pos else 0.0
        return sig

    def run(self, pair: str = "BTCUSDT") -> StrategyArtifacts:
        """Full backtest with proper delta-neutral PnL on both legs."""
        self.set_pair(pair)
        plower = pair.lower()

        spot = (
            pl.scan_parquet(str(settings.raw_dir / "klines" / f"{plower}_1h" / "**" / "*.parquet"))
            .sort("timestamp_ms").select(["timestamp_ms", "close"]).collect()
        )
        perp, funding = self._load_supplementary(pair)
        if perp is None or funding is None:
            raise RuntimeError(f"Missing perp_klines or funding data for {pair}")

        spot = spot.filter(
            pl.col("timestamp_ms").is_not_null() & pl.col("close").is_not_null()
        )
        perp = perp.filter(
            pl.col("timestamp_ms").is_not_null() & pl.col("close").is_not_null()
        )
        funding = funding.filter(
            pl.col("timestamp_ms").is_not_null() & pl.col("funding_rate_pct").is_not_null()
        )
        sts = spot["timestamp_ms"].to_list(); sc = spot["close"].to_list()
        pts = perp["timestamp_ms"].to_list(); pc = perp["close"].to_list()
        fr = funding["funding_rate_pct"].to_list(); fts = funding["timestamp_ms"].to_list()

        entry_cost = COST_MODEL.entry_cost(pair, use_perp=True)
        exit_cost = COST_MODEL.exit_cost(pair, use_perp=True)

        capital = 10_000.0
        equity = [(fts[0], capital)]
        trades: list[dict] = []
        sig = [0.0] * len(fts)
        in_pos = False
        pos_cap = 0.0
        pos_entry_p = 0.0
        pos_entry_s = 0.0
        pos_type = ""

        for i in range(1, len(fts)):
            ts = fts[i]
            rate_prev = fr[i - 1]
            if ts is None or rate_prev is None:
                continue
            sp = _near(sts, sc, ts)
            pp = _near(pts, pc, ts)
            if sp <= 0 or pp <= 0:
                continue

            theory = self._theoretical_price(sp, rate_prev / 100.0)
            dev = (pp - theory) / theory * 100.0

            # ── Exit (close both legs) ──
            if in_pos and abs(dev) < self.exit_threshold_pct:
                if pos_type == "short_perp_long_spot":
                    perp_pnl = (pos_entry_p - pp) / pos_entry_p * pos_cap
                    spot_pnl = (sp - pos_entry_s) / pos_entry_s * pos_cap
                else:  # long_perp_short_spot
                    perp_pnl = (pp - pos_entry_p) / pos_entry_p * pos_cap
                    spot_pnl = (pos_entry_s - sp) / pos_entry_s * pos_cap
                gross = perp_pnl + spot_pnl
                net = gross - 2.0 * exit_cost * pos_cap   # both legs close
                capital += net
                trades.append({"entry_bar": i, "pnl": net, "pos_type": pos_type})
                in_pos = False

            # ── Entry (open both legs) ──
            if not in_pos and abs(dev) > self.entry_threshold_pct:
                pos_cap = capital * self.position_size_pct
                pos_entry_p = pp
                pos_entry_s = sp
                pos_type = "short_perp_long_spot" if dev > 0 else "long_perp_short_spot"
                in_pos = True
                capital -= 2.0 * entry_cost * pos_cap     # both legs open

            if in_pos:
                sig[i] = 1.0

            # ── Mark-to-market equity (BOTH legs) ──
            eq = capital
            if in_pos and pos_entry_p > 0:
                if pos_type == "short_perp_long_spot":
                    perp_mtm = (pos_entry_p - pp) / pos_entry_p * pos_cap
                    spot_mtm = (sp - pos_entry_s) / pos_entry_s * pos_cap
                else:
                    perp_mtm = (pp - pos_entry_p) / pos_entry_p * pos_cap
                    spot_mtm = (pos_entry_s - sp) / pos_entry_s * pos_cap
                eq += perp_mtm + spot_mtm
            equity.append((ts, max(eq, 0.01)))

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
            "entry_threshold_pct": self.entry_threshold_pct,
            "exit_threshold_pct": self.exit_threshold_pct,
            "position_size_pct": self.position_size_pct,
            "risk_free": self.risk_free,
        }


def _near(arr_ts, arr_val, target):
    """Binary search for the largest arr_ts <= target, return corresponding arr_val.

    Returns 0.0 on no-match / null inputs so callers can use `if val <= 0` to skip.
    Tolerates null entries in arr_ts (skips to the right half).
    """
    if target is None or not arr_ts:
        return 0.0
    lo, hi = 0, len(arr_ts) - 1
    best = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        mid_ts = arr_ts[mid]
        mid_val = arr_val[mid] if mid < len(arr_val) else None
        if mid_ts is None or mid_val is None:
            lo = mid + 1
            continue
        if mid_ts <= target:
            best = mid_val
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def run_strategy(pair: str = "BTCUSDT"):
    s = HeArbitrageStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
