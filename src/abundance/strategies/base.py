"""Strategy Interface — Abstract Base Class.

Every strategy must implement:
- signals(df) → polars Series of position signals [0.0 or 1.0]
- trades(signals) → list of {entry_bar, exit_bar, entry_price, exit_price, pnl}
- equity_curve(df, signals) → polars DataFrame with equity

The harness drives: load data → signals → apply costs → trades → metrics.
"""
import polars as pl
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from abundance.backtesting.costs import COST_MODEL


@dataclass
class StrategyArtifacts:
    """All outputs a strategy produces after running on a pair."""
    signals: list[float]           # position signal per bar [0.0 or 1.0]
    equity_curve: pl.DataFrame     # columns: timestamp_ms, equity
    metrics: "MetricsReport"       # risk/return stats
    trades: list[dict] = field(default_factory=list)  # individual trade records
    params: dict = field(default_factory=dict)         # strategy parameters used
    pair: str = "BTCUSDT"          # pair this was run on


class Strategy(ABC):
    """Abstract base for all trading strategies."""

    def __init__(self):
        self._pair = "BTCUSDT"

    @property
    def pair(self) -> str:
        return self._pair

    @abstractmethod
    def signals(self, df: pl.DataFrame) -> list[float]:
        """Compute position signals.

        Args:
            df: Full OHLCV dataframe (sorted by timestamp_ms).
                Columns: open, high, low, close, volume, timestamp_ms.

        Returns:
            List of floats, length = len(df). sig[t] ∈ {0.0, 1.0}.
            sig[t] is the position held DURING bar t, decided using
            data available at the END of bar t-1 (no lookahead).
        """
        ...

    def apply_costs(
        self,
        signals: list[float],
        df: pl.DataFrame,
        use_perp: bool = True,
    ) -> list[float]:
        """Apply transaction costs to strategy returns.

        Charges entry_cost on 0→1 transitions and exit_cost on 1→0
        transitions (each ≈ half the round-trip). No double-charging.

        Returns list of per-bar net returns (strategy return - costs).
        """
        close = df["close"].to_list()
        N = len(close)
        ret = [0.0] + [
            (close[i] / close[i - 1] - 1) for i in range(1, N)
        ]
        pair = self._detect_pair(df)

        entry_cost = COST_MODEL.entry_cost(pair, use_perp=use_perp)
        exit_cost = COST_MODEL.exit_cost(pair, use_perp=use_perp)

        net_ret = [0.0] * N
        for i in range(1, N):
            gross = ret[i] * signals[i]
            txn = 0.0
            if signals[i] > signals[i - 1]:        # entry (0→1)
                txn = entry_cost
            elif signals[i] < signals[i - 1]:      # exit (1→0)
                txn = exit_cost
            net_ret[i] = gross - txn
        return net_ret

    def equity_curve(
        self,
        df: pl.DataFrame,
        signals: list[float],
        net_returns: list[float],
        start_bar: int = 0,
    ) -> pl.DataFrame:
        """Build equity curve from net returns."""
        ts = df["timestamp_ms"].to_list()
        eq = [10000.0]
        for i in range(max(start_bar, 1), len(net_returns)):
            eq.append(eq[-1] * (1 + net_returns[i]))
        return pl.DataFrame({
            "timestamp_ms": ts[max(start_bar, 1):],
            "equity": eq[1:],
        })

    def compute_trades(self, signals: list[float], df: pl.DataFrame) -> list[dict]:
        """Extract trade records from signal transitions."""
        close = df["close"].to_list()
        trades = []
        entry_bar = None
        for i in range(1, len(signals)):
            if signals[i] > signals[i - 1]:  # enter
                entry_bar = i
            elif signals[i] < signals[i - 1] and entry_bar is not None:  # exit
                entry_price = close[entry_bar]
                exit_price = close[i]
                pnl = (exit_price / entry_price - 1)
                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "bars_held": i - entry_bar,
                })
                entry_bar = None
        return trades

    def run(self, pair: str = "BTCUSDT") -> StrategyArtifacts:
        """Full run: load data → signals → costs → equity → metrics.

        Subclasses only need to implement signals(). This method
        handles data loading, cost application, and metrics calculation.
        """
        self.set_pair(pair)
        from abundance.config.settings import settings
        plower = pair.lower()
        df = pl.scan_parquet(
            str(settings.raw_dir / "klines" / f"{plower}_1d" / "**" / "*.parquet")
        ).sort("timestamp_ms").collect()

        sig = self.signals(df)
        net_ret = self.apply_costs(sig, df)
        eq_df = self.equity_curve(df, sig, net_ret)

        from abundance.backtesting.metrics import MetricsCalculator
        mc = MetricsCalculator.from_equity_curve(eq_df)

        trade_list = self.compute_trades(sig, df)
        mc.trades = len(trade_list)

        return StrategyArtifacts(
            signals=sig,
            equity_curve=eq_df,
            metrics=mc,
            trades=trade_list,
            params=self._get_params(),
            pair=pair,
        )

    @abstractmethod
    def _get_params(self) -> dict:
        """Return strategy parameters for artifact metadata."""
        ...

    def _detect_pair(self, df: pl.DataFrame) -> str:
        """Return the pair this strategy is configured for."""
        return self._pair

    def set_pair(self, pair: str):
        self._pair = pair
