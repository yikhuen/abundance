"""Transaction cost model for backtesting.

Uses exchange-published fee tiers (verifiable via CCXT or exchange docs)
and conservative slippage estimates based on market microstructure.

All parameters are configurable and transparent — no black-box guesses.

Sources:
  - Maker/taker fees: Binance docs (https://www.binance.com/en/fee/schedule)
  - Spread estimates: conservative median from research (Alexander & Dakos, 2023)
"""

from dataclasses import dataclass


@dataclass
class CostModel:
    """Per-trade cost model with fees, slippage, and spread.

    All values are fractions (0.001 = 0.1%).

    Default tier: Binance VIP 0 (highest fees, most conservative).
    """

    maker_fee: float = 0.0002  # 0.02% — Binance spot maker
    taker_fee: float = 0.0004  # 0.04% — Binance spot taker
    perp_maker_fee: float = 0.0002  # 0.02% — Binance USDT-margined maker
    perp_taker_fee: float = 0.0005  # 0.05% — Binance USDT-margined taker

    # Conservative spread estimates (median observed, rounded up)
    # Sources: Binance order book snapshots, Kaiko research
    spread_btc: float = 0.0001  # 0.01% — BTCUSDT typical
    spread_eth: float = 0.0002  # 0.02% — ETHUSDT typical
    spread_sol: float = 0.0005  # 0.05% — SOLUSDT (wider, thinner books)
    spread_default: float = 0.001  # 0.1% — conservative fallback

    # Slippage estimates (price impact of market orders)
    # Conservative: assumes worst-case for size up to $10k notional
    slippage_btc: float = 0.0002  # 0.02%
    slippage_eth: float = 0.0003  # 0.03%
    slippage_sol: float = 0.0010  # 0.10%
    slippage_default: float = 0.0010  # 0.10%

    def spread(self, pair: str) -> float:
        """Get spread fraction for a trading pair."""
        pair_upper = pair.upper()
        if "BTC" in pair_upper:
            return self.spread_btc
        elif "ETH" in pair_upper:
            return self.spread_eth
        elif "SOL" in pair_upper:
            return self.spread_sol
        return self.spread_default

    def slippage(self, pair: str) -> float:
        """Get slippage fraction for a trading pair."""
        pair_upper = pair.upper()
        if "BTC" in pair_upper:
            return self.slippage_btc
        elif "ETH" in pair_upper:
            return self.slippage_eth
        elif "SOL" in pair_upper:
            return self.slippage_sol
        return self.slippage_default

    def round_trip_cost(self, pair: str, use_perp: bool = False) -> float:
        """Total round-trip cost as fraction of trade value.

        Includes: entry taker fee + exit taker fee + 2x spread + 2x slippage
        """
        taker = self.perp_taker_fee if use_perp else self.taker_fee
        spread = self.spread(pair)
        slip = self.slippage(pair)
        return (2 * taker) + (2 * spread) + (2 * slip)

    def entry_cost(self, pair: str, use_perp: bool = False) -> float:
        """One-way entry cost."""
        taker = self.perp_taker_fee if use_perp else self.taker_fee
        return taker + self.spread(pair) + self.slippage(pair)

    def exit_cost(self, pair: str, use_perp: bool = False) -> float:
        """One-way exit cost."""
        taker = self.perp_taker_fee if use_perp else self.taker_fee
        return taker + self.spread(pair) + self.slippage(pair)

    def apply_round_trip(
        self,
        gross_pnl_pct: float,
        pair: str,
        use_perp: bool = False,
    ) -> float:
        """Apply round-trip costs to a gross PnL percentage.

        Args:
            gross_pnl_pct: Pre-cost PnL as percentage (e.g. 1.5 for 1.5%).
            pair: Trading pair.
            use_perp: Whether using perpetual futures.

        Returns:
            Net PnL after costs.
        """
        cost = self.round_trip_cost(pair, use_perp)
        return gross_pnl_pct - cost * 100  # convert fraction to percentage


# ── Default instance ──────────────────────────────────────────

COST_MODEL = CostModel()

# ── Fetch live fees from exchange (for production use) ──────


def fetch_live_fees(exchange_id: str = "binance") -> CostModel | None:
    """Fetch actual fee tiers from exchange via CCXT.

    Returns updated CostModel with real data, or None if fetch fails.
    """
    try:
        import ccxt

        exchange = getattr(ccxt, exchange_id)()
        exchange.load_markets()

        # CCXT returns fees per-market. Take the first spot market as representative.
        spot_market = None
        perp_market = None
        for symbol, market in exchange.markets.items():
            if market.get("spot") and spot_market is None:
                spot_market = market
            if market.get("swap") and perp_market is None:
                perp_market = market
            if spot_market and perp_market:
                break

        maker = spot_market.get("maker", 0.0002) if spot_market else 0.0002
        taker = spot_market.get("taker", 0.0004) if spot_market else 0.0004

        perp_maker = perp_market.get("maker", 0.0002) if perp_market else 0.0002
        perp_taker = perp_market.get("taker", 0.0005) if perp_market else 0.0005

        return CostModel(
            maker_fee=maker,
            taker_fee=taker,
            perp_maker_fee=perp_maker,
            perp_taker_fee=perp_taker,
        )
    except Exception as e:
        from loguru import logger

        logger.warning(f"Failed to fetch live fees: {e}")
        return None
