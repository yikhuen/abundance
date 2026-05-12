#!/usr/bin/env python3
"""Sprint 1 · Story 1.1 · Task 3b: CCXT live data pipeline.

Fetches recent klines (last 7 days) for BTC, ETH, SOL across all
timeframes via the Binance CCXT API. Complements Binance Vision
historical data with up-to-date candles.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loguru import logger

from abundance.config.settings import settings
from abundance.data.ccxt_fetcher import CCXTFetcher
from abundance.data.storage import MarketDataStore

PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
TIMEFRAMES = ["1h", "4h", "1d", "5m", "15m"]
LOOKBACK_DAYS = 7


def main() -> None:
    """Pull recent klines via CCXT for all pairs × timeframes."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 1 · Story 1.1 · Task 3b")
    logger.info("CCXT Live Data Pipeline")
    logger.info("=" * 60)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(PAIRS) * len(TIMEFRAMES)
    done = 0

    for pair in PAIRS:
        pair_clean = pair.replace("/", "")
        logger.info(f"--- {pair} ---")
        for tf in TIMEFRAMES:
            done += 1
            logger.info(f"[{done}/{total}] {pair} {tf}")

            try:
                fetcher = CCXTFetcher(
                    symbol=pair,
                    exchange="binance",
                    interval=tf,
                    output_dir=settings.raw_dir / "live",
                )

                df = fetcher.fetch_recent(lookback_days=LOOKBACK_DAYS)
                if df.is_empty():
                    logger.warning(f"  No data for {pair} {tf}")
                    results.append((pair_clean, tf, 0))
                    continue

                df = df.with_columns(
                    pl.from_epoch("timestamp_ms", time_unit="ms")
                    .dt.year()
                    .alias("year"),
                    pl.from_epoch("timestamp_ms", time_unit="ms")
                    .dt.month()
                    .alias("month"),
                )

                parquet_path = fetcher.save_parquet(
                    df, partition_cols=["year", "month"]
                )
                logger.info(f"  {len(df)} candles → {parquet_path}")
                results.append((pair_clean, tf, len(df)))

            except Exception as e:
                logger.error(f"  Failed {pair} {tf}: {e}")
                results.append((pair_clean, tf, 0))

    # ── Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CCXT Live Data Summary:")
    total_rows = 0
    for pair, tf, rows in results:
        logger.info(f"  {pair:<8} {tf:>4}  {rows:>8,} rows")
        total_rows += rows
    logger.info(f"  {'TOTAL':>13}  {total_rows:>8,} rows")
    logger.info("=" * 60)

    # ── DuckDB (live tables) ─────────────────────────────────
    try:
        with MarketDataStore(settings.duckdb_path) as store:
            for pair, tf, _ in results:
                dir_name = f"{pair.lower()}_{tf}"
                table_name = f"{dir_name}_live"
                glob_pattern = str(
                    settings.raw_dir / "live" / dir_name / "**" / "*.parquet"
                )
                store.register_parquet_glob(table_name, glob_pattern)
            logger.info(f"Live tables registered at: {store.db_path}")
    except Exception as e:
        logger.warning(f"DuckDB live registration skipped: {e}")

    logger.info("Sprint 1 · Story 1.1 · Task 3b — COMPLETE")


if __name__ == "__main__":
    import polars as pl  # noqa: F811

    main()
