#!/usr/bin/env python3
"""Sprint 1 · Story 1.1 · Task 1: Pull first batch of Binance Vision data for BTC.

Downloads BTCUSDT 1h klines from 2017-08-17 to yesterday, saves as
partitioned Parquet, and registers in DuckDB for query access.
"""

import sys
from pathlib import Path

# Ensure src/ is on path when running from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.config.settings import settings
from abundance.data.binance_vision import BinanceVisionFetcher
from abundance.data.storage import MarketDataStore


def main() -> None:
    """Download BTCUSDT 1h klines and persist to Parquet + DuckDB."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 1 · Story 1.1 · Task 1")
    logger.info("Fetching BTCUSDT 1h klines from Binance Vision")
    logger.info("=" * 60)

    # Ensure data directories exist
    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Fetch historical klines ──────────────────────────
    fetcher = BinanceVisionFetcher(
        symbol="BTCUSDT",
        interval="1h",
        market_type="spot",
        output_dir=settings.raw_dir / "klines",
    )

    # Start from BTCUSDT inception, up to yesterday
    df = fetcher.fetch(
        start_date="2017-08-17",
        end_date=None,  # defaults to yesterday
        mode="monthly",
    )

    if df.is_empty():
        logger.error("No data fetched. Check network or Binance Vision availability.")
        sys.exit(1)

    # ── Step 2: Save to partitioned Parquet ──────────────────────
    # Add partition columns for year/month
    df = df.with_columns(
        pl.from_epoch("timestamp_ms", time_unit="ms").dt.year().alias("year"),
        pl.from_epoch("timestamp_ms", time_unit="ms").dt.month().alias("month"),
    )

    parquet_path = fetcher.save_parquet(df, partition_cols=["year", "month"])
    logger.info(f"Parquet files written to: {parquet_path}")

    # ── Step 3: Validate with Polars ───────────────────────────
    validate = pl.scan_parquet(str(parquet_path / "**" / "*.parquet"))
    ts_min_ms = validate.select(pl.col("timestamp_ms").min()).collect().item()
    ts_max_ms = validate.select(pl.col("timestamp_ms").max()).collect().item()
    row_count = validate.select(pl.len()).collect().item()

    from datetime import datetime, timezone

    ts_min = datetime.fromtimestamp(ts_min_ms / 1000, tz=timezone.utc).isoformat()
    ts_max = datetime.fromtimestamp(ts_max_ms / 1000, tz=timezone.utc).isoformat()

    logger.info("Parquet validation:")
    logger.info(f"  Range: {ts_min} → {ts_max}")
    logger.info(f"  Rows:  {row_count:,}")

    # ── Step 4: Register in DuckDB (may be slow on /mnt/c/) ─────
    try:
        with MarketDataStore(settings.duckdb_path) as store:
            glob_pattern = str(parquet_path / "**" / "*.parquet")
            store.register_parquet_glob("btcusdt_1h", glob_pattern)
            logger.info(f"DuckDB registered at: {store.db_path}")
    except Exception as e:
        logger.warning(f"DuckDB registration skipped: {e}")

    logger.info("=" * 60)
    logger.info("Sprint 1 · Story 1.1 · Task 1 — COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
