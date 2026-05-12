#!/usr/bin/env python3
"""Sprint 1 · Story 1.1 · Task 2: Pull additional timeframes for BTCUSDT.

Downloads 5m, 15m, 4h, 1d klines from Binance Vision and saves as
partitioned Parquet alongside the existing 1h data.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.config.settings import settings
from abundance.data.binance_vision import BinanceVisionFetcher
from abundance.data.storage import MarketDataStore

# Intervals to fetch (1h is already done, skip it)
TIMEFRAMES = ["5m", "15m", "4h", "1d"]

# One-liner descriptions for logs
TIMEFRAME_LABELS = {
    "5m": "5-minute",
    "15m": "15-minute",
    "4h": "4-hour",
    "1d": "daily",
}


def fetch_interval(interval: str) -> tuple[str, int, str, str]:
    """Fetch a single interval, return (interval, rows, start, end)."""
    label = TIMEFRAME_LABELS[interval]
    logger.info(f"--- Fetching {label} ({interval}) klines ---")

    fetcher = BinanceVisionFetcher(
        symbol="BTCUSDT",
        interval=interval,
        market_type="spot",
        output_dir=settings.raw_dir / "klines",
    )

    df = fetcher.fetch(
        start_date="2017-08-17",
        end_date=None,
        mode="monthly",
    )

    if df.is_empty():
        logger.error(f"No data for {interval}")
        return interval, 0, "N/A", "N/A"

    # Standardize schema (normalizes timestamps to ms, renames columns)
    df = fetcher.standardize_schema(df)

    # Add partition columns
    df = df.with_columns(
        pl.from_epoch("timestamp_ms", time_unit="ms").dt.year().alias("year"),
        pl.from_epoch("timestamp_ms", time_unit="ms").dt.month().alias("month"),
    )

    # Save to Parquet
    parquet_path = fetcher.save_parquet(df, partition_cols=["year", "month"])

    # Quick validation
    ts_min_ms = df["timestamp_ms"].min()
    ts_max_ms = df["timestamp_ms"].max()
    row_count = len(df)

    ts_min = datetime.fromtimestamp(ts_min_ms / 1000, tz=timezone.utc).isoformat()
    ts_max = datetime.fromtimestamp(ts_max_ms / 1000, tz=timezone.utc).isoformat()

    logger.info(f"  {label}: {row_count:,} rows · {ts_min[:10]} → {ts_max[:10]}")

    return interval, row_count, ts_min, ts_max


def main() -> None:
    """Fetch all additional timeframes and register in DuckDB."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 1 · Story 1.1 · Task 2")
    logger.info("Fetching additional BTCUSDT timeframes")
    logger.info("=" * 60)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for interval in TIMEFRAMES:
        try:
            result = fetch_interval(interval)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to fetch {interval}: {e}")
            results.append((interval, 0, "ERROR", "ERROR"))

    # ── Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Fetch Summary:")
    total_rows = 0
    for interval, rows, start, end in results:
        logger.info(f"  {interval:>4s}  {rows:>10,} rows  {start[:10]} → {end[:10]}")
        total_rows += rows
    logger.info(f"  {'TOTAL':>4s}  {total_rows:>10,} rows (new timeframes only)")
    logger.info("=" * 60)

    # ── Register in DuckDB ──────────────────────────────────
    try:
        with MarketDataStore(settings.duckdb_path) as store:
            for interval, _, _, _ in results:
                glob_pattern = str(
                    settings.raw_dir / "klines" / f"btcusdt_{interval}"
                    / "**" / "*.parquet"
                )
                table_name = f"btcusdt_{interval}"
                store.register_parquet_glob(table_name, glob_pattern)
            logger.info(f"DuckDB tables registered at: {store.db_path}")
    except Exception as e:
        logger.warning(f"DuckDB registration skipped: {e}")

    logger.info("Sprint 1 · Story 1.1 · Task 2 — COMPLETE")


if __name__ == "__main__":
    main()
