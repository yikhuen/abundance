#!/usr/bin/env python3
"""Sprint 3 · Stage 3.1: Pull historical funding rates for BTC, ETH, SOL.

Funding rate data is the input for the carry strategy.
Stored as partitioned Parquet alongside existing kline data.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.config.settings import settings
from abundance.data.funding_rates import FundingRateFetcher
from abundance.data.storage import MarketDataStore


PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def main() -> None:
    """Fetch funding rates for all pairs and register in DuckDB."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 3 · Stage 3.1")
    logger.info("Funding Rate Data Pipeline")
    logger.info("=" * 60)

    output_dir = settings.raw_dir / "funding"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for pair in PAIRS:
        logger.info(f"--- {pair} ---")
        try:
            fetcher = FundingRateFetcher(
                symbol=pair,
                exchange="binance",
                output_dir=output_dir,
            )

            df = fetcher.fetch(start_date="2020-01-01", end_date=None)

            if df.is_empty():
                logger.warning(f"No funding data for {pair}")
                continue

            # Add partition columns
            df = df.with_columns(
                pl.from_epoch("timestamp_ms", time_unit="ms").dt.year().alias("year"),
                pl.from_epoch("timestamp_ms", time_unit="ms").dt.month().alias("month"),
            )

            parquet_path = fetcher.save_parquet(df, partition_cols=["year", "month"])

            ts_min = datetime.fromtimestamp(
                df["timestamp_ms"].min() / 1000, tz=timezone.utc
            ).isoformat()[:10]
            ts_max = datetime.fromtimestamp(
                df["timestamp_ms"].max() / 1000, tz=timezone.utc
            ).isoformat()[:10]

            results.append((pair, len(df), ts_min, ts_max))
            logger.info(f"  {pair}: {len(df):,} records · {ts_min} → {ts_max}")

        except Exception as e:
            logger.error(f"Failed {pair}: {e}")

    # ── Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Funding Rate Summary:")
    total = 0
    for pair, rows, start, end in results:
        logger.info(f"  {pair:<10} {rows:>8,} records  {start} → {end}")
        total += rows
    logger.info(f"  {'TOTAL':<10} {total:>8,} records")
    logger.info("=" * 60)

    # ── DuckDB ───────────────────────────────────────────────
    try:
        with MarketDataStore(settings.duckdb_path) as store:
            for pair, _, _, _ in results:
                dir_name = pair.lower()
                table_name = f"{dir_name}_funding"
                glob_pattern = str(
                    output_dir / dir_name / "**" / "*.parquet"
                )
                store.register_parquet_glob(table_name, glob_pattern)
            logger.info(f"Funding tables registered at: {store.db_path}")
    except Exception as e:
        logger.warning(f"DuckDB registration skipped: {e}")

    logger.info("Stage 3.1 — COMPLETE")


if __name__ == "__main__":
    main()
