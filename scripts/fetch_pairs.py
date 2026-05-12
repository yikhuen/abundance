#!/usr/bin/env python3
"""Sprint 1 · Story 1.1 · Task 3a: Fetch ETHUSDT + SOLUSDT historical klines.

Reuses BinanceVisionFetcher across additional pairs and all 5 timeframes.
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

# Pairs and timeframes to fetch
PAIRS = ["ETHUSDT", "SOLUSDT"]
TIMEFRAMES = ["1h", "4h", "1d", "5m", "15m"]

# Each pair's listing date (approximate first Binance listing)
LISTING_DATES = {
    "ETHUSDT": "2017-08-17",
    "SOLUSDT": "2020-08-11",
}


def fetch_pair_interval(pair: str, interval: str) -> tuple[str, str, int, str, str]:
    """Fetch one pair × interval, return (pair, interval, rows, start, end)."""
    fetcher = BinanceVisionFetcher(
        symbol=pair,
        interval=interval,
        market_type="spot",
        output_dir=settings.raw_dir / "klines",
    )

    start_date = LISTING_DATES.get(pair, "2017-08-17")
    logger.info(f"  Fetching {pair} {interval} (since {start_date})...")

    df = fetcher.fetch(start_date=start_date, end_date=None, mode="monthly")

    if df.is_empty():
        logger.warning(f"  No data for {pair} {interval}")
        return pair, interval, 0, "N/A", "N/A"

    df = fetcher.standardize_schema(df)

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

    return pair, interval, len(df), ts_min, ts_max


def main() -> None:
    """Fetch all pairs × timeframes and register in DuckDB."""
    logger.info("=" * 60)
    logger.info("Abundance · Sprint 1 · Story 1.1 · Task 3a")
    logger.info(f"Fetching {', '.join(PAIRS)} × {', '.join(TIMEFRAMES)}")
    logger.info("=" * 60)

    settings.raw_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_combos = len(PAIRS) * len(TIMEFRAMES)
    done = 0

    for pair in PAIRS:
        logger.info(f"--- {pair} ---")
        for interval in TIMEFRAMES:
            done += 1
            logger.info(f"[{done}/{total_combos}] {pair} {interval}")
            try:
                result = fetch_pair_interval(pair, interval)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed {pair} {interval}: {e}")
                results.append((pair, interval, 0, "ERROR", "ERROR"))

    # ── Summary ─────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"{'Pair':<10} {'TF':>5} {'Rows':>12} {'Range'}")
    logger.info("-" * 60)
    total = 0
    for pair, tf, rows, start, end in results:
        logger.info(f"{pair:<10} {tf:>5} {rows:>12,} {start} → {end}")
        total += rows
    logger.info("-" * 60)
    logger.info(f"{'TOTAL':>16} {total:>12,} rows")
    logger.info("=" * 60)

    # ── DuckDB ───────────────────────────────────────────────
    try:
        with MarketDataStore(settings.duckdb_path) as store:
            for pair, tf, _, _, _ in results:
                table_name = f"{pair.lower()}_{tf}"
                glob_pattern = str(
                    settings.raw_dir / "klines" / table_name / "**" / "*.parquet"
                )
                store.register_parquet_glob(table_name, glob_pattern)
            logger.info(f"DuckDB tables registered at: {store.db_path}")
    except Exception as e:
        logger.warning(f"DuckDB registration skipped: {e}")

    logger.info("Sprint 1 · Story 1.1 · Task 3a — COMPLETE")


if __name__ == "__main__":
    main()
