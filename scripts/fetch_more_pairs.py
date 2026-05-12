#!/usr/bin/env python3
"""Fetch additional crypto pairs for multi-asset AdaptiveTrend.

Adds top-10 liquid pairs with sufficient history (2020+):
  BNB, ADA, XRP, DOGE, DOT, AVAX, LINK, UNI, MATIC, ATOM
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.config.settings import settings
from abundance.data.binance_vision import BinanceVisionFetcher

NEW_PAIRS = [
    "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "DOTUSDT",
    "AVAXUSDT", "LINKUSDT", "UNIUSDT", "MATICUSDT", "ATOMUSDT",
]
TIMEFRAMES = ["1d", "4h"]  # daily for rebalancing, 4h for signals
LISTING_DATES = {
    "BNBUSDT": "2017-11-01",
    "ADAUSDT": "2018-04-17",
    "XRPUSDT": "2018-05-04",
    "DOGEUSDT": "2019-07-05",
    "DOTUSDT": "2020-08-19",
    "AVAXUSDT": "2020-09-22",
    "LINKUSDT": "2019-01-16",
    "UNIUSDT": "2020-09-17",
    "MATICUSDT": "2019-04-26",
    "ATOMUSDT": "2019-04-29",
}


def main() -> None:
    logger.info(f"Fetching {len(NEW_PAIRS)} pairs × {len(TIMEFRAMES)} timeframes")
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    total = len(NEW_PAIRS) * len(TIMEFRAMES)
    done = 0

    for pair in NEW_PAIRS:
        start = LISTING_DATES.get(pair, "2020-01-01")
        for tf in TIMEFRAMES:
            done += 1
            logger.info(f"[{done}/{total}] {pair} {tf}")

            try:
                fetcher = BinanceVisionFetcher(
                    symbol=pair,
                    interval=tf,
                    market_type="spot",
                    output_dir=settings.raw_dir / "klines",
                )
                df = fetcher.fetch(start_date=start, end_date=None, mode="monthly")
                if df.is_empty():
                    logger.warning(f"  No data for {pair} {tf}")
                    continue

                df = df.with_columns(
                    pl.from_epoch("timestamp_ms", time_unit="ms").dt.year().alias("year"),
                    pl.from_epoch("timestamp_ms", time_unit="ms").dt.month().alias("month"),
                )
                fetcher.save_parquet(df, partition_cols=["year", "month"])
                logger.info(f"  {pair} {tf}: {len(df):,} rows")
            except Exception as e:
                logger.error(f"  Failed {pair} {tf}: {e}")

    logger.info("Done fetching additional pairs")


if __name__ == "__main__":
    main()
