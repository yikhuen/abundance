#!/usr/bin/env python3
"""Fetch 150 crypto pairs (daily data) for cross-sectional momentum.

Downloads 1d klines from Binance Vision for top liquid USDT pairs.
Background task — expected runtime ~60-90 minutes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import polars as pl
from loguru import logger

from abundance.config.settings import settings
from abundance.data.binance_vision import BinanceVisionFetcher

# Top 150 crypto pairs by market cap with Binance spot listing
# Sourced from CoinGecko top-200, filtered for Binance USDT pairs
PAIRS = [
    # Top 50 by market cap
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT", "ATOMUSDT",
    "MATICUSDT", "LTCUSDT", "ETCUSDT", "FILUSDT", "APTUSDT", "ARBUSDT",
    "NEARUSDT", "OPUSDT", "INJUSDT", "TIAUSDT", "IMXUSDT", "STXUSDT",
    "GRTUSDT", "THETAUSDT", "RUNEUSDT", "FTMUSDT", "ALGOUSDT", "SANDUSDT",
    "MANAUSDT", "AXSUSDT", "EGLDUSDT", "VETUSDT", "ICPUSDT", "HBARUSDT",
    "MAKERUSDT", "AAVEUSDT", "CRVUSDT", "SNXUSDT", "COMPUSDT", "YFIUSDT",
    "SUSHIUSDT", "1INCHUSDT", "BATUSDT", "ZRXUSDT", "ENJUSDT", "CHZUSDT",
    "LRCUSDT", "KAVAUSDT",
    # 51-100
    "FLOWUSDT", "GALAUSDT", "XTZUSDT", "CELOUSDT", "QTUMUSDT", "IOTAUSDT",
    "ZILUSDT", "ICXUSDT", "WAVESUSDT", "ONTUSDT", "DASHUSDT", "NEOUSDT",
    "ZECUSDT", "DYDXUSDT", "GMXUSDT", "MINAUSDT", "ROSEUSDT", "KSMUSDT",
    "RNDRUSDT", "FETUSDT", "AGIXUSDT", "OCEANUSDT", "CFXUSDT", "SUIUSDT",
    "PEPEUSDT", "WLDUSDT", "BLURUSDT", "SEIUSDT", "STRKUSDT", "JUPUSDT",
    "PYTHUSDT", "BONKUSDT", "ENAUSDT", "WIFUSDT", "ORDIUSDT", "1000SATSUSDT",
    "MEMEUSDT", "NOTUSDT", "PEOPLEUSDT", "ACEUSDT", "NFPUSDT", "AIUSDT",
    "XAIUSDT", "PORTALUSDT", "PIXELUSDT", "ALTUSDT", "ETHFIUSDT", "ENAUSDT",
    "SAGAUSDT", "TAOUSDT",
    # 101-150
    "TONUSDT", "XMRUSDT", "OKBUSDT", "CAKEUSDT", "BAKEUSDT", "TRXUSDT",
    "BCHUSDT", "XLMUSDT", "EOSUSDT", "KLAYUSDT", "ARUSDT", "ENSUSDT",
    "LDOUSDT", "MASKUSDT", "DYDXUSDT", "SSVUSDT", "PENDLEUSDT", "LQTYUSDT",
    "RDNTUSDT", "RPLUSDT", "FXSUSDT", "CVXUSDT", "BALUSDT", "ONEUSDT",
    "HOTUSDT", "ANKRUSDT", "COTIUSDT", "CTSIUSDT", "BANDUSDT", "STORJUSDT",
    "OCEANUSDT", "KNCUSDT", "OMGUSDT", "REEFUSDT", "SKLUSDT", "CVCUSDT",
    "DENTUSDT", "STMXUSDT", "SUNUSDT", "WINUSDT", "BTTUSDT", "TUSDT",
    "JASMYUSDT", "XECUSDT", "ACHUSDT", "JOEUSDT", "SPELLUSDT", "GTCUSDT",
    "NKNUSDT", "OGNUSDT",
]


def main() -> None:
    logger.info(f"Fetching daily data for {len(PAIRS)} pairs")
    settings.raw_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    errors = 0

    for pair in PAIRS:
        done += 1
        try:
            fetcher = BinanceVisionFetcher(
                symbol=pair,
                interval="1d",
                market_type="spot",
                output_dir=settings.raw_dir / "klines",
            )
            df = fetcher.fetch(start_date="2018-01-01", end_date=None, mode="monthly")
            if df.is_empty():
                logger.debug(f"[{done}/{len(PAIRS)}] {pair}: no data (skip)")
                errors += 1
                continue

            df = df.with_columns(
                pl.from_epoch("timestamp_ms", time_unit="ms").dt.year().alias("year"),
                pl.from_epoch("timestamp_ms", time_unit="ms").dt.month().alias("month"),
            )
            fetcher.save_parquet(df, partition_cols=["year", "month"])
            logger.info(f"[{done}/{len(PAIRS)}] {pair}: {len(df):,} rows OK")
        except Exception as e:
            logger.warning(f"[{done}/{len(PAIRS)}] {pair}: {e}")
            errors += 1

    logger.info(f"Done: {done - errors}/{done} pairs fetched, {errors} errors")


if __name__ == "__main__":
    main()
