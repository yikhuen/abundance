"""Application configuration via environment variables and dotenv."""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration loaded from environment variables and .env file.

    All paths are resolved relative to the project root.
    """

    model_config = SettingsConfigDict(
        env_prefix="ABUNDANCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Data paths ──────────────────────────────────────────────
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")

    # ── Binance Vision ──────────────────────────────────────────
    binance_vision_base_url: str = "https://data.binance.vision"
    binance_vision_checksum_url: str = (
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1h/"
    )

    # ── CCXT ────────────────────────────────────────────────────
    ccxt_default_exchange: str = "binance"
    ccxt_rate_limit_ms: int = 200

    # ── DuckDB / Storage ────────────────────────────────────────
    duckdb_path: Path = Path("data/processed/market_data.duckdb")
    parquet_compression: str = "zstd"

    # ── Logging ─────────────────────────────────────────────────
    log_level: str = "INFO"

    def model_post_init(self, _context) -> None:
        """Resolve relative paths against the project root."""
        project_root = Path(__file__).resolve().parents[3]
        for field_name in ("data_dir", "raw_dir", "processed_dir", "duckdb_path"):
            val = getattr(self, field_name)
            if isinstance(val, Path) and not val.is_absolute():
                object.__setattr__(self, field_name, (project_root / val).resolve())


# Singleton instance
settings = Settings()
