"""Binance Testnet trading client — direct REST API (no CCXT).

CCXT's binance class has trouble with testnet URL routing for SAPI/margin
endpoints. This module uses the Binance Futures REST API directly, which is
simpler and more reliable for testnet.
"""

import hashlib
import hmac
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from loguru import logger


# Load from env or use defaults
API_KEY = os.getenv("BINANCE_TESTNET_API_KEY", "")
API_SECRET = os.getenv("BINANCE_TESTNET_SECRET", "")
BASE_URL = "https://testnet.binancefuture.com"


@dataclass
class TestnetClient:
    """Binance Futures Testnet client.

    Handles authentication, order placement, balance queries,
    and funding rate monitoring. All operations are on testnet —
    no real money.
    """

    api_key: str
    api_secret: str
    base_url: str = BASE_URL
    session: requests.Session | None = None

    def __post_init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": self.api_key})

    def _sign(self, params: dict) -> str:
        """Create HMAC-SHA256 signature."""
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        return hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()

    def _request(self, method: str, path: str, params: dict | None = None) -> dict:
        """Make an authenticated request to the testnet API."""
        params = params or {}
        params["timestamp"] = int(time.time() * 1000)
        params["signature"] = self._sign(params)

        url = f"{self.base_url}{path}"
        if method == "GET":
            resp = self.session.get(url, params=params)
        else:
            resp = self.session.post(url, data=params)

        resp.raise_for_status()
        data = resp.json()

        if isinstance(data, dict) and "code" in data and data["code"] < 0:
            raise RuntimeError(f"API error {data['code']}: {data.get('msg', '')}")

        return data

    def _get_public(self, path: str, params: dict | None = None) -> dict:
        """Make an unauthenticated public request."""
        resp = self.session.get(f"{self.base_url}{path}", params=params)
        resp.raise_for_status()
        return resp.json()

    # ── Public data ──────────────────────────────────────────────

    def get_price(self, symbol: str) -> float:
        """Get current mark price."""
        data = self._get_public(
            "/fapi/v1/ticker/price", {"symbol": symbol}
        )
        return float(data["price"])

    def get_funding_rate(self, symbol: str) -> float:
        """Get latest funding rate (as fraction, e.g. 0.0001 = 0.01%)."""
        data = self._get_public(
            "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1}
        )
        return float(data[0]["fundingRate"])

    # ── Private data ─────────────────────────────────────────────

    def get_balance(self) -> dict[str, float]:
        """Get USDT balance on testnet."""
        data = self._request("GET", "/fapi/v2/balance")
        balances = {}
        for entry in data:
            asset = entry["asset"]
            bal = float(entry["balance"])
            if bal > 0:
                balances[asset] = bal
        return balances

    def get_positions(self) -> list[dict]:
        """Get open positions."""
        data = self._request("GET", "/fapi/v2/positionRisk")
        return [
            p for p in data
            if float(p.get("positionAmt", 0)) != 0
        ]

    # ── Trading ──────────────────────────────────────────────────

    def place_market_order(
        self,
        symbol: str,
        side: str,  # "BUY" or "SELL"
        quantity: float,
        reduce_only: bool = False,
    ) -> dict:
        """Place a market order on testnet.

        WARNING: This executes on TESTNET only. No real money.
        """
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": str(quantity),
            "reduceOnly": "true" if reduce_only else "false",
        }
        result = self._request("POST", "/fapi/v1/order", params)
        logger.info(
            f"📌 {side.upper()} {quantity} {symbol} @ MARKET | "
            f"OrderID: {result.get('orderId', '?')}"
        )
        return result

    def set_leverage(self, symbol: str, leverage: int = 1) -> dict:
        """Set leverage for a symbol (required before trading)."""
        params = {"symbol": symbol, "leverage": leverage}
        return self._request("POST", "/fapi/v1/leverage", params)

    # ── Health check ─────────────────────────────────────────────

    def ping(self) -> bool:
        """Verify connection to testnet."""
        try:
            price = self.get_price("BTCUSDT")
            bal = self.get_balance()
            usdt = bal.get("USDT", 0)
            logger.info(
                f"✅ Testnet OK | BTC: ${price:,.2f} | USDT: ${usdt:,.2f}"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Testnet connection failed: {e}")
            return False


def get_testnet_client() -> TestnetClient:
    """Factory: create a TestnetClient from env vars."""
    return TestnetClient(api_key=API_KEY, api_secret=API_SECRET)
