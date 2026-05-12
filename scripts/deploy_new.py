"""Deploy Donchian20 + EMA20 strategies to testnet."""
import sys, os, time, hmac, hashlib, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(Path(__file__).resolve().parents[1])
from dotenv import load_dotenv
load_dotenv()
from abundance.paper_trading.testnet_client import get_testnet_client

client = get_testnet_client()
key = client.api_key; secret = client.api_secret; base = client.base_url

# Check current positions
pos = client.get_positions()
print("Current positions:")
for p in pos:
    print(f"  {p['symbol']}: {p['positionAmt']} @ ${float(p['entryPrice']):,.0f} uPnL=${float(p['unRealizedProfit']):+,.2f}")

bal = client.get_balance()
available = bal.get('USDT', 0)
equity = bal.get('USDT', 5000) + bal.get('USDC', 0)
print(f"\nAvailable USDT: ${available:,.2f} | Equity: ${equity:,.2f}")

# New positions: 5% each
pos_size = equity * 0.05
btc_price = client.get_price('BTCUSDT')
eth_price = client.get_price('ETHUSDT')
print(f"BTC: ${btc_price:,.0f}, ETH: ${eth_price:,.2f}")

donchian_qty = round(pos_size / btc_price, 3)
ema_qty = round(pos_size / eth_price, 3)

print(f"\nDeploying:")
print(f"  Donchian20 → BUY {donchian_qty} BTC (~${pos_size:,.0f})")
print(f"  EMA20 → BUY {ema_qty} ETH (~${pos_size:,.0f})")

def place_market_order(symbol, side, quantity):
    ts = int(time.time() * 1000)
    params = {'symbol': symbol, 'side': side, 'type': 'MARKET',
              'quantity': quantity, 'timestamp': ts}
    query = '&'.join(f'{k}={v}' for k,v in sorted(params.items()))
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    r = requests.post(f'{base}/fapi/v1/order?{query}&signature={sig}',
                      headers={'X-MBX-APIKEY': key})
    return r.json()

r1 = place_market_order('BTCUSDT', 'BUY', donchian_qty)
print(f"  Donchian20 BTC: {r1.get('status', r1)} order #{r1.get('orderId', '?')}")

r2 = place_market_order('ETHUSDT', 'BUY', ema_qty)
print(f"  EMA20 ETH: {r2.get('status', r2)} order #{r2.get('orderId', '?')}")

print("\n✅ Deployed! Check dashboard and testnet.binancefuture.com")
