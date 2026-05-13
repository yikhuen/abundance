"""Close all open testnet positions."""
import sys, os, time, hmac, hashlib, requests
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
os.chdir(Path(__file__).resolve().parents[1])
from dotenv import load_dotenv
load_dotenv()
from abundance.paper_trading.testnet_client import get_testnet_client

client = get_testnet_client()
key = client.api_key; secret = client.api_secret; base = client.base_url

positions = client.get_positions()
print("Closing all positions:")
for p in positions:
    symbol = p['symbol']
    amt = float(p['positionAmt'])
    if amt == 0: continue
    side = 'SELL' if amt > 0 else 'BUY'
    qty = abs(amt)
    
    ts = int(time.time() * 1000)
    params = {'symbol': symbol, 'side': side, 'type': 'MARKET',
              'quantity': qty, 'timestamp': ts}
    query = '&'.join(f'{k}={v}' for k, v in sorted(params.items()))
    sig = hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()
    r = requests.post(f'{base}/fapi/v1/order?{query}&signature={sig}',
                      headers={'X-MBX-APIKEY': key})
    resp = r.json()
    print(f"  {symbol}: CLOSE {qty} → {resp.get('status', resp)}")

time.sleep(2)
remaining = [p for p in client.get_positions() if abs(float(p['positionAmt'])) > 0]
print(f"Positions remaining: {len(remaining)}")
if not remaining:
    print("✅ All positions closed. Ready for clean restart.")
