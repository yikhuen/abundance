#!/usr/bin/env python3
"""Abundance Live Dashboard — web-based PnL, trades, and agent monitor.

Usage:
  python scripts/dashboard.py                    # Start on port 8080
  python scripts/dashboard.py --port 3000        # Custom port
  python scripts/dashboard.py --refresh 5        # 5s auto-refresh
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Abundance Dashboard")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Abundance — Live Trading Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'SF Mono', 'Menlo', 'Consolas', monospace; background: #0a0a0f; color: #e0e0e0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; background: #12121a; border-radius: 8px; margin-bottom: 20px; }
        .header h1 { font-size: 20px; color: #7cff6b; }
        .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-dot.ok { background: #7cff6b; box-shadow: 0 0 8px #7cff6b; }
        .status-dot.warn { background: #ffaa00; box-shadow: 0 0 8px #ffaa00; }
        .status-dot.crit { background: #ff4444; box-shadow: 0 0 8px #ff4444; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; }
        .card { background: #12121a; border-radius: 8px; padding: 16px; border: 1px solid #1e1e2e; }
        .card h2 { font-size: 13px; text-transform: uppercase; letter-spacing: 1px; color: #666; margin-bottom: 12px; }
        .metric { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a25; }
        .metric:last-child { border-bottom: none; }
        .metric .label { color: #888; }
        .metric .value { font-weight: bold; }
        .positive { color: #7cff6b; }
        .negative { color: #ff4444; }
        .neutral { color: #e0e0e0; }
        .trade-row { display: flex; justify-content: space-between; padding: 4px 0; font-size: 12px; border-bottom: 1px solid #1a1a25; }
        .trade-row:last-child { border-bottom: none; }
        .tag { padding: 2px 6px; border-radius: 3px; font-size: 10px; font-weight: bold; }
        .tag-buy { background: #1a3a1a; color: #7cff6b; }
        .tag-sell { background: #3a1a1a; color: #ff4444; }
        .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; }
        .badge-ok { background: #1a3a1a; color: #7cff6b; }
        .badge-warn { background: #3a2a10; color: #ffaa00; }
        .badge-crit { background: #3a1a1a; color: #ff4444; }
        #refresh-timer { font-size: 11px; color: #555; }
        .anomaly { background: #2a1a1a; padding: 6px 10px; border-radius: 4px; margin: 4px 0; font-size: 12px; color: #ff8888; }
    </style>
</head>
<body>
    <div class="header">
        <div>
            <span class="status-dot ok" id="status-dot"></span>
            <h1 style="display:inline">ABUNDANCE LIVE</h1>
        </div>
        <div style="text-align:right">
            <div id="refresh-timer">Next refresh: --s</div>
            <div style="font-size: 11px; color: #555" id="last-update">--</div>
        </div>
    </div>

    <div class="grid">
        <!-- Prices -->
        <div class="card">
            <h2>📊 Live Prices</h2>
            <div id="prices">Loading...</div>
        </div>

        <!-- PnL & Equity -->
        <div class="card">
            <h2>💰 Portfolio</h2>
            <div id="portfolio">Loading...</div>
        </div>

        <!-- Positions -->
        <div class="card">
            <h2>📈 Positions</h2>
            <div id="positions">Loading...</div>
        </div>

        <!-- Risk -->
        <div class="card">
            <h2>🛡️ Risk Status</h2>
            <div id="risk">Loading...</div>
        </div>

        <!-- Recent Trades -->
        <div class="card">
            <h2>📝 Recent Trades</h2>
            <div id="trades">Loading...</div>
        </div>

        <!-- Anomalies -->
        <div class="card">
            <h2>🚨 Alerts</h2>
            <div id="alerts">No alerts</div>
        </div>

        <!-- Agent Status -->
        <div class="card">
            <h2>🤖 Agent Pipeline</h2>
            <div id="agents">Loading...</div>
        </div>

        <!-- System Health -->
        <div class="card">
            <h2>🔧 System</h2>
            <div id="system">Loading...</div>
        </div>
    </div>

    <script>
        const REFRESH = {{ refresh_sec }};
        let countdown = REFRESH;

        function showReport(node) {
            const reports = window._reports || {};
            const r = reports[node];
            const div = document.getElementById('agent-report');
            if (r && r.content) {
                div.style.display = 'block';
                var txt = (r.content || '').split(String.fromCharCode(10)).join('<br>');
                div.innerHTML = '<strong>' + (r.title || node) + '</strong><br><br>' + txt + '<br><br><em>' + (r.updated || '') + '</em>';
            }
        }
        async function fetchData() {
            try {
                const resp = await fetch('/api/status');
                const data = await resp.json();
                render(data);
                document.getElementById('status-dot').className = 'status-dot ok';
            } catch(e) {
                document.getElementById('status-dot').className = 'status-dot crit';
            }
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            countdown = REFRESH;
        }

        function render(data) {
            // Prices
            let pricesHtml = '';
            for (const [pair, price] of Object.entries(data.prices || {})) {
                pricesHtml += `<div class="metric"><span class="label">${pair}</span><span class="value neutral">\$${price.toLocaleString()}</span></div>`;
            }
            if (data.funding) {
                for (const [pair, rate] of Object.entries(data.funding)) {
                    const cls = rate > 0 ? 'positive' : 'negative';
                    pricesHtml += `<div class="metric"><span class="label">${pair} funding</span><span class="value ${cls}">${rate.toFixed(4)}%</span></div>`;
                }
            }
            document.getElementById('prices').innerHTML = pricesHtml || 'No data';

            // Portfolio
            let portHtml = '';
            if (data.pnl) {
                const cls = data.pnl.total_pnl >= 0 ? 'positive' : 'negative';
                portHtml += `<div class="metric"><span class="label">Equity</span><span class="value neutral">\$${data.pnl.equity?.toLocaleString() || '0'}</span></div>`;
                portHtml += `<div class="metric"><span class="label">Unrealized PnL</span><span class="value ${cls}">\$${data.pnl.total_upnl?.toLocaleString() || '0'}</span></div>`;
                portHtml += `<div class="metric"><span class="label">Drawdown</span><span class="value negative">${data.pnl.drawdown_pct?.toFixed(2) || '0'}%</span></div>`;
            }
            document.getElementById('portfolio').innerHTML = portHtml || 'No positions';

            // Positions
            let posHtml = '';
            if (data.positions && data.positions.length > 0) {
                for (const p of data.positions) {
                    const cls = parseFloat(p.unRealizedProfit) >= 0 ? 'positive' : 'negative';
                    posHtml += `<div class="metric"><span class="label">${p.symbol}</span><span class="value">${p.positionAmt} @ \$${parseFloat(p.entryPrice).toLocaleString()} <span class="${cls}">\$${parseFloat(p.unRealizedProfit).toFixed(2)}</span></span></div>`;
                }
            } else {
                posHtml = '<div class="metric"><span class="label">No open positions</span></div>';
            }
            document.getElementById('positions').innerHTML = posHtml;

            // Risk
            let riskHtml = '';
            if (data.risk) {
                riskHtml += `<div class="metric"><span class="label">Drawdown</span><span class="value negative">${data.risk.drawdown_pct}%</span></div>`;
                riskHtml += `<div class="metric"><span class="label">Daily PnL</span><span class="value ${data.risk.daily_pnl_pct >= 0 ? 'positive' : 'negative'}">${data.risk.daily_pnl_pct >= 0 ? '+' : ''}${data.risk.daily_pnl_pct}%</span></div>`;
                const halted = data.risk.halted;
                riskHtml += `<div class="metric"><span class="label">Circuit Breaker</span><span class="badge ${halted ? 'badge-crit' : 'badge-ok'}">${halted ? 'HALTED' : 'ARMED'}</span></div>`;
                if (halted) riskHtml += `<div class="anomaly">${data.risk.halt_reason}</div>`;
            }
            document.getElementById('risk').innerHTML = riskHtml || 'No risk data';

            // Trades
            let tradeHtml = '';
            if (data.trades && data.trades.length > 0) {
                for (const t of data.trades.slice(0, 10)) {
                    const tag = t.side === 'BUY' ? 'tag-buy' : 'tag-sell';
                    const dt = new Date(t.ts).toLocaleString();
                    tradeHtml += `<div class="trade-row"><span>${dt}</span><span class="tag ${tag}">${t.side}</span><span>${parseFloat(t.qty).toFixed(4)} ${t.pair}</span><span>\$${parseFloat(t.price).toLocaleString()}</span></div>`;
                }
            } else {
                tradeHtml = '<div class="trade-row">No trades yet</div>';
            }
            document.getElementById('trades').innerHTML = tradeHtml;

            // Alerts
            let alertHtml = '';
            if (data.anomalies && data.anomalies.length > 0) {
                for (const a of data.anomalies.slice(-5)) {
                    alertHtml += `<div class="anomaly">🚨 ${a}</div>`;
                }
            } else {
                alertHtml = '<div style="color:#666">No active alerts</div>';
            }
            document.getElementById('alerts').innerHTML = alertHtml;

            // Agents
            let agentHtml = '';
            if (data.agent_status) {
                const reports = data.workflow_reports || {};
                window._reports = reports;
                for (const [node, status] of Object.entries(data.agent_status)) {
                    const icon = {research:'🔍',hypothesis:'💡',coding:'💻',backtest:'📊',adversarial:'🛡️',decision:'⚖️',paper_trade:'📈'}[node] || '✅';
                    const r = reports[node];
                    const hasReport = r && r.title;
                    const clickHandler = hasReport ? ` onclick="showReport('${node}')"` : '';
                    agentHtml += `<div class="metric" style="cursor:${hasReport?'pointer':'default'}"${clickHandler}><span class="label">${icon} ${node}${hasReport?' 📄':''}</span><span class="badge badge-${status === 'active' ? 'ok' : status === 'error' ? 'crit' : 'warn'}">${status.toUpperCase()}</span></div>`;
                }
            }
            document.getElementById('agents').innerHTML = agentHtml || 'No agent data';

            // System
            let sysHtml = '';
            if (data.system) {
                sysHtml += `<div class="metric"><span class="label">Testnet</span><span class="badge ${data.system.connected ? 'badge-ok' : 'badge-crit'}">${data.system.connected ? 'ONLINE' : 'OFFLINE'}</span></div>`;
                sysHtml += `<div class="metric"><span class="label">Pairs loaded</span><span class="value">${data.system.pairs_loaded || 0}</span></div>`;
                sysHtml += `<div class="metric"><span class="label">Trades logged</span><span class="value">${data.system.total_trades || 0}</span></div>`;
                sysHtml += `<div class="metric"><span class="label">Anomalies</span><span class="value">${data.system.anomalies || 0}</span></div>`;
            }
            document.getElementById('system').innerHTML = sysHtml || 'No system data';
        }

        // Countdown timer
        setInterval(() => {
            countdown--;
            if (countdown <= 0) { fetchData(); countdown = REFRESH; }
            document.getElementById('refresh-timer').textContent = `Next refresh: ${countdown}s`;
        }, 1000);

        fetchData();
    </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def dashboard(refresh: int = 1):
    return HTML_TEMPLATE.replace("{{ refresh_sec }}", str(refresh))


@app.get("/api/status")
async def api_status():
    """Aggregate all live data for the dashboard."""
    data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prices": {},
        "funding": {},
        "pnl": {},
        "positions": [],
        "risk": {},
        "trades": [],
        "anomalies": [],
        "agent_status": {
            "research": "ready",
            "hypothesis": "ready",
            "coding": "ready",
            "backtest": "ready",
            "adversarial": "ready",
            "decision": "ready",
            "paper_trade": "ready",
        },
        "system": {"connected": False, "pairs_loaded": 0, "total_trades": 0, "anomalies": 0},
        "workflow_reports": {},
    }

    # Load workflow reports for clickable agent cards
    try:
        ws_path = Path("data/processed/workflow_status.json")
        if ws_path.exists():
            data["workflow_reports"] = json.loads(ws_path.read_text())
    except Exception:
        pass

    # Prices + funding
    try:
        from abundance.paper_trading.testnet_client import get_testnet_client

        client = get_testnet_client()
        data["system"]["connected"] = True

        for pair in ["BTCUSDT", "ETHUSDT", "SOLUSDT"]:
            try:
                data["prices"][pair] = client.get_price(pair)
                data["funding"][pair] = client.get_funding_rate(pair) * 100
            except Exception:
                pass

        # Positions
        try:
            positions = client.get_positions()
            data["positions"] = positions
            total_upnl = sum(float(p.get("unRealizedProfit", 0)) for p in positions)
            # Get actual balance
            try:
                bal = client.get_balance()
                equity = bal.get("USDT", 5000) + total_upnl
            except Exception:
                equity = 5000.0
            data["pnl"] = {"equity": round(equity, 2), "total_upnl": round(total_upnl, 2), "drawdown_pct": 0.0}
        except Exception:
            pass
    except Exception:
        pass

    # Trade log
    try:
        from abundance.deployment.monitoring import TradeLogger

        tl = TradeLogger()
        data["system"]["total_trades"] = tl.get_trade_count()
        data["trades"] = tl.get_recent_trades(10)
    except Exception:
        pass

    # Monitor state
    try:
        state_path = Path("data/processed/monitor_state.json")
        if state_path.exists():
            state = json.loads(state_path.read_text())
            data["system"]["anomalies"] = state.get("anomaly_count", 0)
    except Exception:
        pass

    # Risk
    try:
        data["risk"] = {
            "drawdown_pct": 0.0,
            "daily_pnl_pct": 0.0,
            "halted": False,
            "halt_reason": "",
        }
    except Exception:
        pass

    # Pairs
    try:
        from abundance.config.settings import settings

        klines_dir = settings.raw_dir / "klines"
        data["system"]["pairs_loaded"] = len([
            d for d in klines_dir.iterdir() if d.is_dir() and d.name.endswith("_1d")
        ])
    except Exception:
        pass

    return JSONResponse(content=data)


def main():
    parser = argparse.ArgumentParser(description="Abundance Live Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--refresh", type=int, default=10, help="Auto-refresh interval (seconds)")
    args = parser.parse_args()

    import uvicorn

    print(f"\n📊 Abundance Dashboard → http://localhost:{args.port}")
    print(f"   Auto-refresh: {args.refresh}s")
    print(f"   Press Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
