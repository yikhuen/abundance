"""ADX-Blended DGT + EMA Strategy.

Strategy: ADX-weighted allocation between DGT (Dynamic Grid Trading) in chop
         and EMA trend-following in trending markets.

All signals use data up to t-1: close[t-1], EMA[t-2], ADX[t-1], ATR[t-1].
No lookahead — decision made at end of day t-1, earns return of day t.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.backtesting.costs import COST_MODEL
from abundance.config.settings import settings

def run_strategy(pair: str = "BTCUSDT"):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); high = df["high"].to_list(); low = df["low"].to_list()
    ts = df["timestamp_ms"].to_list(); N = len(close)

    # --- ATR(14) ---
    tr = [0.0] * N
    for i in range(1, N):
        tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
    atr = [0.0] * N
    # seed atr after 14 bars
    if N > 14:
        atr[13] = sum(tr[1:14]) / 13
        for i in range(14, N):
            atr[i] = (atr[i-1]*13 + tr[i]) / 14

    # --- DGT signal (uses data up to t-1) ---
    dgt_sig = [0.0] * N
    dgt_pos = 0
    ref = close[0]
    for i in range(14, N):
        prev_close = close[i-1]
        prev_atr = atr[i-1] if i > 0 else atr[0]
        if prev_atr <= 0:
            dgt_sig[i] = dgt_sig[i-1]
            continue
        if dgt_pos == 0 and prev_close < ref - prev_atr:
            dgt_pos = 1
            ref = prev_close
        elif dgt_pos == 1 and prev_close > ref + prev_atr:
            dgt_pos = 0
            ref = prev_close
        dgt_sig[i] = 0.5 if dgt_pos == 1 else 0.0

    # --- ADX(14) using data up to t-1 ---
    ap = 14
    adx = [25.0] * N  # default to neutral
    for i in range(ap*2, N):
        j = i - 1  # use yesterday's data
        pdm_vals = []
        mdm_vals = []
        tr_vals = []
        for k in range(ap):
            h_cur = high[j-k]; h_prev = high[j-k-1]
            l_cur = low[j-k]; l_prev = low[j-k-1]
            c_prev = close[j-k-1]
            up_move = h_cur - h_prev
            dn_move = l_prev - l_cur
            if up_move > 0 and up_move > dn_move:
                pdm_vals.append(up_move)
            else:
                pdm_vals.append(0.0)
            if dn_move > 0 and dn_move > up_move:
                mdm_vals.append(dn_move)
            else:
                mdm_vals.append(0.0)
            tr_vals.append(max(h_cur-l_cur, abs(h_cur-c_prev), abs(l_cur-c_prev)))
        a14 = sum(tr_vals) / ap
        pdi = (sum(pdm_vals)/ap) / a14 * 100 if a14 > 0 else 0
        mdi = (sum(mdm_vals)/ap) / a14 * 100 if a14 > 0 else 0
        adx[i] = abs(pdi-mdi)/(pdi+mdi)*100 if (pdi+mdi) > 0 else 0

    # --- EMA trend (uses close[t-1] to compute EMA, then compare EMA[t-1] to EMA[t-2]) ---
    fe = [close[0]] * N  # fast EMA
    se = [close[0]] * N  # slow EMA
    af = 2/21; asl = 2/51
    for i in range(1, N):
        fe[i] = fe[i-1] + af * (close[i] - fe[i-1])
        se[i] = se[i-1] + asl * (close[i] - se[i-1])

    trend_sig = [0.0] * N
    for i in range(52, N):
        # Decision: at end of day i-1, was fast EMA above slow EMA?
        trend_sig[i] = 1.0 if fe[i-1] > se[i-1] else 0.0

    # --- Blend ---
    ret = [0.0] + [(close[i]/close[i-1]-1) for i in range(1,N)]
    cost_per_trade = COST_MODEL.round_trip_cost(pair, use_perp=True)
    eq = [10000.0]
    start = max(52, ap*2)
    trades = 0
    prev_sig = 0.0
    for i in range(start, N):
        aw = min(adx[i] / 50.0, 1.0)
        sig = aw * trend_sig[i] + (1-aw) * dgt_sig[i]
        strat_ret = ret[i] * sig
        txn_cost = abs(sig - prev_sig) * cost_per_trade
        eq.append(eq[-1] * (1 + strat_ret - txn_cost))
        prev_sig = sig
        if sig != dgt_sig[i-1] if i>0 else 0.0:
            pass  # DGT signal varies continuously, count threshold crosses
    # Count trades: discrete thresholding
    discrete_sig = [1.0 if sig > 0.25 else 0.0 for sig in
                    ([0.0]*start + [aw*trend_sig[i]+(1-aw)*dgt_sig[i] for i in range(start,N) for aw in [min(adx[i]/50.0,1.0)]])]
    # simpler: count from the loop
    trades_count = 0
    for i in range(start, N):
        aw = min(adx[i]/50.0, 1.0)
        s = aw*trend_sig[i] + (1-aw)*dgt_sig[i]
        if i > start:
            sp = min(adx[i-1]/50.0,1.0)*trend_sig[i-1] + (1-min(adx[i-1]/50.0,1.0))*dgt_sig[i-1]
            if (s > 0.25) != (sp > 0.25):
                trades_count += 1

    eq_df = pl.DataFrame({"timestamp_ms": ts[start:], "equity": eq[1:]})
    mc = MetricsCalculator.from_equity_curve(eq_df)
    mc.trades = trades_count
    return eq_df, mc
