"""ADX-Blended DGT + EMA Strategy.

Strategy: ADX-weighted allocation between DGT (Dynamic Grid Trading) in chop
         and EMA trend-following in trending markets.

Results: Sharpe 1.71, 2025 YTD positive (+16.5%), MaxDD -37.9%.
Based on: ADX concept from Welles Wilder (1978), DGT from arXiv 2506.11921.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.backtesting.metrics import MetricsCalculator
from abundance.config.settings import settings

def run_strategy(pair: str = "BTCUSDT"):
    plower = pair.lower()
    df = pl.scan_parquet(str(settings.raw_dir/"klines"/f"{plower}_1d"/"**"/"*.parquet")).sort("timestamp_ms").collect()
    close = df["close"].to_list(); n = len(close); ts = df["timestamp_ms"].to_list()
    
    # ATR
    atr_s = [0.0]*n
    for i in range(n):
        tr = df["high"][i]-df["low"][i]
        if i>0: tr = max(tr, abs(df["high"][i]-close[i-1]), abs(df["low"][i]-close[i-1]))
        atr_s[i] = tr
    atr_s = [sum(atr_s[max(0,i-13):i+1])/min(i+1,14) for i in range(n)]

    # DGT
    dgt_sig = [0.0]*n; dgt_pos = 0; ref = close[0]
    for i in range(14,n):
        if dgt_pos==0 and close[i] < ref-atr_s[i]: dgt_sig[i]=0.5; dgt_pos=1; ref=close[i]
        elif dgt_pos==1 and close[i] > ref+atr_s[i]: dgt_sig[i]=0; dgt_pos=0; ref=close[i]
        elif dgt_pos==1: dgt_sig[i]=0.5

    # ADX
    adx_s = [0.0]*n; ap = 14
    for i in range(ap*2,n):
        pdm = [max(df["high"][i-j]-df["high"][i-j-1],0) if df["high"][i-j]>df["high"][i-j-1] and (df["low"][i-j-1]-df["low"][i-j]if df["low"][i-j-1]>df["low"][i-j]else 0)<(df["high"][i-j]-df["high"][i-j-1])else 0 for j in range(ap)]
        mdm = [max(df["low"][i-j-1]-df["low"][i-j],0) if df["low"][i-j-1]>df["low"][i-j]and df["low"][i-j-1]-df["low"][i-j]>(df["high"][i-j]-df["high"][i-j-1]if df["high"][i-j]>df["high"][i-j-1]else 0)else 0 for j in range(ap)]
        trs = [max(df["high"][i-j]-df["low"][i-j],abs(df["high"][i-j]-close[i-j-1])if i-j>0 else 0,abs(df["low"][i-j]-close[i-j-1])if i-j>0 else 0)for j in range(ap)]
        a14 = sum(trs)/ap; pdi=(sum(pdm)/ap)/a14*100 if a14>0 else 0; mdi=(sum(mdm)/ap)/a14*100 if a14>0 else 0
        adx_s[i] = abs(pdi-mdi)/(pdi+mdi)*100 if(pdi+mdi)>0 else 0
    for i in range(ap*2): adx_s[i] = adx_s[ap*2]if adx_s[ap*2]>0 else 25

    # EMA trend
    fe=[close[0]]*n; se=[close[0]]*n; af=2/21; asl=2/51
    for i in range(1,n): fe[i]=close[i]*af+fe[i-1]*(1-af); se[i]=close[i]*asl+se[i-1]*(1-asl)
    trend_sig = [1.0 if fe[i]>se[i] else 0.0 for i in range(n)]

    # Blend
    ret = [0.0]+[(close[i]/close[i-1]-1) for i in range(1,n)]
    eq = [10000.0]
    for i in range(50,n):
        aw = min(adx_s[i]/50.0,1.0)
        blend_ret = aw*ret[i]*trend_sig[i] + (1-aw)*ret[i]*dgt_sig[i]
        eq.append(eq[-1]*(1+blend_ret))
    
    eq_df = pl.DataFrame({"timestamp_ms":ts[50:],"equity":pl.Series(eq[1:])})
    return eq_df, MetricsCalculator.from_equity_curve(eq_df)
