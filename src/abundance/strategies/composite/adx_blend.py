"""ADX-Blended DGT + EMA Strategy.

ADX-weighted allocation between DGT (grid) in chop and EMA trend-following.
All signals use data up to t-1. Implements Strategy ABC.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import polars as pl
from abundance.strategies.base import Strategy, StrategyArtifacts


class ADXBlendStrategy(Strategy):
    """ADX-weighted blend of DGT grid + EMA trend."""

    def __init__(self, adx_period: int = 14, fast_ema: int = 21, slow_ema: int = 51):
        self.adx_period = adx_period
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema

    def signals(self, df: pl.DataFrame) -> list[float]:
        close = df["close"].to_list()
        high = df["high"].to_list()
        low = df["low"].to_list()
        N = len(close)

        # --- ATR(14) ---
        tr = [0.0] * N
        for i in range(1, N):
            tr[i] = max(high[i] - low[i],
                        abs(high[i] - close[i-1]),
                        abs(low[i] - close[i-1]))
        atr = [0.0] * N
        ap = self.adx_period
        if N > ap:
            atr[ap-1] = sum(tr[1:ap]) / (ap - 1)
            for i in range(ap, N):
                atr[i] = (atr[i-1] * (ap - 1) + tr[i]) / ap

        # --- DGT signal (uses data up to t-1) ---
        dgt_sig = [0.0] * N
        dgt_pos = 0
        ref = close[0]
        for i in range(ap, N):
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

        # --- ADX (uses data up to t-1) ---
        adx = [25.0] * N
        for i in range(ap * 2, N):
            j = i - 1  # yesterday's data
            pdm_sum = 0.0; mdm_sum = 0.0; tr_sum = 0.0
            for k in range(ap):
                h_cur = high[j-k]; h_prev = high[j-k-1]
                l_cur = low[j-k]; l_prev = low[j-k-1]
                c_prev = close[j-k-1]
                up_move = h_cur - h_prev
                dn_move = l_prev - l_cur
                pdm_sum += up_move if (up_move > 0 and up_move > dn_move) else 0.0
                mdm_sum += dn_move if (dn_move > 0 and dn_move > up_move) else 0.0
                tr_sum += max(h_cur - l_cur, abs(h_cur - c_prev), abs(l_cur - c_prev))
            a14 = tr_sum / ap if ap > 0 else 0
            pdi = (pdm_sum/ap) / a14 * 100 if a14 > 0 else 0
            mdi = (mdm_sum/ap) / a14 * 100 if a14 > 0 else 0
            adx[i] = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0

        # --- EMA trend (signal at t uses data up to t-1) ---
        fe = [close[0]] * N  # fast EMA
        se = [close[0]] * N  # slow EMA
        af = 2 / (self.fast_ema + 1)
        asl = 2 / (self.slow_ema + 1)
        for i in range(1, N):
            fe[i] = fe[i-1] + af * (close[i] - fe[i-1])
            se[i] = se[i-1] + asl * (close[i] - se[i-1])

        trend_sig = [0.0] * N
        start = max(52, ap * 2)
        for i in range(start, N):
            trend_sig[i] = 1.0 if fe[i-1] > se[i-1] else 0.0

        # --- Blend ---
        sig = [0.0] * N
        for i in range(start, N):
            aw = min(adx[i] / 50.0, 1.0)
            sig[i] = aw * trend_sig[i] + (1 - aw) * dgt_sig[i]

        return sig

    def _get_params(self) -> dict:
        return {
            "adx_period": self.adx_period,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
        }

def run_strategy(pair: str = "BTCUSDT"):
    s = ADXBlendStrategy()
    art = s.run(pair)
    return art.equity_curve, art.metrics
