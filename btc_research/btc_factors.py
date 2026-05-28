# btc_factors.py — Factor builder untuk BTC-only research (single asset).
#
# Berdiri sendiri dari factors.py — hanya import helper `zscore`. Output: satu
# DataFrame ber-index waktu dengan kolom factor + forward_ret_{fp} multi-horizon
# + tag `regime` (bull/bear via 50-MA).

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from factors import zscore


FORWARD_PERIODS = [3, 6, 12, 24]   # candle 4H → 12h, 24h, 48h, 4d

WVWAP_WINDOW    = 42               # weekly = 7 hari × 6 candle
ZW              = config.TS_ZSCORE_WINDOW   # 120


def _streak(returns: pd.Series) -> pd.Series:
    sgn   = np.sign(returns).fillna(0)
    grp   = (sgn != sgn.shift()).cumsum()
    count = sgn.groupby(grp).cumcount() + 1
    return count * sgn


def build_btc_factors(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute BTC-only factors + multi-horizon forward returns + regime tag."""
    df = btc_df.copy()
    c, h, l, v = df['close'], df['high'], df['low'], df['volume']
    ret = c.pct_change()

    out = pd.DataFrame(index=df.index)

    # ── Momentum group ────────────────────────────────────────
    out['mom_12h'] = zscore(c.pct_change(3),  window=ZW)
    out['mom_1d']  = zscore(c.pct_change(6),  window=ZW)
    out['mom_3d']  = zscore(c.pct_change(18), window=ZW)
    out['mom_7d']  = zscore(c.pct_change(42), window=ZW)

    # ── Mean-reversion group ─────────────────────────────────
    streak_sgn = _streak(ret)
    rng        = (h.rolling(18).max() - l.rolling(18).min()).replace(0, np.nan)
    stoch      = (c - l.rolling(18).min()) / rng
    clv_raw    = ((c - l) - (h - c)) / ((h - l).replace(0, np.nan))

    out['streak_rev']  = zscore(-streak_sgn,        window=ZW)
    out['stoch_rev']   = zscore(-(stoch - 0.5),     window=ZW)
    out['clv']         = zscore(clv_raw.rolling(3).mean(), window=ZW)
    out['reversal_1d'] = zscore(-ret,               window=ZW)

    # ── NEW: price vs weekly (42-candle) VWAP ────────────────
    # True volume-weighted, bukan HLC/3 proxy.
    pv             = c * v
    wvwap          = pv.rolling(WVWAP_WINDOW).sum() / v.rolling(WVWAP_WINDOW).sum()
    pv_signal      = -(c - wvwap) / wvwap     # negatif kalau harga di atas VWAP
    out['price_vs_wvwap'] = zscore(pv_signal, window=ZW)

    # ── Funding group ────────────────────────────────────────
    if 'funding_rate' in df.columns:
        fr = df['funding_rate']
        out['funding_signal']  = zscore(-fr,                  window=ZW)
        out['funding_mom']     = zscore(-fr.diff(2),          window=ZW)
        out['funding_extreme'] = zscore(fr.abs(),             window=ZW)
    else:
        out['funding_signal']  = np.nan
        out['funding_mom']     = np.nan
        out['funding_extreme'] = np.nan

    # ── Forward returns (multi-horizon) ──────────────────────
    for fp in FORWARD_PERIODS:
        out[f'forward_ret_{fp}'] = c.shift(-fp) / c - 1

    # ── Regime tag (bull/bear via 50-MA) ─────────────────────
    ma = c.rolling(config.BTC_REGIME_MA, min_periods=10).mean()
    out['regime'] = np.where(c > ma, 'bull', 'bear')

    return out


FACTOR_COLS = [
    'mom_12h', 'mom_1d', 'mom_3d', 'mom_7d',
    'streak_rev', 'stoch_rev', 'clv', 'reversal_1d', 'price_vs_wvwap',
    'funding_signal', 'funding_mom', 'funding_extreme',
]


FACTOR_GROUP = {
    'mom_12h': 'momentum', 'mom_1d': 'momentum', 'mom_3d': 'momentum', 'mom_7d': 'momentum',
    'streak_rev': 'mean-rev', 'stoch_rev': 'mean-rev', 'clv': 'mean-rev',
    'reversal_1d': 'mean-rev', 'price_vs_wvwap': 'mean-rev',
    'funding_signal': 'funding', 'funding_mom': 'funding', 'funding_extreme': 'funding',
}
