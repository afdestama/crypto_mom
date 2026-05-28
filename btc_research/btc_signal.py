# btc_signal.py — Reusable BTC composite signal untuk alt ranking gate.
#
# Walk-forward weights (refit dari N hari terakhir), evaluasi composite di
# candle terbaru, klasifikasi BULL/NEUTRAL/BEAR berdasarkan threshold.
# Untuk testing tanpa menunggu kondisi BTC nyata: set env BTC_FORCE_MODE
# ke BULL/NEUTRAL/BEAR.

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_research.btc_data        import fetch_btc_research_data
from btc_research.btc_factors     import build_btc_factors, FACTOR_COLS
from btc_research.btc_backtest_wf import composite_on


CANDLES_PER_DAY = 6
DEFAULT_FPS     = [3, 6, 12, 24]


def select_factors_multi_fp(fdf_train: pd.DataFrame,
                            fps: list = None,
                            threshold: float = 0.05,
                            min_obs: int = 50) -> dict:
    """Per factor, pilih FP dengan |IC train| terkuat. Include kalau ≥ threshold.

    Returns: {factor: {'sign': ±1, 'best_fp': fp, 'best_ic': ic}}.
    Catatan: testing multi-FP per factor → multiple-comparison optimistic.
    """
    fps = fps or DEFAULT_FPS
    chosen = {}
    for f in FACTOR_COLS:
        if f not in fdf_train.columns:
            continue
        x = fdf_train[f]
        best_ic, best_fp = 0.0, None
        for fp in fps:
            tgt_col = f'forward_ret_{fp}'
            if tgt_col not in fdf_train.columns:
                continue
            target = fdf_train[tgt_col]
            mask   = x.notna() & target.notna()
            if mask.sum() < min_obs:
                continue
            ic, _ = spearmanr(x[mask], target[mask])
            if np.isnan(ic):
                continue
            if abs(ic) > abs(best_ic):
                best_ic, best_fp = float(ic), fp
        if best_fp is not None and abs(best_ic) >= threshold:
            chosen[f] = {
                'sign'   : +1.0 if best_ic > 0 else -1.0,
                'best_fp': best_fp,
                'best_ic': best_ic,
            }
    return chosen


def _classify(composite: float, long_thr: float, short_thr: float) -> str:
    if composite > long_thr:
        return 'BULL'
    if composite < short_thr:
        return 'BEAR'
    return 'NEUTRAL'


def _forced_signal(mode: str, long_thr: float, short_thr: float) -> dict:
    composite = {'BULL': long_thr + 0.5,
                 'BEAR': short_thr - 0.5,
                 'NEUTRAL': 0.0}.get(mode.upper(), 0.0)
    return {
        'composite'    : composite,
        'mode'         : mode.upper(),
        'signs'        : {},
        'factors_meta' : {},
        'last_dt'      : None,
        'btc_close'    : None,
        'forced'       : True,
    }


def compute_live_btc_signal(train_days: int = 45,
                            fps: list = None,
                            ic_threshold: float = 0.05,
                            long_thr: float = 1.0,
                            short_thr: float = -1.0,
                            force_refresh: bool = False) -> dict:
    """Hitung BTC directional signal di candle terbaru, multi-FP per-factor.

    Per factor, FP terbaik dipilih dari sweep `fps` (default {3,6,12,24}) — yang
    punya |IC train| terkuat. Hanya factor dengan |best IC| ≥ ic_threshold yang
    masuk composite. Returns dict berisi:
      • composite     — nilai composite di candle terbaru
      • mode          — 'BULL'/'NEUTRAL'/'BEAR' berdasarkan threshold
      • signs         — {factor: ±1} (untuk composite_on)
      • factors_meta  — {factor: {sign, best_fp, best_ic}} (untuk display)
      • last_dt, btc_close, forced
    """
    fps = fps or DEFAULT_FPS

    forced = os.environ.get('BTC_FORCE_MODE')
    if forced:
        return _forced_signal(forced, long_thr, short_thr)

    # fetch_btc_research_data sudah drop candle ongoing di sisi fetcher,
    # jadi btc_df di sini dijamin hanya berisi candle complete.
    btc_df = fetch_btc_research_data(force_refresh=force_refresh)
    fdf    = build_btc_factors(btc_df)

    train_n   = train_days * CANDLES_PER_DAY
    drop_tail = max(fps)   # buang tail sebanyak FP terbesar untuk label validity
    if len(fdf) < train_n + drop_tail + 1:
        raise RuntimeError(
            f"Data BTC tidak cukup: butuh ≥{train_n + drop_tail + 1} candle, "
            f"punya {len(fdf)}. Coba force_refresh=True."
        )

    train_slice  = fdf.iloc[-(train_n + drop_tail):-drop_tail]
    factors_meta = select_factors_multi_fp(train_slice, fps=fps, threshold=ic_threshold)

    if not factors_meta:
        return {
            'composite'    : 0.0,
            'mode'         : 'NEUTRAL',
            'signs'        : {},
            'factors_meta' : {},
            'last_dt'      : fdf.index[-1],
            'btc_close'    : float(btc_df['close'].iloc[-1]),
            'forced'       : False,
        }

    signs = {f: meta['sign'] for f, meta in factors_meta.items()}

    comp_series = composite_on(fdf, signs).dropna()
    if comp_series.empty:
        composite = 0.0
        last_dt   = fdf.index[-1]
    else:
        composite = float(comp_series.iloc[-1])
        last_dt   = comp_series.index[-1]

    return {
        'composite'    : composite,
        'mode'         : _classify(composite, long_thr, short_thr),
        'signs'        : signs,
        'factors_meta' : factors_meta,
        'last_dt'      : last_dt,
        'btc_close'    : float(btc_df['close'].loc[last_dt]) if last_dt in btc_df.index
                          else float(btc_df['close'].iloc[-1]),
        'forced'       : False,
    }


if __name__ == '__main__':
    sig = compute_live_btc_signal()
    print(f"composite : {sig['composite']:+.3f}")
    print(f"mode      : {sig['mode']}")
    print(f"factors   : {len(sig['signs'])}")
    for f, meta in sig['factors_meta'].items():
        sign_c = '+' if meta['sign'] > 0 else '-'
        print(f"  {f:<18} sign={sign_c}  best_fp={meta['best_fp']:>2}  ic={meta['best_ic']:+.3f}")
    print(f"last_dt   : {sig['last_dt']}")
    print(f"btc_close : ${sig['btc_close']:,.1f}" if sig['btc_close'] else "btc_close : N/A")
