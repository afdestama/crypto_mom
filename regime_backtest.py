# regime_backtest.py — Uji composite regime-conditional vs full-sample.
#
# 3 cara membangun composite_score, lalu dijalankan di mesin bracket yang sama
# (backtest_bracket.run, config A: ±2ATR RR1:1 + time-stop, net fee):
#
#   1. BASELINE   : full-sample, positive-ICIR only  (= yang rugi −58%)
#   2. CONTROL    : full-sample, bobot = signed ICIR  (isolasi efek signed)
#   3. REGIME-COND: bobot signed ICIR dihitung TERPISAH per regime bull/bear,
#                   dipakai sesuai btc_regime tiap candle
#
# Tujuan: pisahkan kontribusi "signed weighting" vs "regime switching".
# CAVEAT sama spt backtest_bracket: bobot in-sample (look-ahead) di KETIGA
# varian → perbandingan apple-to-apple, tapi level absolut optimistis.

import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import config
from factors      import build_panel, FACTOR_COLS
from ic_analysis  import cross_sectional_ic, build_composite_score
from diag_regime_ic import regime_ic
from backtest_bracket import (load_cache, add_atr, run, stats,
                              HOLD, REBAL_STEP, WARMUP)


def signed_weights(icir_summary):
    """factor -> signed weight, dinormalisasi oleh sum |ICIR|. Skip NaN."""
    raw = {c: v[1] for c, v in icir_summary.items()
           if np.isfinite(v[1])}
    denom = sum(abs(w) for w in raw.values()) + 1e-9
    return {c: w / denom for c, w in raw.items()}


def apply_weights(panel, factors, w):
    s = pd.Series(0.0, index=panel.index)
    for c in factors:
        if c in w and c in panel.columns:
            s += w[c] * panel[c].fillna(0)
    return s


def main():
    raw, btc, universe = load_cache()
    panel = build_panel(raw, btc_df=btc)
    avail = [c for c in FACTOR_COLS if c in panel.columns
             and panel[c].notna().sum() > config.IC_ROLLING_WINDOW * 5]

    # ── 1. BASELINE: pipeline asli (full-sample positive-only) ──
    ic_summary, _ = cross_sectional_ic(panel, avail, forward_col='forward_1d')
    panel_base, _ = build_composite_score(panel, ic_summary)
    panel_base = panel_base.rename(columns={'composite_score': 'score_base'})

    # ── 2. CONTROL: full-sample signed ICIR ──
    full = {r['factor']: (r['ic_mean'], r['icir'], r['n_obs'])
            for _, r in ic_summary.iterrows()}
    w_full = signed_weights(full)

    # ── 3. REGIME-COND: signed ICIR per regime ──
    bull, bear = regime_ic(panel, avail)
    w_bull, w_bear = signed_weights(bull), signed_weights(bear)

    print("\nBobot signed-ICIR (top 5 |w| tiap kondisi):")
    for nm, w in [('FULL', w_full), ('BULL', w_bull), ('BEAR', w_bear)]:
        top = sorted(w.items(), key=lambda kv: -abs(kv[1]))[:5]
        print(f"  {nm:<5}: " + "  ".join(f"{k}={v:+.2f}" for k, v in top))

    p = panel.copy()
    p['score_base']   = panel_base['score_base'].values
    p['score_ctrl']   = apply_weights(p, avail, w_full)
    bull_s = apply_weights(p, avail, w_bull)
    bear_s = apply_weights(p, avail, w_bear)
    p['score_regime'] = np.where(p['btc_regime'] >= 0.5, bull_s, bear_s)

    # ── Backtest ketiganya (config A) ──
    atr_map = {s: add_atr(df) for s, df in raw.items()}
    times = sorted(p['datetime'].unique())
    rebal = times[WARMUP: len(times) - HOLD][::REBAL_STEP]

    print(f"\n{'='*64}")
    print(f"  REGIME-CONDITIONAL TEST — config A (±2ATR RR1:1 + time-stop, net fee)")
    print(f"  {len(rebal)} cycles, top-{config.TOP_N_COINS}, hold {HOLD} candle")
    print('='*64)

    for label, col in [("1. BASELINE  (full-sample, positive-only)", 'score_base'),
                       ("2. CONTROL   (full-sample, signed ICIR)",    'score_ctrl'),
                       ("3. REGIME    (per-regime signed ICIR)",      'score_regime')]:
        pp = p.rename(columns={col: 'composite_score'})
        _, tdf, _, eq = run(label, pp, raw, atr_map, universe, rebal,
                            tp_mult=2.0, sl_mult=2.0, apply_fee=True)
        print("\n" + stats(label, tdf, eq, len(rebal)))

    seg = btc[(btc.index >= pd.Timestamp(rebal[0])) &
              (btc.index <= pd.Timestamp(rebal[-1]))]
    bh = seg['close'].iloc[-1] / seg['close'].iloc[0] - 1
    print(f"\n{'-'*64}\n  Benchmark BTC buy&hold: {bh:+.1%}\n{'='*64}")


if __name__ == "__main__":
    main()
