# factor_research_wf.py — Walk-forward factor validation per regime
#
# Jalankan sekali (offline) untuk menentukan faktor mana yang robust.
# Output: rekomendasi TS_FACTOR_COLS per regime untuk di-paste ke factors.py
#
# Usage: python factor_research_wf.py

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import config
from factors import build_ts_factors, zscore

TRAIN_DAYS   = 60    # hari untuk initial training window
STEP_DAYS    = 15    # hari per step OOS
CANDLES      = 6     # candle per hari (@4H)
MIN_IC_OBS   = 10    # minimum IC observations per OOS window
MIN_COINS    = 5     # minimum coins per timestamp untuk hitung IC
MIN_ICIR     = 0.10  # median OOS ICIR threshold untuk "robust"
SIGN_CONS    = 0.60  # minimum fraction windows dengan IC sign konsisten

# Mapping: faktor → regime yang sesuai untuk diuji
FACTOR_REGIME_MAP = {
    'ts_momentum_follow' : 'trend',
    'ts_streak_rev'      : 'reversal',
    'ts_stoch_rev'       : 'reversal',
    'ts_clv'             : 'reversal',
    'ts_funding_contra'  : 'both',
    'ts_ls_contra'       : 'both',
    'ts_oi_buildup'      : 'both',
}


def load_cache_panel() -> tuple:
    """Load semua parquet dari cache, build panel + btc_df."""
    cache_dir = config.CACHE_DIR
    if not os.path.exists(cache_dir):
        raise FileNotFoundError(f"Cache tidak ditemukan: {cache_dir}. Jalankan main.py dulu.")

    frames, btc_df = [], pd.DataFrame()
    for fname in os.listdir(cache_dir):
        if not fname.endswith('_4h.parquet'):
            continue
        path = os.path.join(cache_dir, fname)
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty or 'close' not in df.columns:
            continue

        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else fname.replace('_4h.parquet', '')
        if 'BTC' in symbol and btc_df.empty:
            btc_df = df.copy()

        ts = build_ts_factors(df)
        fdf = pd.DataFrame(ts, index=df.index)
        fdf['symbol'] = symbol
        fdf['close']  = df['close']
        # forward return: cumulative FORWARD_PERIOD candles
        fdf['fwd'] = df['close'].shift(-config.FORWARD_PERIOD) / df['close'] - 1
        frames.append(fdf)

    if not frames:
        raise ValueError("Tidak ada data di cache.")

    panel = pd.concat(frames)
    if panel.index.name in ['datetime', 'date']:
        panel = panel.reset_index().rename(columns={panel.index.name: 'datetime'})
    elif 'datetime' not in panel.columns:
        panel = panel.reset_index().rename(columns={'index': 'datetime'})

    panel['datetime'] = pd.to_datetime(panel['datetime'])
    return panel.sort_values(['datetime', 'symbol']).reset_index(drop=True), btc_df


def compute_regime_map(btc_df: pd.DataFrame, trend_threshold: float = None) -> pd.Series:
    """
    Volatility-based regime per timestamp.
    Returns Series: index=datetime, value='trend'/'reversal'.
    """
    if trend_threshold is None:
        trend_threshold = config.REGIME_TREND_THRESHOLD

    w   = config.REGIME_TREND_WINDOW
    ret = btc_df['close'].pct_change()
    roc = btc_df['close'].pct_change(w).abs()
    vol = ret.rolling(w, min_periods=w // 2).std()
    strength = roc / (vol + 1e-9)

    regime = strength.apply(lambda x: 'trend' if x > trend_threshold else 'reversal')
    regime.index = pd.to_datetime(btc_df.index)
    return regime


def wf_icir_regime(panel: pd.DataFrame, factor_col: str,
                   regime_map: pd.Series, target_regime: str = 'both') -> pd.DataFrame:
    """
    Walk-forward OOS IC per expanding window, filtered ke target_regime.
    Returns DataFrame: window, regime, oos_ic, oos_icir, n_periods
    """
    all_dts = sorted(panel['datetime'].unique())
    train_c = TRAIN_DAYS * CANDLES
    step_c  = STEP_DAYS  * CANDLES
    results = []

    for i in range(train_c, len(all_dts) - step_c, step_c):
        oos_dts   = all_dts[i : i + step_c]
        oos_panel = panel[panel['datetime'].isin(oos_dts)].copy()

        if target_regime != 'both' and not regime_map.empty:
            regime_dts = set(regime_map[regime_map == target_regime].index)
            oos_panel  = oos_panel[oos_panel['datetime'].isin(regime_dts)]

        if oos_panel.empty:
            continue

        ic_vals = []
        for _, grp in oos_panel.groupby('datetime'):
            valid = grp[[factor_col, 'fwd']].dropna()
            if len(valid) >= MIN_COINS:
                ic = valid[factor_col].corr(valid['fwd'], method='spearman')
                if pd.notna(ic):
                    ic_vals.append(ic)

        if len(ic_vals) >= MIN_IC_OBS:
            arr = np.array(ic_vals)
            results.append({
                'window'   : i // step_c,
                'regime'   : target_regime,
                'oos_ic'   : round(arr.mean(), 5),
                'oos_icir' : round(arr.mean() / (arr.std() + 1e-9), 4),
                'n'        : len(arr),
            })

    return pd.DataFrame(results)


def is_robust(wf_df: pd.DataFrame) -> tuple:
    """Returns (bool, stats_dict)."""
    if wf_df.empty:
        return False, {'median_icir': np.nan, 'sign_cons': np.nan, 'n_windows': 0}

    median_icir   = wf_df['oos_icir'].median()
    dominant_sign = np.sign(wf_df['oos_ic'].median())
    sign_cons     = (np.sign(wf_df['oos_ic']) == dominant_sign).mean()

    robust = abs(median_icir) > MIN_ICIR and sign_cons >= SIGN_CONS
    return robust, {
        'median_icir' : round(median_icir, 4),
        'sign_cons'   : round(sign_cons, 3),
        'n_windows'   : len(wf_df),
        'sign'        : int(dominant_sign),
    }


def main():
    print("\n" + "━"*62)
    print("  WALK-FORWARD FACTOR VALIDATION (per regime)")
    print(f"  train={TRAIN_DAYS}d  step={STEP_DAYS}d  "
          f"min_ICIR={MIN_ICIR}  sign_cons={SIGN_CONS:.0%}")
    print(f"  Regime threshold: trend_strength > {config.REGIME_TREND_THRESHOLD}")
    print("━"*62)

    print("\n  Loading cache panel...")
    panel, btc_df = load_cache_panel()
    print(f"  Panel: {panel['symbol'].nunique()} symbols, "
          f"{panel['datetime'].nunique()} timestamps")
    print(f"  Window: {panel['datetime'].min()} → {panel['datetime'].max()}")

    regime_map = pd.Series(dtype=str)
    if not btc_df.empty:
        regime_map = compute_regime_map(btc_df)
        n_trend    = (regime_map == 'trend').sum()
        n_rev      = (regime_map == 'reversal').sum()
        print(f"  Regime dist: TREND={n_trend} ({n_trend/len(regime_map):.0%})  "
              f"REVERSAL={n_rev} ({n_rev/len(regime_map):.0%})")
    else:
        print("  ⚠ BTC data tidak ditemukan — semua timestamp dianggap 'both'")

    print(f"\n  {'Factor':<22} {'Regime':<10} {'med_ICIR':>9}  "
          f"{'sign_cons':>9}  {'windows':>7}  ROBUST?")
    print(f"  {'─'*60}")

    robust_trend    = []
    robust_reversal = []
    robust_both     = []

    all_factor_cols = [c for c in panel.columns
                       if c in FACTOR_REGIME_MAP]

    for factor_col in FACTOR_REGIME_MAP:
        if factor_col not in panel.columns:
            print(f"  {factor_col:<22} {'—':<10} {'N/A':>9}  {'N/A':>9}  {'—':>7}  ✗ MISSING")
            continue

        target = FACTOR_REGIME_MAP[factor_col]
        wf_df  = wf_icir_regime(panel, factor_col, regime_map, target)
        ok, st = is_robust(wf_df)

        flag = '✓ YES' if ok else '✗ NO '
        icir_str = f"{st['median_icir']:+.4f}" if pd.notna(st['median_icir']) else '   N/A'
        cons_str = f"{st['sign_cons']:.0%}"    if pd.notna(st['sign_cons'])    else '  N/A'

        print(f"  {factor_col:<22} {target:<10} {icir_str:>9}  "
              f"{cons_str:>9}  {st['n_windows']:>7}  {flag}")

        if ok:
            if target == 'trend':
                robust_trend.append(factor_col)
            elif target == 'reversal':
                robust_reversal.append(factor_col)
            else:
                robust_both.append(factor_col)

    # Sentiment selalu di kedua regime
    final_trend    = robust_trend    + robust_both
    final_reversal = robust_reversal + robust_both

    print("\n" + "━"*62)
    print("  REKOMENDASI — paste ke TS_FACTOR_COLS di factors.py:")
    print("━"*62)
    print(f"\nTS_FACTOR_COLS = {{")
    print(f"    'trend'   : {final_trend},")
    print(f"    'reversal': {final_reversal},")
    print(f"}}")
    print()

    # Simpan hasil ke output
    os.makedirs('./output', exist_ok=True)
    rows = []
    for fc in FACTOR_REGIME_MAP:
        if fc not in panel.columns:
            continue
        t  = FACTOR_REGIME_MAP[fc]
        wf = wf_icir_regime(panel, fc, regime_map, t)
        ok, st = is_robust(wf)
        rows.append({'factor': fc, 'regime': t, 'robust': ok, **st})
    pd.DataFrame(rows).to_csv('./output/wf_factor_validation.csv', index=False)
    print("  ✓ Hasil disimpan ke ./output/wf_factor_validation.csv")
    print("━"*62 + "\n")

    return {'trend': final_trend, 'reversal': final_reversal}


if __name__ == "__main__":
    main()
