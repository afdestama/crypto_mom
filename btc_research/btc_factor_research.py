# btc_factor_research.py — Time-series IC/ICIR diagnostic untuk BTC-only.
#
# Single asset → BUKAN cross-sectional IC. Untuk setiap (factor, forward_period):
#   - full-sample IC (Spearman),
#   - ICIR via rolling Spearman over IC_ROLLING_WINDOW,
#   - sign-stability H1 vs H2,
#   - per-regime IC (bull/bear).
# Sweep FP ∈ {3, 6, 12, 24} → cek di horizon mana setiap factor "hidup".

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from btc_research.btc_data    import fetch_btc_research_data
from btc_research.btc_factors import (
    build_btc_factors, FACTOR_COLS, FACTOR_GROUP, FORWARD_PERIODS,
)


MIN_OBS_IC      = 50           # minimum overlapping non-NaN sample
MIN_HALF_OBS    = 30           # minimum sample per half (H1/H2) untuk sign-test
IC_MEAN_ROBUST  = 0.05         # threshold robust: |IC mean rolling| ≥ ini
IC_HALF_MIN     = 0.02         # minimum |IC| di each half untuk dianggap sign-stable
RANK_WINDOW     = config.IC_ROLLING_WINDOW   # 120 candle (~20 hari) untuk rolling IC


def _safe_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < MIN_OBS_IC:
        return np.nan
    rho, _ = spearmanr(x[mask], y[mask])
    return float(rho) if not np.isnan(rho) else np.nan


def _rolling_ic(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Rolling Spearman via rolling rank Pearson (Spearman ≡ Pearson of ranks)."""
    xr = x.rolling(window, min_periods=window // 2).rank(pct=True)
    yr = y.rolling(window, min_periods=window // 2).rank(pct=True)
    return xr.rolling(window, min_periods=window // 2).corr(yr)


def evaluate_factor(fdf: pd.DataFrame, factor: str, fp: int) -> dict:
    x      = fdf[factor]
    y      = fdf[f'forward_ret_{fp}']
    mask   = x.notna() & y.notna()
    n_obs  = int(mask.sum())

    if n_obs < MIN_OBS_IC:
        return {
            'factor': factor, 'group': FACTOR_GROUP.get(factor, '?'),
            'fp': fp, 'n_obs': n_obs,
            'ic_full': np.nan, 'ic_mean': np.nan,
            'ic_h1': np.nan, 'ic_h2': np.nan, 'sign_stable': False,
            'ic_bull': np.nan, 'ic_bear': np.nan,
            'robust': False,
        }

    ic_full = _safe_spearman(x, y)

    # Rolling IC → mean (rata-rata IC sepanjang waktu)
    ic_ts   = _rolling_ic(x, y, RANK_WINDOW).dropna()
    ic_mean = float(ic_ts.mean()) if len(ic_ts) >= 30 else np.nan

    # H1/H2 sign stability
    idx     = fdf.index[mask]
    cutoff  = idx[len(idx) // 2]
    h1_mask = mask & (fdf.index <= cutoff)
    h2_mask = mask & (fdf.index >  cutoff)

    if h1_mask.sum() >= MIN_HALF_OBS and h2_mask.sum() >= MIN_HALF_OBS:
        ic_h1 = _safe_spearman(x[h1_mask], y[h1_mask])
        ic_h2 = _safe_spearman(x[h2_mask], y[h2_mask])
        sign_stable = (
            not np.isnan(ic_h1) and not np.isnan(ic_h2)
            and np.sign(ic_h1) == np.sign(ic_h2)
            and abs(ic_h1) >= IC_HALF_MIN
            and abs(ic_h2) >= IC_HALF_MIN
        )
    else:
        ic_h1 = ic_h2 = np.nan
        sign_stable = False

    # Per-regime IC
    bull_mask = mask & (fdf['regime'] == 'bull')
    bear_mask = mask & (fdf['regime'] == 'bear')
    ic_bull = _safe_spearman(x[bull_mask], y[bull_mask]) if bull_mask.sum() >= MIN_HALF_OBS else np.nan
    ic_bear = _safe_spearman(x[bear_mask], y[bear_mask]) if bear_mask.sum() >= MIN_HALF_OBS else np.nan

    robust = (
        not np.isnan(ic_mean) and abs(ic_mean) >= IC_MEAN_ROBUST and sign_stable
    )

    return {
        'factor': factor, 'group': FACTOR_GROUP.get(factor, '?'),
        'fp': fp, 'n_obs': n_obs,
        'ic_full': ic_full, 'ic_mean': ic_mean,
        'ic_h1': ic_h1, 'ic_h2': ic_h2, 'sign_stable': sign_stable,
        'ic_bull': ic_bull, 'ic_bear': ic_bear,
        'robust': robust,
    }


def _fmt(x, w=7, p=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ' ' * (w - 3) + 'nan'
    return f"{x:>{w}.{p}f}"


def print_table(rows: list, fp: int):
    rows = sorted(rows, key=lambda r: (-abs(r['ic_mean']) if not np.isnan(r['ic_mean']) else 1.0))
    print(f"\n=== Forward Period = {fp} candle ({fp * 4}h) ===")
    print(f"  {'factor':<18}{'group':<10}{'n':>5}  {'ic_full':>8}{'ic_mean':>8}{'ic_h1':>8}{'ic_h2':>8}"
          f"{'stab':>6}{'ic_bull':>9}{'ic_bear':>9}  flag")
    for r in rows:
        flag = ' ★' if r['robust'] else ''
        print(
            f"  {r['factor']:<18}{r['group']:<10}{r['n_obs']:>5}  "
            f"{_fmt(r['ic_full'])}{_fmt(r['ic_mean'])}{_fmt(r['ic_h1'])}{_fmt(r['ic_h2'])}"
            f"{'Y' if r['sign_stable'] else 'N':>6}"
            f"{_fmt(r['ic_bull'], 9)}{_fmt(r['ic_bear'], 9)}"
            f"{flag}"
        )


def print_top_combinations(all_rows: list, k: int = 10):
    valid = [r for r in all_rows if not np.isnan(r['ic_mean'])]
    valid.sort(key=lambda r: -abs(r['ic_mean']))
    print(f"\n=== TOP {k} (factor, fp) by |IC mean| ===")
    print(f"  {'rank':<6}{'factor':<18}{'fp':>4}{'group':<12}{'ic_mean':>8}"
          f"{'ic_full':>9}{'stab':>6}{'robust':>8}")
    for i, r in enumerate(valid[:k], 1):
        print(
            f"  {i:<6}{r['factor']:<18}{r['fp']:>4}{r['group']:<12}"
            f"{_fmt(r['ic_mean'])}{_fmt(r['ic_full'], 9)}"
            f"{'Y' if r['sign_stable'] else 'N':>6}"
            f"{'Y' if r['robust'] else 'N':>8}"
        )


def main():
    print("Loading BTC data...")
    btc_df = fetch_btc_research_data(force_refresh=False)
    print(f"  shape={btc_df.shape}  range={btc_df.index.min()} → {btc_df.index.max()}")
    print(f"  funding_rate non-null: {btc_df.get('funding_rate', pd.Series()).notna().sum()}")
    print(f"  long_short_ratio non-null: {btc_df.get('long_short_ratio', pd.Series()).notna().sum()}")
    print(f"  open_interest non-null: {btc_df.get('open_interest', pd.Series()).notna().sum()}")

    print("\nBuilding factors...")
    fdf = build_btc_factors(btc_df)
    print(f"  factor panel shape={fdf.shape}")
    n_bull = (fdf['regime'] == 'bull').sum()
    n_bear = (fdf['regime'] == 'bear').sum()
    print(f"  regime counts: bull={n_bull}, bear={n_bear}")

    all_rows = []
    for fp in FORWARD_PERIODS:
        rows = [evaluate_factor(fdf, f, fp) for f in FACTOR_COLS]
        print_table(rows, fp)
        all_rows.extend(rows)

    print_top_combinations(all_rows, k=10)

    robust = [r for r in all_rows if r['robust']]
    print(f"\nRobust factors (|IC mean|≥{IC_MEAN_ROBUST} & sign-stable across halves): {len(robust)}")
    for r in robust:
        print(f"  - {r['factor']:<18} fp={r['fp']:>2} ic_mean={r['ic_mean']:+.3f}  group={r['group']}")


if __name__ == '__main__':
    main()
