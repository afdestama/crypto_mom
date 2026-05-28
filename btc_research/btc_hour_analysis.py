# btc_hour_analysis.py — Analisis per-jam UTC: kapan paling reliable untuk
# screening (factor IC kuat) dan eksekusi (liquidity tinggi, return clean).
#
# 4H candle UTC: 00, 04, 08, 12, 16, 20.
#   00:00 UTC → US close / Asia AM
#   04:00 UTC → Asia day / Europe pre-market
#   08:00 UTC → Europe open  + funding payment
#   12:00 UTC → Europe/US pre-market
#   16:00 UTC → US open / Europe close + funding payment
#   20:00 UTC → US afternoon / Asia overnight
# Funding rate Binance perp dibayar 00 / 08 / 16 UTC.

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_research.btc_data    import fetch_btc_research_data
from btc_research.btc_factors import build_btc_factors, FACTOR_COLS


FORWARD_PERIODS = [3, 6, 12]
HOURS           = [0, 4, 8, 12, 16, 20]
HOUR_LABEL = {
    0  : 'US close/Asia AM',
    4  : 'Asia day/EU pre',
    8  : 'EU open  ⚡fund',
    12 : 'EU/US pre-mkt',
    16 : 'US open  ⚡fund',
    20 : 'US PM/Asia eve',
}


def _fmt_pct(x, p=2):
    return '   nan' if pd.isna(x) else f"{x*100:+.{p}f}%"


def per_hour_return_table(btc_df, fdf, fp=6):
    """Stats forward return + liquidity per jam."""
    df = btc_df[['close', 'volume', 'funding_rate']].copy()
    df['hour']    = df.index.hour
    df['fwd_ret'] = fdf[f'forward_ret_{fp}']

    print(f"\n=== Return & Liquidity per jam (FP={fp} candle = {fp*4}h horizon) ===")
    print(f"  {'hour':<6}{'label':<22}{'n':>6}{'mean_ret':>10}{'std':>10}"
          f"{'sharpe':>9}{'win%':>7}{'vol_avg':>11}{'|fund|_avg':>12}")
    rows = []
    for h in HOURS:
        sub = df[df['hour'] == h].dropna(subset=['fwd_ret'])
        if len(sub) < 20:
            continue
        ret = sub['fwd_ret']
        sharpe_d = ret.mean() / (ret.std() + 1e-12) * np.sqrt(365)  # daily-equivalent
        rows.append({
            'hour'    : h,
            'n'       : len(sub),
            'mean_ret': ret.mean(),
            'std'     : ret.std(),
            'sharpe_d': sharpe_d,
            'win_pct' : (ret > 0).mean(),
            'vol'     : sub['volume'].mean(),
            'fund_abs': sub['funding_rate'].abs().mean() if 'funding_rate' in sub else np.nan,
        })

    for r in rows:
        print(f"  {r['hour']:>2}:00 {HOUR_LABEL[r['hour']]:<22}{r['n']:>6}"
              f"{_fmt_pct(r['mean_ret']):>10}{_fmt_pct(r['std']):>10}"
              f"{r['sharpe_d']:>+9.2f}{r['win_pct']*100:>6.1f}%"
              f"{r['vol']/1e3:>10.0f}K"
              f"{r['fund_abs']*100:>+11.4f}%" if pd.notna(r['fund_abs']) else "")
    return rows


def per_hour_factor_ic(fdf, fp=6):
    """IC per factor di tiap jam — kapan factor paling predictive."""
    fdf = fdf.copy()
    fdf['hour'] = fdf.index.hour
    target_col  = f'forward_ret_{fp}'

    print(f"\n=== Factor IC per jam (FP={fp}, |IC|>0.1 di-highlight) ===")
    print(f"  {'factor':<18}", end='')
    for h in HOURS:
        print(f"   h{h:02d}", end='')
    print(f"   {'best_hour':>10}")

    ic_matrix = {}
    for f in FACTOR_COLS:
        if f not in fdf.columns:
            continue
        row_ic = {}
        for h in HOURS:
            sub = fdf[fdf['hour'] == h]
            mask = sub[f].notna() & sub[target_col].notna()
            if mask.sum() < 30:
                row_ic[h] = np.nan
                continue
            ic, _ = spearmanr(sub.loc[mask, f], sub.loc[mask, target_col])
            row_ic[h] = float(ic) if not np.isnan(ic) else np.nan
        ic_matrix[f] = row_ic

        best_h = max(row_ic, key=lambda x: abs(row_ic[x]) if pd.notna(row_ic[x]) else -1)
        print(f"  {f:<18}", end='')
        for h in HOURS:
            v = row_ic[h]
            if pd.isna(v):
                print(f"  nan ", end='')
            else:
                marker = '*' if abs(v) > 0.1 else ' '
                print(f"{v:+.2f}{marker}", end='')
        print(f"   h{best_h:02d}={row_ic[best_h]:+.3f}" if pd.notna(row_ic[best_h]) else "    n/a")

    return ic_matrix


def composite_ic_per_hour(fdf, ic_matrix, fp=6):
    """Composite (signed by majority-sign across all hours) IC per jam."""
    target_col = f'forward_ret_{fp}'
    # Pakai factor robust: |median IC across hours| > 0.05
    chosen = {}
    for f, row_ic in ic_matrix.items():
        vals = [v for v in row_ic.values() if pd.notna(v)]
        if not vals:
            continue
        median_ic = np.median(vals)
        if abs(median_ic) >= 0.05:
            chosen[f] = +1.0 if median_ic > 0 else -1.0

    if not chosen:
        print("\n(No factors pass robustness threshold for composite)")
        return

    fdf = fdf.copy()
    fdf['hour'] = fdf.index.hour
    parts = [chosen[f] * fdf[f] for f in chosen]
    composite = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    fdf['_comp'] = composite

    print(f"\n=== Composite IC per jam (FP={fp}, {len(chosen)} factors signed by median IC) ===")
    print(f"  {'hour':<6}{'label':<22}{'n':>6}{'comp_IC':>10}{'t-stat':>9}")
    for h in HOURS:
        sub = fdf[fdf['hour'] == h]
        mask = sub['_comp'].notna() & sub[target_col].notna()
        n = int(mask.sum())
        if n < 30:
            continue
        ic, _ = spearmanr(sub.loc[mask, '_comp'], sub.loc[mask, target_col])
        # rough t-stat: t = ic × sqrt(n-2) / sqrt(1-ic²)
        t = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-12) if not np.isnan(ic) else np.nan
        print(f"  {h:>2}:00 {HOUR_LABEL[h]:<22}{n:>6}{ic:>+10.3f}{t:>+9.2f}")


def main():
    print("Loading BTC data (1 tahun)...")
    btc_df = fetch_btc_research_data(force_refresh=False)
    print(f"  shape={btc_df.shape}  range={btc_df.index.min()} → {btc_df.index.max()}")

    fdf = build_btc_factors(btc_df)
    print(f"  factor panel shape={fdf.shape}")

    # Coverage check per hour
    print(f"\nDistribusi candle per jam UTC:")
    hour_counts = btc_df.index.hour.value_counts().sort_index()
    for h in HOURS:
        print(f"  {h:>2}:00 → {hour_counts.get(h, 0)} candles")

    # 1. Return + liquidity per hour
    per_hour_return_table(btc_df, fdf, fp=6)
    per_hour_return_table(btc_df, fdf, fp=12)

    # 2. Factor IC per hour
    ic_matrix = per_hour_factor_ic(fdf, fp=6)

    # 3. Composite IC per hour
    composite_ic_per_hour(fdf, ic_matrix, fp=6)


if __name__ == '__main__':
    main()
