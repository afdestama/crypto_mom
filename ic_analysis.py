# ic_analysis.py — Cross-Sectional IC, ICIR, Composite Score (4H)

import pandas as pd
import numpy as np
from scipy import stats
from tqdm import tqdm
import config


def cross_sectional_ic(panel: pd.DataFrame,
                        factor_cols: list,
                        date_col: str = 'datetime',
                        forward_col: str = 'forward_1d') -> tuple:
    """
    Hitung IC cross-sectional per candle (4H period).
    
    Untuk setiap timestamp: korelasi rank(factor) vs rank(forward_return)
    di semua coin → rata-ratakan sepanjang waktu.

    Returns:
        summary : IC Mean, IC Std, ICIR, t-stat per factor
        ic_ts   : time-series IC per factor
    """
    results = {col: [] for col in factor_cols}
    times   = []

    grouped = panel.groupby(date_col)
    for dt, group in tqdm(grouped, desc="Computing IC", ncols=80):
        valid = group.dropna(subset=[forward_col])
        if len(valid) < config.MIN_COINS_PER_PERIOD:
            continue
        times.append(dt)

        for col in factor_cols:
            sub = valid[[col, forward_col]].dropna()
            if len(sub) < config.MIN_COINS_PER_PERIOD:
                results[col].append(np.nan)
                continue
            ic, _ = stats.spearmanr(sub[col], sub[forward_col])
            results[col].append(ic)

    ic_ts = pd.DataFrame(results, index=times)

    summary_rows = []
    for col in factor_cols:
        vals = ic_ts[col].dropna().values
        if len(vals) < 20:
            continue
        ic_mean = np.nanmean(vals)
        ic_std  = np.nanstd(vals)
        icir    = ic_mean / (ic_std + 1e-9)
        t_stat  = ic_mean / (ic_std / np.sqrt(len(vals)) + 1e-9)
        pct_pos = (vals > 0).mean() * 100

        summary_rows.append({
            'factor'        : col,
            'ic_mean'       : round(ic_mean, 5),
            'ic_std'        : round(ic_std, 5),
            'icir'          : round(icir, 4),
            'ic_positive_%' : round(pct_pos, 1),
            't_stat'        : round(t_stat, 3),
            'n_obs'         : len(vals)
        })

    summary = pd.DataFrame(summary_rows).sort_values('icir', ascending=False)
    return summary, ic_ts


def check_ic_drift(
    ic_ts: pd.DataFrame,
    ic_summary: pd.DataFrame,
    recent_window: int = None,
    drift_threshold: float = None,
) -> list:
    """
    Bandingkan ICIR full-window vs recent_window candle terakhir per faktor.
    Return list dict untuk faktor yang ICIR_recent/ICIR_full < drift_threshold.
    """
    recent_window   = recent_window   or config.IC_DRIFT_RECENT_WINDOW
    drift_threshold = drift_threshold or config.IC_DRIFT_THRESHOLD

    if ic_ts is None or ic_ts.empty or ic_summary is None or ic_summary.empty:
        return []

    # ic_ts index = datetime; reset agar sort_values/tail bekerja konsisten
    ts = ic_ts.reset_index() if ic_ts.index.name == 'datetime' else ic_ts.copy()
    if 'datetime' not in ts.columns:
        ts = ts.rename(columns={ts.columns[0]: 'datetime'})
    ts = ts.sort_values('datetime')
    recent_ts = ts.tail(recent_window)

    drifted = []
    for _, row in ic_summary.iterrows():
        fc = row['factor']
        if fc not in ts.columns:
            continue
        full_icir = abs(row.get('icir', 0) or 0)
        if full_icir < 1e-6:
            continue
        recent_vals = recent_ts[fc].dropna()
        if len(recent_vals) < 10:
            continue
        recent_icir = abs(recent_vals.mean() / (recent_vals.std() + 1e-9))
        ratio = recent_icir / full_icir
        if ratio < drift_threshold:
            drifted.append({
                'factor'      : fc,
                'full_icir'   : round(full_icir, 4),
                'recent_icir' : round(recent_icir, 4),
                'ratio'       : round(ratio, 3),
            })

    return drifted


def build_composite_score(
    panel: pd.DataFrame,
    ic_summary: pd.DataFrame,
    min_icir: float = config.MIN_ICIR,
    sentiment_cols: list = None,
) -> tuple:
    """
    IC-weighted composite score (unified).

    Faktor utama: ICIR > min_icir (positif saja — negatif weight berbahaya di trend).
    Sentiment cols (funding/LS/OI): force-include jika ICIR > 0, walau < min_icir.
    Semua weight proporsional ICIR — tidak ada yang hardcoded.
    """
    # Faktor utama
    main = ic_summary[ic_summary['icir'] > min_icir].copy()
    main['_sent'] = False

    if main.empty:
        print(f"  ⚠ Tidak ada factor ICIR > +{min_icir}, fallback ke ICIR > 0")
        main = ic_summary[ic_summary['icir'] > 0].copy()
        main['_sent'] = False

    if main.empty:
        print(f"  ⚠ Semua factor ICIR negatif. Pakai top 2 by ICIR")
        main = ic_summary.nlargest(2, 'icir').copy()
        main['_sent'] = False

    # Sentiment cols: tambahkan jika ICIR > 0 dan belum ada di main
    if sentiment_cols:
        sent = ic_summary[
            ic_summary['factor'].isin(sentiment_cols) &
            (ic_summary['icir'] > 0)
        ].copy()
        sent = sent[~sent['factor'].isin(main['factor'])]
        sent['_sent'] = True
        main = pd.concat([main, sent], ignore_index=True)

    # Normalize weight proporsional ICIR
    total_icir = main['icir'].sum()
    main['weight'] = main['icir'] / (total_icir + 1e-9)

    print(f"\n  Composite menggunakan {len(main)} factors:")
    for _, row in main.iterrows():
        tag = ' [sentiment]' if row['_sent'] else ''
        bar = '█' * int(abs(row['icir']) * 30)
        print(f"    {row['factor']:<24} ICIR={row['icir']:+.4f}  "
              f"w={row['weight']:+.3f}  {bar}{tag}")

    panel = panel.copy()
    panel['composite_score'] = 0.0
    for _, row in main.iterrows():
        col = row['factor']
        w   = row['weight']
        if col in panel.columns:
            panel['composite_score'] += w * panel[col].fillna(0)

    return panel, main


def rank_coins(panel: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
    """Ranking coins berdasarkan composite score pada timestamp tertentu."""
    snap = panel[panel['datetime'] == dt].copy()
    snap = snap.dropna(subset=['composite_score'])
    snap['rank'] = snap['composite_score'].rank(ascending=False)
    return snap.sort_values('rank')
