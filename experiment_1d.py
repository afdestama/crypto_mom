# experiment_1d.py — Factor experiment pada timeframe 1D
#
# Resample cache 4H → 1D, build factor set dari screenshot,
# hitung cross-sectional IC, tampilkan bar chart.
#
# Usage: python experiment_1d.py

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config

# ── Konstanta ──────────────────────────────────────────────
FORWARD_PERIOD   = 1      # 1 hari ke depan
ZSCORE_WINDOW    = 60     # rolling window z-score (hari)
LOOKBACK_DAYS_1D = 730    # 2 tahun OHLCV (funding/LS/OI tetap ~30 hari dari API)
MIN_COINS        = 5      # minimum coin per period untuk hitung IC
IC_THRESHOLD     = 0.03   # garis referensi di chart
OUTPUT_DIR       = './output'


# ── Helpers ────────────────────────────────────────────────

def zscore(series: pd.Series, window: int) -> pd.Series:
    m = series.rolling(window, min_periods=window // 2).mean()
    s = series.rolling(window, min_periods=window // 2).std()
    return (series - m) / (s + 1e-9)


# ── Fetch 1D Data ──────────────────────────────────────────

import time
from datetime import datetime, timedelta
import ccxt
from data_fetcher import (get_exchange, fetch_funding_rate,
                          fetch_long_short_ratio, fetch_open_interest,
                          load_current_universe)

CACHE_1D_DIR = './cache_1d'


def fetch_ohlcv_1d(exchange, symbol: str,
                   days: int = LOOKBACK_DAYS_1D) -> pd.DataFrame:
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00Z')
    )
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=500)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            if len(ohlcv) < 500:
                break
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"  OHLCV 1d error {symbol}: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('datetime').set_index('datetime').sort_index()
    return df[['open','high','low','close','volume']].astype(float)


def load_or_fetch_symbol_1d(symbol: str, exchange=None) -> pd.DataFrame:
    os.makedirs(CACHE_1D_DIR, exist_ok=True)
    safe   = symbol.replace('/', '_').replace(':', '_')
    path   = f"{CACHE_1D_DIR}/{safe}_1d.parquet"

    if exchange is None:
        exchange = get_exchange()

    ohlcv = fetch_ohlcv_1d(exchange, symbol, days=LOOKBACK_DAYS_1D)
    if ohlcv.empty:
        return pd.DataFrame()

    # Funding: resample 8h → 1D (mean per hari)
    funding = fetch_funding_rate(exchange, symbol)
    if not funding.empty:
        funding = funding.resample('1D').mean()

    # L/S: resample 4h → 1D (last per hari)
    ls = fetch_long_short_ratio(exchange, symbol)
    if not ls.empty:
        ls = ls.resample('1D').last()

    # OI: resample 4h → 1D (last per hari)
    oi = fetch_open_interest(exchange, symbol)
    if not oi.empty:
        oi = oi.resample('1D').last()

    df = ohlcv.copy()
    for extra in [funding, ls, oi]:
        if not extra.empty:
            df = df.join(extra, how='left')

    df = df.ffill(limit=1).bfill(limit=1)
    df['symbol'] = symbol
    df.to_parquet(path)
    return df


def load_daily_panel(force_refresh: bool = False) -> pd.DataFrame:
    """Fetch 1D OHLCV + funding/LS/OI langsung dari Binance."""
    universe = load_current_universe()
    if not universe:
        raise ValueError("Universe kosong — jalankan main.py dulu.")

    os.makedirs(CACHE_1D_DIR, exist_ok=True)
    exchange = get_exchange()
    frames   = []

    print(f"  Fetching 1D data untuk {len(universe)} symbols...")
    for symbol in tqdm(universe, desc='Fetching 1D', ncols=80):
        safe = symbol.replace('/', '_').replace(':', '_')
        path = f"{CACHE_1D_DIR}/{safe}_1d.parquet"

        if not force_refresh and os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if not df.empty:
                    frames.append(df.reset_index())
                    continue
            except Exception:
                pass

        try:
            df = load_or_fetch_symbol_1d(symbol, exchange)
            if not df.empty and len(df) >= 30:
                frames.append(df.reset_index())
        except Exception as e:
            print(f"  Error {symbol}: {e}")
        time.sleep(0.25)

    if not frames:
        raise ValueError("Tidak ada data 1D berhasil di-fetch.")

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(columns={'index': 'datetime'}) if 'index' in panel.columns else panel
    panel['datetime'] = pd.to_datetime(panel['datetime'])
    return panel.sort_values(['datetime', 'symbol']).reset_index(drop=True)


# ── Build Factors ──────────────────────────────────────────

def build_factors_1d(df: pd.DataFrame) -> pd.DataFrame:
    """Build semua faktor 1D untuk satu coin."""
    f   = pd.DataFrame(index=df.index)
    ret = df['close'].pct_change()

    # 1. reversal_1d: kontrarian return kemarin
    f['reversal_1d'] = zscore(-ret, window=ZSCORE_WINDOW)

    # 2. liquidity_30d: volume relatif vs 30d
    f['liquidity_30d'] = zscore(df['volume'], window=30)

    # 3. liquidation_imbalance: body direction × volume z-score
    rng      = (df['high'] - df['low']).replace(0, np.nan)
    body_dir = (df['close'] - df['open']) / rng
    vol_z    = zscore(df['volume'], window=30)
    f['liquidation_imbalance'] = zscore(body_dir * vol_z.clip(-3, 3), window=ZSCORE_WINDOW)

    # 4. funding_rate_contrarian: -funding (contrarian)
    if 'funding_rate' in df.columns:
        f['funding_rate_contrarian'] = zscore(-df['funding_rate'].fillna(0), window=ZSCORE_WINDOW)

    # 5. vol_compression_30d: realized vol rendah vs historis = kompresi
    realized_vol = ret.rolling(5).std()
    f['vol_compression_30d'] = zscore(-realized_vol, window=30)

    # 6. ls_ratio_contrarian: -L/S ratio (crowd terlalu long = bearish)
    if 'long_short_ratio' in df.columns:
        f['ls_ratio_contrarian'] = zscore(-df['long_short_ratio'], window=ZSCORE_WINDOW)

    # 7. momentum_30d: price ROC 30 hari
    f['momentum_30d'] = zscore(df['close'].pct_change(30), window=ZSCORE_WINDOW)

    # 8. volume_compression_30d: volume sekarang vs MA30 volume (rendah = kompresi)
    vol_ma = df['volume'].rolling(30).mean()
    f['volume_compression_30d'] = zscore(-(df['volume'] / (vol_ma + 1e-9)), window=ZSCORE_WINDOW)

    # 9. oi_price_signal: OI change × price direction
    if 'open_interest' in df.columns:
        oi_chg    = df['open_interest'].pct_change()
        price_dir = np.sign(ret)
        f['oi_price_signal'] = zscore(oi_chg * price_dir, window=ZSCORE_WINDOW)

    # Forward return 1 hari
    f['forward_1d'] = df['close'].shift(-FORWARD_PERIOD) / df['close'] - 1

    return f


def build_panel_1d(panel: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for sym, grp in panel.groupby('symbol'):
        grp = grp.sort_values('datetime').set_index('datetime')
        fdf = build_factors_1d(grp)
        fdf['symbol']   = sym
        fdf['datetime'] = fdf.index
        frames.append(fdf.reset_index(drop=True))
    return pd.concat(frames).sort_values(['datetime', 'symbol']).reset_index(drop=True)


# ── Cross-Sectional IC ─────────────────────────────────────

def compute_ic(panel: pd.DataFrame) -> pd.DataFrame:
    factor_cols = [c for c in panel.columns
                   if c not in ('symbol', 'datetime', 'forward_1d')
                   and panel[c].notna().sum() > 50]

    results = {col: [] for col in factor_cols}
    times   = []

    for dt, grp in tqdm(panel.groupby('datetime'), desc='Computing IC', ncols=80):
        valid = grp.dropna(subset=['forward_1d'])
        if len(valid) < MIN_COINS:
            continue
        times.append(dt)
        for col in factor_cols:
            sub = valid[[col, 'forward_1d']].dropna()
            if len(sub) < MIN_COINS:
                results[col].append(np.nan)
                continue
            ic, _ = stats.spearmanr(sub[col], sub['forward_1d'])
            results[col].append(ic)

    ic_ts = pd.DataFrame(results, index=times)

    rows = []
    for col in factor_cols:
        vals = ic_ts[col].dropna().values
        if len(vals) < 10:
            continue
        ic_mean = np.nanmean(vals)
        ic_std  = np.nanstd(vals)
        icir    = ic_mean / (ic_std + 1e-9)
        t_stat  = ic_mean / (ic_std / np.sqrt(len(vals)) + 1e-9)
        rows.append({
            'factor'  : col,
            'ic_mean' : round(ic_mean, 5),
            'ic_std'  : round(ic_std, 5),
            'icir'    : round(icir, 4),
            't_stat'  : round(t_stat, 3),
            'n_obs'   : len(vals),
        })

    summary = pd.DataFrame(rows).sort_values('ic_mean', ascending=False)
    return summary, ic_ts


# ── Walk-Forward Validation ────────────────────────────────

WF_TRAIN_DAYS = 60
WF_STEP_DAYS  = 15
WF_MIN_IC_OBS = 8
WF_MIN_ICIR   = 0.10
WF_SIGN_CONS  = 0.60


def wf_validate(panel: pd.DataFrame, factor_col: str) -> pd.DataFrame:
    """OOS IC per expanding window untuk satu faktor."""
    all_dts = sorted(panel['datetime'].unique())
    results = []

    for i in range(WF_TRAIN_DAYS, len(all_dts) - WF_STEP_DAYS, WF_STEP_DAYS):
        oos_dts   = all_dts[i : i + WF_STEP_DAYS]
        oos_panel = panel[panel['datetime'].isin(oos_dts)]

        ic_vals = []
        for _, grp in oos_panel.groupby('datetime'):
            sub = grp[[factor_col, 'forward_1d']].dropna()
            if len(sub) >= MIN_COINS:
                ic, _ = stats.spearmanr(sub[factor_col], sub['forward_1d'])
                if pd.notna(ic):
                    ic_vals.append(ic)

        if len(ic_vals) >= WF_MIN_IC_OBS:
            arr = np.array(ic_vals)
            results.append({
                'window'  : i // WF_STEP_DAYS,
                'oos_ic'  : round(arr.mean(), 5),
                'oos_icir': round(arr.mean() / (arr.std() + 1e-9), 4),
                'n'       : len(arr),
            })

    return pd.DataFrame(results)


def is_robust(wf_df: pd.DataFrame) -> tuple:
    if wf_df.empty:
        return False, {'median_icir': np.nan, 'sign_cons': np.nan, 'n_windows': 0}
    median_icir   = wf_df['oos_icir'].median()
    dominant_sign = np.sign(wf_df['oos_ic'].median())
    sign_cons     = (np.sign(wf_df['oos_ic']) == dominant_sign).mean()
    robust = abs(median_icir) > WF_MIN_ICIR and sign_cons >= WF_SIGN_CONS
    return robust, {
        'median_icir': round(median_icir, 4),
        'sign_cons'  : round(sign_cons, 3),
        'n_windows'  : len(wf_df),
        'sign'       : int(dominant_sign),
    }


# ── Chart ──────────────────────────────────────────────────

def plot_ic_bar(summary: pd.DataFrame, output_path: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#c2185b' if v >= 0 else '#880e4f' for v in summary['ic_mean']]
    bars   = ax.barh(summary['factor'], summary['ic_mean'], color=colors)

    ax.axvline(x=IC_THRESHOLD,  color='gray', linestyle='--', linewidth=1,
               label=f'IC={IC_THRESHOLD}')
    ax.axvline(x=-IC_THRESHOLD, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=0.5)

    ax.set_xlabel('IC Mean')
    ax.set_title(f'IC Mean per Sinyal (Spearman, {FORWARD_PERIOD}-day forward return)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  ✓ Chart disimpan → {output_path}")


# ── Main ───────────────────────────────────────────────────

def main(force_refresh: bool = False):
    print("\n" + "━"*60)
    print("  EXPERIMENT 1D — Factor IC Analysis")
    print(f"  Forward: {FORWARD_PERIOD}d  |  Z-score window: {ZSCORE_WINDOW}d")
    print("━"*60)

    print("\n  Fetching 1D data dari Binance...")
    raw = load_daily_panel(force_refresh=force_refresh)
    print(f"  Symbols : {raw['symbol'].nunique()}")
    print(f"  Window  : {raw['datetime'].min().date()} → {raw['datetime'].max().date()}")

    print("\n  Building factors...")
    panel = build_panel_1d(raw)
    print(f"  Panel   : {panel.shape}")

    available = [c for c in panel.columns
                 if c not in ('symbol', 'datetime', 'forward_1d')
                 and panel[c].notna().sum() > 50]
    print(f"  Factors : {available}")

    print("\n  Computing IC...")
    summary, ic_ts = compute_ic(panel)

    print(f"\n  {'Factor':<28} {'IC Mean':>8}  {'ICIR':>7}  {'t-stat':>7}  {'n':>5}")
    print(f"  {'─'*55}")
    for _, r in summary.iterrows():
        flag = '✓' if abs(r['icir']) > 0.15 else ' '
        print(f"  {flag} {r['factor']:<26} {r['ic_mean']:>+8.5f}  "
              f"{r['icir']:>+7.4f}  {r['t_stat']:>+7.3f}  {int(r['n_obs']):>5}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary.to_csv(f'{OUTPUT_DIR}/ic_summary_1d.csv', index=False)
    ic_ts.to_csv(f'{OUTPUT_DIR}/ic_ts_1d.csv')
    print(f"\n  ✓ IC summary → {OUTPUT_DIR}/ic_summary_1d.csv")

    plot_ic_bar(summary, f'{OUTPUT_DIR}/ic_bar_1d.png')

    # ── Walk-Forward Validation ─────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  WALK-FORWARD VALIDATION  "
          f"(train={WF_TRAIN_DAYS}d  step={WF_STEP_DAYS}d  "
          f"min_ICIR={WF_MIN_ICIR}  sign_cons={WF_SIGN_CONS:.0%})")
    print(f"  {'─'*60}")
    print(f"  {'Factor':<28} {'med_ICIR':>9}  {'sign_cons':>9}  {'windows':>7}  ROBUST?")
    print(f"  {'─'*60}")

    wf_rows    = []
    robust_pos = []
    robust_neg = []

    factor_cols = [c for c in panel.columns
                   if c not in ('symbol', 'datetime', 'forward_1d')
                   and panel[c].notna().sum() > 50]

    for fc in factor_cols:
        wf_df      = wf_validate(panel, fc)
        ok, st     = is_robust(wf_df)
        flag       = '✓ YES' if ok else '✗ NO '
        icir_str   = f"{st['median_icir']:+.4f}" if pd.notna(st['median_icir']) else '   N/A'
        cons_str   = f"{st['sign_cons']:.0%}"    if pd.notna(st['sign_cons'])    else '  N/A'
        print(f"  {fc:<28} {icir_str:>9}  {cons_str:>9}  {st['n_windows']:>7}  {flag}")
        wf_rows.append({'factor': fc, 'robust': ok, **st})
        if ok:
            if st['sign'] > 0:
                robust_pos.append(fc)
            else:
                robust_neg.append(fc)

    print(f"\n  Faktor robust (+): {robust_pos}")
    print(f"  Faktor robust (-): {robust_neg}")
    print(f"  (Faktor negatif bisa dipakai dengan membalik tanda saat composite)")

    wf_df_all = pd.DataFrame(wf_rows)
    wf_df_all.to_csv(f'{OUTPUT_DIR}/wf_validation_1d.csv', index=False)
    print(f"\n  ✓ WF results → {OUTPUT_DIR}/wf_validation_1d.csv")
    print("━"*60 + "\n")
    return summary, ic_ts


if __name__ == '__main__':
    import sys
    refresh = '--refresh' in sys.argv
    main(force_refresh=refresh)
