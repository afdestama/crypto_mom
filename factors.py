# factors.py — Build all factors untuk 4H intraday swing

import pandas as pd
import numpy as np
import config


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score normalization."""
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / (std + 1e-9)


def detect_regime(btc_df: pd.DataFrame) -> str:
    """
    Volatility-based regime detection dari BTC.
    Returns 'trend' atau 'reversal'.
    trend_strength = |ROC_12c| / realized_vol_12c
    """
    if btc_df is None or btc_df.empty or len(btc_df) < config.REGIME_TREND_WINDOW + 1:
        return 'reversal'

    w   = config.REGIME_TREND_WINDOW
    ret = btc_df['close'].pct_change()
    roc = btc_df['close'].pct_change(w).abs().iloc[-1]
    vol = ret.rolling(w).std().iloc[-1]

    trend_strength = roc / (vol + 1e-9)
    return 'trend' if trend_strength > config.REGIME_TREND_THRESHOLD else 'reversal'


def build_ts_factors(df: pd.DataFrame) -> dict:
    """
    Build semua kandidat TS factors per-coin (z-scored vs own history).
    Semua faktor selalu dibangun; pemilihan mana yang aktif bergantung regime (di screener).
    """
    w   = config.TS_ZSCORE_WINDOW
    c, h, l = df['close'], df['high'], df['low']
    ret = c.pct_change()

    # ── TREND category ─────────────────────────────────────
    out = {
        'ts_momentum_follow': zscore(c.pct_change(12), window=w),
    }

    # ── REVERSAL category ──────────────────────────────────
    sgn    = np.sign(ret).fillna(0)
    grp    = (sgn != sgn.shift()).cumsum()
    streak = sgn.groupby(grp).cumcount() + 1
    rng    = (h.rolling(18).max() - l.rolling(18).min()).replace(0, np.nan)
    stoch  = (c - l.rolling(18).min()) / rng

    out.update({
        'ts_streak_rev': zscore(-(streak * sgn), window=w),
        'ts_stoch_rev' : zscore(-(stoch - 0.5), window=w),
        'ts_clv'       : zscore(((c - l) - (h - c)) / ((h - l) + 1e-9).rolling(3).mean(), window=w),
    })

    # ── SENTIMENT overlay (selalu dibangun) ────────────────
    if 'funding_rate' in df.columns:
        fr = df['funding_rate'].fillna(0)
        out['ts_funding_contra'] = zscore(-fr, window=w)

    if 'long_short_ratio' in df.columns:
        out['ts_ls_contra'] = zscore(-df['long_short_ratio'], window=w)

    # OI: gunakan real OI jika tersedia (>=30 baris), fallback ke volume proxy
    if 'open_interest' in df.columns and df['open_interest'].notna().sum() >= 30:
        out['ts_oi_buildup'] = zscore(df['open_interest'].pct_change(6), window=w)
    else:
        vol_fast = df['volume'].ewm(span=6).mean()
        vol_slow = df['volume'].ewm(span=24).mean()
        out['ts_oi_buildup'] = zscore((vol_fast - vol_slow) / (vol_slow + 1e-9), window=w)

    return out


def build_factors(df: pd.DataFrame, btc_df: pd.DataFrame = None,
                  forward_period: int = None) -> pd.DataFrame:
    """
    Build semua factor untuk satu symbol (4H candles).

    Kolom wajib  : open, high, low, close, volume
    Kolom opsional: funding_rate, long_short_ratio
    btc_df       : BTC OHLCV untuk regime detection (opsional)
    """
    f = pd.DataFrame(index=df.index)
    f['symbol'] = df['symbol'] if 'symbol' in df.columns else ''

    # ── Base return per candle ─────────────────────────────
    f['return_1c'] = df['close'].pct_change()   # return 1 candle (4 jam)

    # ── FACTOR 1: reversal_1c ──────────────────────────────
    # Candle sebelumnya terlalu ekstrem → ekspektasi berbalik
    # Z-score dari negatif return candle lalu
    raw_rev          = -1 * f['return_1c']
    f['reversal_1d'] = zscore(raw_rev, window=config.REVERSAL_WINDOW)

    # ── FACTOR 2: liquidity_30 ─────────────────────────────
    # Volume relatif vs rolling window — proxy liquidity & interest
    f['liquidity_30'] = zscore(df['volume'], window=config.LIQUIDITY_WINDOW)

    # ── FACTOR 3: liquidation_imbalance ────────────────────
    # Proxy: body direction × volume z-score
    # Positif = candle bullish kuat + volume tinggi → short squeeze / bullish pressure
    candle_range       = df['high'] - df['low'] + 1e-9
    body_dir           = (df['close'] - df['open']) / candle_range
    vol_z              = zscore(df['volume'], window=30)
    raw_imb            = body_dir * vol_z.clip(-3, 3)

    # Jika ada data real liquidation (dari Coinglass):
    if 'liq_long' in df.columns and 'liq_short' in df.columns:
        total_liq  = df['liq_short'] + df['liq_long'] + 1e-9
        raw_imb    = (df['liq_short'] - df['liq_long']) / total_liq

    f['liq_imbalance'] = zscore(raw_imb, window=config.LIQ_IMBALANCE_WINDOW)

    # ── FACTOR 4: Momentum ROC ─────────────────────────────
    # ROC_WINDOWS sudah dalam satuan candle (42=7d, 84=14d, 180=30d)
    for w in config.ROC_WINDOWS:
        days_approx    = w // config.CANDLES_PER_DAY
        f[f'roc_{days_approx}d'] = zscore(df['close'].pct_change(w), window=config.IC_ROLLING_WINDOW)

    # ── FACTOR 5: Intraday Momentum (pendek) ──────────────
    # Momentum 3 candle = 12 jam — lebih relevan untuk intraday
    f['roc_12h'] = zscore(df['close'].pct_change(3), window=60)
    # Momentum 6 candle = 1 hari
    f['roc_1d']  = zscore(df['close'].pct_change(6), window=60)

    # ── FACTOR 6: Funding Rate Signal ─────────────────────
    if 'funding_rate' in df.columns:
        # Contrarian: funding sangat positif = terlalu banyak long = bearish
        f['funding_signal'] = -1 * zscore(df['funding_rate'], window=config.FUNDING_WINDOW)

        # Funding momentum: funding_rate makin naik = warning signal
        f['funding_mom'] = -1 * zscore(
            df['funding_rate'].diff(2),  # perubahan 2 candle = 8 jam
            window=config.FUNDING_WINDOW
        )

        # Funding extreme: magnitude funding (kedua arah) → potensi mean-reversion
        # Sign -1 supaya konvensi bullish-when-high konsisten (lihat funding_signal).
        f['funding_extreme'] = -1 * zscore(df['funding_rate'].abs(),
                                           window=config.FUNDING_WINDOW)
    else:
        f['funding_signal']  = np.nan
        f['funding_mom']     = np.nan
        f['funding_extreme'] = np.nan

    # ── FACTOR 7: Volume proxies (pengganti OI) ───────────
    # Binance OI history hanya ~30 hari, jadi pakai proxy berbasis volume.
    vol_chg              = df['volume'].pct_change()

    # Volume Momentum (proxy OI accumulation):
    # volume naik konsisten = akumulasi → mirip OI naik
    f['vol_momentum']    = zscore(vol_chg, window=60)

    # Volume × Price Direction (proxy OI conviction):
    # volume besar + candle naik = smart money masuk (bullish)
    price_dir            = np.sign(f['return_1c'])
    f['vol_conviction']  = zscore(vol_chg * price_dir, window=60)

    # Abnormal Volume Spike (proxy OI spike):
    # volume 2× rata-rata = event besar = potensi volatilitas
    f['vol_spike']       = zscore(vol_chg.abs(), window=60)

    # ── FACTOR 8: Long/Short Ratio ────────────────────────
    if 'long_short_ratio' in df.columns:
        # Contrarian: terlalu banyak long = bearish
        f['ls_signal'] = -1 * zscore(df['long_short_ratio'], window=60)
    else:
        f['ls_signal'] = np.nan

    # ── FACTOR 9: Volatility Regime ───────────────────────
    # Realized vol 12 candle = 2 hari
    realized_vol        = f['return_1c'].rolling(12).std()
    f['inv_volatility'] = -1 * zscore(realized_vol, window=60)

    # ── FACTOR 10: Price vs Weekly VWAP (true volume-weighted) ────
    # 42 candle 4H ≈ 7 hari. Σ(close×volume)/Σ(volume), bukan HLC/3 proxy.
    # Harga di bawah VWAP → potensi reversal naik.
    pv                  = df['close'] * df['volume']
    wvwap               = pv.rolling(42).sum() / (df['volume'].rolling(42).sum() + 1e-9)
    pv_signal           = -(df['close'] - wvwap) / (wvwap + 1e-9)
    f['price_vs_vwap']  = zscore(pv_signal, window=config.TS_ZSCORE_WINDOW)

    # ── TS FACTORS (per-coin, adaptive — pilihan aktif bergantung regime) ──
    for col, series in build_ts_factors(df).items():
        f[col] = series

    # Raw values untuk decision engine (nilai asli, bukan z-score)
    f['funding_rate_raw'] = df['funding_rate']     if 'funding_rate'     in df.columns else np.nan
    f['ls_ratio_raw']     = df['long_short_ratio'] if 'long_short_ratio' in df.columns else np.nan
    f['oi_raw']           = df['open_interest']    if 'open_interest'    in df.columns else np.nan

    # ── REGIME FACTOR (BTC sebagai proxy market) ───────────
    if btc_df is not None and not btc_df.empty:
        btc_aligned = btc_df['close'].reindex(df.index, method='ffill')
        btc_ma      = btc_aligned.rolling(config.BTC_REGIME_MA, min_periods=10).mean()
        # 1 = bull (close > MA), 0 = bear/ranging
        f['btc_regime'] = (btc_aligned > btc_ma).astype(float)
        # Trend strength: jarak close dari MA, dinormalisasi
        f['btc_trend_strength'] = (
            (btc_aligned - btc_ma) / (btc_ma + 1e-9)
        ).rolling(12).mean()
    else:
        f['btc_regime']         = 1.0
        f['btc_trend_strength'] = 0.0

    # ── Forward Return Target ──────────────────────────────
    # Cumulative return dari entry(t) → exit(t+FORWARD_PERIOD), bukan return
    # satu candle terpencil. FORWARD_PERIOD=3 candle = ~12 jam.
    _fp = forward_period if forward_period is not None else config.FORWARD_PERIOD
    f['forward_1d'] = df['close'].shift(-_fp) / df['close'] - 1

    return f


def build_panel(data: dict, btc_df: pd.DataFrame = None,
                forward_period: int = None) -> pd.DataFrame:
    """Gabungkan semua symbol menjadi panel DataFrame."""
    panels = []
    for symbol, df in data.items():
        fdf          = build_factors(df, btc_df=btc_df, forward_period=forward_period)
        fdf['symbol'] = symbol
        panels.append(fdf)

    panel = pd.concat(panels)

    # Normalize index ke kolom 'datetime'
    if panel.index.name in ['datetime', 'date']:
        panel = panel.reset_index().rename(
            columns={panel.index.name: 'datetime'}
        )
    elif 'datetime' not in panel.columns:
        panel = panel.reset_index().rename(columns={'index': 'datetime'})

    panel['datetime'] = pd.to_datetime(panel['datetime'])
    return panel.sort_values(['datetime', 'symbol']).reset_index(drop=True)


# Semua factor columns untuk IC analysis (cross-sectional, unchanged)
FACTOR_COLS = [
    'reversal_1d',
    'liquidity_30',
    'liq_imbalance',
    'roc_7d',
    'roc_14d',
    'roc_30d',
    'roc_12h',
    'roc_1d',
    'funding_signal',
    'funding_mom',
    'funding_extreme',
    'vol_momentum',
    'vol_conviction',
    'vol_spike',
    'ls_signal',
    'inv_volatility',
    'price_vs_vwap',
]

# TS factor cols per regime — diupdate setelah menjalankan factor_research_wf.py
# Faktor SENTIMENT selalu dimasukkan di kedua regime
TS_FACTOR_COLS = {
    'trend'   : [],
    'reversal': ['ts_streak_rev', 'ts_clv'],
}
