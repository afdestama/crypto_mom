# data_fetcher.py — Fetch 4H OHLCV + Perp Data dari Binance via CCXT

import ccxt
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import config


def get_exchange():
    exchange = ccxt.binanceusdm({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    return exchange


def load_current_universe() -> list:
    """
    Symbols di rolling universe TERBARU (dari cache parquet).
    Dipakai screener untuk membatasi ranking ke coin yang liquid SAAT INI.
    Return [] kalau cache belum ada.
    """
    cache_path = f"{config.CACHE_DIR}/_rolling_universe.parquet"
    if not os.path.exists(cache_path):
        return []
    ru = pd.read_parquet(cache_path)
    if ru.empty:
        return []
    latest = ru['date'].max()
    return ru[ru['date'] == latest]['symbol'].tolist()


def fetch_btc_data(exchange=None, days: int = config.LOOKBACK_DAYS,
                   force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch BTC 4H OHLCV khusus untuk regime detection.
    BTC adalah proxy market regime untuk seluruh crypto.
    """
    if exchange is None:
        exchange = get_exchange()

    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = f"{config.CACHE_DIR}/_btc_regime.parquet"

    if not force_refresh and os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < config.DATA_CACHE_HOURS:
            return pd.read_parquet(cache_path)

    df = fetch_ohlcv(exchange, "BTC/USDT:USDT", days=days)
    if not df.empty:
        df.to_parquet(cache_path)
    return df


# ─────────────────────────────────────────────────────────────
# DYNAMIC UNIVERSE
# ─────────────────────────────────────────────────────────────

def fetch_universe(exchange=None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Rolling universe (anti survivorship bias).

    Untuk setiap tanggal, universe ditentukan dari avg volume 30 hari
    SEBELUMNYA — bukan snapshot volume hari ini. Coin juga harus sudah
    listing minimal MIN_LISTING_DAYS hari.

    Returns: DataFrame kolom [date, symbol]
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = f"{config.CACHE_DIR}/_rolling_universe.parquet"

    if not force_refresh and os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < config.SYMBOL_CACHE_HOURS:
            df = pd.read_parquet(cache_path)
            print(f"  ✓ Rolling universe from cache: "
                  f"{df['date'].nunique()} dates, {df['symbol'].nunique()} unique symbols")
            return df

    if exchange is None:
        exchange = get_exchange()

    print("  Fetching all perpetual candidates...")
    try:
        markets = exchange.load_markets()
    except Exception as e:
        print(f"  ✗ Gagal load markets: {e}")
        return pd.DataFrame(columns=['date', 'symbol'])

    # ── Step 1: Filter basic (perpetual USDT linear, no expiry) ──
    candidates = []
    for symbol, market in markets.items():
        if not (market.get('swap', False) or market.get('future', False)):
            continue
        if market.get('quote') != config.SYMBOL_QUOTE:
            continue
        if not market.get('linear', True):
            continue
        if market.get('expiry') is not None:
            continue
        if symbol in config.EXCLUDE_SYMBOLS:
            continue
        if any(pat in symbol for pat in config.EXCLUDE_PATTERNS):
            continue
        candidates.append(symbol)

    print(f"  Candidates after basic filter: {len(candidates)}")

    # ── Step 2: Daily volume history untuk semua candidates ──
    # Fetch cukup panjang untuk (a) ranking window & (b) cek listing age.
    hist_days = config.LOOKBACK_DAYS + max(30, config.MIN_LISTING_DAYS)
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=hist_days)).strftime('%Y-%m-%dT00:00:00Z')
    )
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()

    print(f"  Fetching ~{hist_days}d daily volume history untuk {len(candidates)} candidates...")
    vol_records = []
    for symbol in tqdm(candidates, desc="Volume history", ncols=80):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, '1d', since=since, limit=hist_days + 10)
            if len(ohlcv) < 30:
                continue
            # Listing-age filter: candle pertama harus >= MIN_LISTING_DAYS lalu
            first_dt = pd.to_datetime(ohlcv[0][0], unit='ms').normalize()
            if (today - first_dt).days < config.MIN_LISTING_DAYS:
                continue
            for candle in ohlcv:
                vol_records.append({
                    'date'      : pd.to_datetime(candle[0], unit='ms').normalize(),
                    'symbol'    : symbol,
                    'volume_usd': float(candle[5]) * float(candle[4]),  # base vol × close
                })
            time.sleep(0.15)
        except Exception:
            continue

    if not vol_records:
        print("  ✗ Tidak ada candidate yang lolos filter listing-age/volume.")
        return pd.DataFrame(columns=['date', 'symbol'])

    vol_df = pd.DataFrame(vol_records)
    print(f"  Symbols passing listing-age (≥{config.MIN_LISTING_DAYS}d): "
          f"{vol_df['symbol'].nunique()}")

    # ── Step 3: Rolling universe per date ──
    # Untuk tiap tanggal: top-N by avg volume 30 hari SEBELUMNYA.
    print("  Building rolling universe per date...")
    dates = sorted(vol_df['date'].unique())
    rolling_records = []
    for date in dates:
        window_start = date - timedelta(days=30)
        window_end   = date - timedelta(days=1)   # exclude hari ini (no look-ahead)
        window_vol = vol_df[(vol_df['date'] >= window_start) &
                            (vol_df['date'] <= window_end)]
        if window_vol.empty:
            continue
        avg_vol = (window_vol.groupby('symbol')['volume_usd']
                   .mean().sort_values(ascending=False))
        avg_vol = avg_vol[avg_vol >= config.MIN_VOLUME_USD]
        for sym in avg_vol.head(config.TOP_N_SYMBOLS).index.tolist():
            rolling_records.append({'date': date, 'symbol': sym})

    rolling_df = pd.DataFrame(rolling_records)
    if rolling_df.empty:
        print(f"  ✗ Tidak ada symbol lolos MIN_VOLUME_USD ${config.MIN_VOLUME_USD:,.0f}.")
        return rolling_df

    rolling_df.to_parquet(cache_path)

    # Ringkasan: universe terbaru
    latest_date = rolling_df['date'].max()
    latest_syms = rolling_df[rolling_df['date'] == latest_date]['symbol'].tolist()
    latest_vol  = (vol_df[vol_df['date'] == latest_date]
                   .set_index('symbol')['volume_usd'])
    print(f"  ✓ Rolling universe: {rolling_df['date'].nunique()} dates, "
          f"avg {rolling_df.groupby('date').size().mean():.1f} symbols/day")
    print(f"\n  Universe terbaru ({latest_date.date()}), top 10:")
    for sym in latest_syms[:10]:
        v = latest_vol.get(sym, float('nan'))
        print(f"    {sym:<28} ${v:>18,.0f}")

    return rolling_df


# ─────────────────────────────────────────────────────────────
# OHLCV — 4H
# ─────────────────────────────────────────────────────────────

def fetch_ohlcv(exchange, symbol: str, days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch 4H OHLCV candles."""
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00Z')
    )
    all_ohlcv = []
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, config.TIMEFRAME, since=since, limit=500)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            if len(ohlcv) < 500:
                break
            since = ohlcv[-1][0] + 1
            time.sleep(exchange.rateLimit / 1000)
        except Exception as e:
            print(f"  OHLCV error {symbol}: {e}")
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates('datetime').set_index('datetime').sort_index()
    df = df[['open','high','low','close','volume']].astype(float)
    return df


# ─────────────────────────────────────────────────────────────
# FUNDING RATE — tiap 8 jam, resample ke 4H (forward fill)
# ─────────────────────────────────────────────────────────────

def fetch_funding_rate(exchange, symbol: str, days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Funding rate update setiap 8 jam.
    Di-resample ke 4H dengan forward fill supaya align dengan OHLCV.
    """
    since = exchange.parse8601(
        (datetime.utcnow() - timedelta(days=days)).strftime('%Y-%m-%dT00:00:00Z')
    )
    try:
        funding = exchange.fetch_funding_rate_history(symbol, since=since, limit=1000)
    except Exception as e:
        print(f"  Funding error {symbol}: {e}")
        return pd.DataFrame()

    if not funding:
        return pd.DataFrame()

    records = [
        {
            'datetime'    : pd.to_datetime(item['timestamp'], unit='ms'),
            'funding_rate': float(item['fundingRate'])
        }
        for item in funding
    ]
    df = pd.DataFrame(records).set_index('datetime').sort_index()

    # Resample ke 4H, forward fill (funding berlaku sampai update berikutnya)
    df = df.resample('4h').last().ffill()
    return df


# ─────────────────────────────────────────────────────────────
# LONG/SHORT RATIO — daily → resample ke 4H
# ─────────────────────────────────────────────────────────────

def fetch_long_short_ratio(exchange, symbol: str, days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch top trader L/S ratio (tersedia dalam 1h atau 4h dari Binance)."""
    records = []
    try:
        raw_symbol = symbol.replace('/USDT:USDT', 'USDT')

        # Coba ambil per 4h dulu
        base   = "https://fapi.binance.com/futures/data/topLongShortPositionRatio"
        params = {'symbol': raw_symbol, 'period': '4h', 'limit': min(days * 6, 500)}
        # ccxt's Exchange.fetch() takes no `params` kwarg — encode into the URL query string
        response = exchange.fetch(base + '?' + exchange.urlencode(params))

        for item in response:
            records.append({
                'datetime'        : pd.to_datetime(int(item['timestamp']), unit='ms'),
                'long_short_ratio': float(item['longShortRatio'])
            })
    except Exception as e:
        print(f"  L/S ratio error {symbol}: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index('datetime').sort_index()
    df = df.resample('4h').last().ffill()
    return df


# ─────────────────────────────────────────────────────────────
# OPEN INTEREST — 4H dari Binance (hanya ~30 hari historis)
# ─────────────────────────────────────────────────────────────

def fetch_open_interest(exchange, symbol: str, days: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """Fetch OI history dari Binance (tersedia ~30 hari terakhir, period=4h)."""
    records = []
    try:
        raw_symbol = symbol.replace('/USDT:USDT', 'USDT')
        base   = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {'symbol': raw_symbol, 'period': '4h', 'limit': min(days * 6, 500)}
        response = exchange.fetch(base + '?' + exchange.urlencode(params))

        for item in response:
            records.append({
                'datetime'     : pd.to_datetime(int(item['timestamp']), unit='ms'),
                'open_interest': float(item['sumOpenInterest']),
            })
    except Exception as e:
        print(f"  OI error {symbol}: {e}")
        return pd.DataFrame()

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).set_index('datetime').sort_index()
    return df.resample('4h').last().ffill()


# ─────────────────────────────────────────────────────────────
# LOAD / FETCH PER SYMBOL
# ─────────────────────────────────────────────────────────────

def load_or_fetch_symbol(symbol: str, exchange=None, force_refresh: bool = False) -> pd.DataFrame:
    """
    OHLCV di-cache (refresh tiap DATA_CACHE_HOURS).
    Funding rate, L/S ratio, OI selalu di-fetch fresh (LIVE_DATA_CACHE_HOURS=0).
    """
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    safe_name  = symbol.replace('/', '_').replace(':', '_')
    cache_path = f"{config.CACHE_DIR}/{safe_name}_4h.parquet"

    if exchange is None:
        exchange = get_exchange()

    # ── OHLCV: pakai cache kalau masih segar ──────────────
    ohlcv_cached = False
    if not force_refresh and os.path.exists(cache_path):
        age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
        if age_hours < config.DATA_CACHE_HOURS:
            df = pd.read_parquet(cache_path)
            # Strip kolom live agar di-replace dengan data segar di bawah
            live_cols = ['funding_rate', 'long_short_ratio', 'open_interest', 'symbol']
            df = df.drop(columns=[c for c in live_cols if c in df.columns])
            ohlcv_cached = True

    if not ohlcv_cached:
        ohlcv = fetch_ohlcv(exchange, symbol)
        if ohlcv.empty:
            return pd.DataFrame()
        df = ohlcv.copy()

    # ── Live data: selalu fetch terbaru ───────────────────
    funding = fetch_funding_rate(exchange, symbol)
    ls      = fetch_long_short_ratio(exchange, symbol)
    oi      = fetch_open_interest(exchange, symbol)

    for extra in [funding, ls, oi]:
        if not extra.empty:
            df = df.join(extra, how='left')

    df = df.ffill(limit=2).bfill(limit=2)
    df['symbol'] = symbol

    df.to_parquet(cache_path)
    return df


# ─────────────────────────────────────────────────────────────
# FETCH ALL
# ─────────────────────────────────────────────────────────────

def fetch_all_symbols(force_refresh: bool = False,
                      force_refresh_universe: bool = False) -> tuple:
    """
    Fetch rolling universe + download 4H data untuk semua symbol unik.

    force_refresh          : paksa re-fetch OHLCV + live data per symbol
    force_refresh_universe : paksa re-scan semua market (lambat, jarang perlu)

    Returns: (data, btc_df)
    """
    exchange = get_exchange()
    rolling_universe = fetch_universe(exchange, force_refresh=force_refresh_universe)

    if rolling_universe.empty:
        print("  ✗ Universe kosong.")
        return {}, pd.DataFrame()

    all_symbols = rolling_universe['symbol'].unique().tolist()

    print(f"\n{'='*55}")
    print(f"  Downloading 4H data: {len(all_symbols)} unique symbols")
    print(f"  Timeframe  : {config.TIMEFRAME}")
    print(f"  Lookback   : {config.LOOKBACK_DAYS} hari ({config.LOOKBACK_DAYS * config.CANDLES_PER_DAY} candles)")
    print(f"  Forward    : {config.FORWARD_PERIOD} candle (~{config.FORWARD_PERIOD * 4} jam)")
    print(f"{'='*55}")

    # BTC untuk regime detection — selalu fresh agar regime akurat
    print("  Fetching BTC regime data...")
    btc_data = fetch_btc_data(exchange, force_refresh=force_refresh)

    data = {}
    for symbol in tqdm(all_symbols, desc="Downloading"):
        try:
            df = load_or_fetch_symbol(symbol, exchange=exchange, force_refresh=force_refresh)
            # Minimum 2× IC rolling window untuk analisis yang valid
            if not df.empty and len(df) >= config.IC_ROLLING_WINDOW * 2:
                data[symbol] = df
        except Exception as e:
            print(f"  Error {symbol}: {e}")
        time.sleep(0.2)

    print(f"\n  ✓ Berhasil: {len(data)} / {len(all_symbols)} symbols")
    return data, btc_data
